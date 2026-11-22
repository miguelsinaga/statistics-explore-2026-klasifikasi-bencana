"""Fine-tune CNN hierarchical (EfficientNet-B0 / ConvNeXt-Tiny) dengan Mixup/CutMix.

Kerangka:
  Preprocessing   - resize 224, normalisasi statistik ImageNet
  Augmentasi      - RandomResizedCrop, flip, rotasi, brightness/contrast/saturation
  Transfer learn  - EfficientNet-B0 / ConvNeXt-Tiny pretrained (timm, Apache-2.0)
  Hierarchical    - Tahap 1 kepala jenis; Tahap 2 kepala kerusakan yang DIBERI TAHU
                    hasil tahap 1 (softmax jenis disambung ke fitur)
  Regularisasi    - Mixup/CutMix, label smoothing, weight decay, dropout
  Validasi        - StratifiedGroupKFold pada level KEJADIAN (bukan near-duplicate),
                    supaya skor validasi tidak diisi hafalan adegan yang sama

Label internal memakai konvensi train (1=RINGAN, 2=SEDANG, 3=BERAT). Konversi ke
konvensi penomoran panitia dilakukan sekali saat menulis submission.

Jalankan:  python scripts/train_effnet.py --folds 0,1,2,3,4 [--arch ..] [--size ..] [--tag ..]
           --jenis J  melatih spesialis yang hanya melihat satu jenis bencana
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output_effnet"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------------------------------------------------------- data
class SEData(Dataset):
    """Baca dari cache 640px, augmentasi, keluarkan tensor ternormalisasi."""

    def __init__(self, paths, jenis, kerusakan, size, train):
        self.paths, self.jenis, self.kerusakan = paths, jenis, kerusakan
        self.size, self.train = size, train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        im = self._aug(im, self.size) if self.train else im.resize(
            (self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, np.uint8).copy()).permute(2, 0, 1).float() / 255
        return (x - MEAN) / STD, self.jenis[i] - 1, self.kerusakan[i] - 1

    @staticmethod
    def _aug(im, s):
        w, h = im.size
        for _ in range(10):                                        # RandomResizedCrop
            a = random.uniform(0.5, 1.0) * w * h
            r = random.uniform(3 / 4, 4 / 3)
            cw, ch = int(round((a * r) ** 0.5)), int(round((a / r) ** 0.5))
            if cw <= w and ch <= h:
                x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
                im = im.crop((x0, y0, x0 + cw, y0 + ch))
                break
        im = im.resize((s, s), Image.BILINEAR)
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:                                  # rotasi
            im = im.rotate(random.uniform(-15, 15), Image.BILINEAR, fillcolor=(124, 116, 104))
        if random.random() < 0.7:                                  # brightness/contrast/sat
            for E in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                im = E(im).enhance(random.uniform(0.75, 1.3))
        return im


# -------------------------------------------------------------------------- model
class HierNet(nn.Module):
    """Tahap 1: jenis. Tahap 2: kerusakan, diberi tahu keluaran tahap 1."""

    def __init__(self, arch="efficientnet_b0", drop=0.3):
        super().__init__()
        self.trunk = timm.create_model(arch, pretrained=True, num_classes=0, drop_rate=drop)
        d = self.trunk.num_features
        self.head_j = nn.Linear(d, 3)
        self.head_k = nn.Sequential(nn.Dropout(drop), nn.Linear(d + 3, 3))

    def forward(self, x):
        f = self.trunk(x)
        lj = self.head_j(f)
        return lj, self.head_k(torch.cat([f, lj.softmax(1)], 1))


def mixup_cutmix(x, yj, yk, alpha=0.4, p_cut=0.5):
    """Satu batch dicampur; label ikut tercampur untuk KEDUA sumbu."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < p_cut:                                   # CutMix
        H, W = x.shape[2:]
        rh, rw = int(H * (1 - lam) ** 0.5), int(W * (1 - lam) ** 0.5)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x[:, :, y0:y1, x0:x1] = x[idx][:, :, y0:y1, x0:x1]
        lam = 1 - ((y1 - y0) * (x1 - x0) / (H * W))
    else:                                                          # Mixup
        x = lam * x + (1 - lam) * x[idx]
    return x, (yj, yj[idx]), (yk, yk[idx]), lam


def crit(logit, pair, lam, ls=0.1):
    a, b = pair
    return lam * F.cross_entropy(logit, a, label_smoothing=ls) + \
        (1 - lam) * F.cross_entropy(logit, b, label_smoothing=ls)


def make_folds(df, n=5, seed=42):
    lab9 = (df.jenis.values - 1) * 3 + (df.kerusakan.values - 1)
    f = np.zeros(len(df), int)
    for k, (_, v) in enumerate(StratifiedGroupKFold(n, shuffle=True, random_state=seed)
                               .split(df, lab9, df.event.values)):
        f[v] = k
    return f


# ----------------------------------------------------------------------------- run
def run_fold(df, folds, fo, args, dev):
    tr, va = np.where(folds != fo)[0], np.where(folds == fo)[0]
    if args.jenis:
        # Model khusus satu jenis bencana: latih DAN validasi hanya pada jenis itu.
        # Fold tetap sama dengan model global, jadi hasilnya bisa dibandingkan
        # baris per baris.
        tr = tr[df.jenis.values[tr] == args.jenis]
        va = va[df.jenis.values[va] == args.jenis]
        print(f"  fold{fo}: khusus jenis {args.jenis} - latih {len(tr)}, "
              f"validasi {len(va)}", flush=True)
    def mk(i, t):
        return SEData(df.cpath.values[i], df.jenis.values[i],
                      df.kerusakan.values[i], args.size, t)

    dl_tr = DataLoader(mk(tr, True), args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(mk(va, False), args.bs * 2, shuffle=False, num_workers=2)

    net = HierNet(args.arch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, pct_start=0.25,
                                              total_steps=args.epochs * len(dl_tr))
    scaler = torch.amp.GradScaler("cuda")
    best, best_state = -1.0, None

    for ep in range(args.epochs):
        net.train(); t0 = time.time(); tot = 0.0
        for bi, (x, yj, yk) in enumerate(dl_tr):
            x, yj, yk = x.to(dev, non_blocking=True), yj.to(dev), yk.to(dev)
            x, tj, tk, lam = mixup_cutmix(x, yj, yk)
            with torch.amp.autocast("cuda"):
                lj, lk = net(x)
                loss = crit(lj, tj, lam) + crit(lk, tk, lam)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item()
            if bi % 50 == 0:
                print(f"  fold{fo} ep{ep+1}/{args.epochs} step {bi}/{len(dl_tr)} "
                      f"loss {tot/(bi+1):.4f}", flush=True)

        net.eval(); PJ, PK = [], []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for x, _, _ in dl_va:
                lj, lk = net(x.to(dev, non_blocking=True))
                PJ.append(lj.float().softmax(1).cpu()); PK.append(lk.float().softmax(1).cpu())
        pj = torch.cat(PJ).argmax(1).numpy() + 1
        pk = torch.cat(PK).argmax(1).numpy() + 1
        aj = (pj == df.jenis.values[va]).mean()
        ak = (pk == df.kerusakan.values[va]).mean()
        mf = (aj + ak) / 2
        print(f"fold{fo} ep{ep+1}: loss {tot/len(dl_tr):.4f} | jenis {aj:.4f} | "
              f"kerusakan {ak:.4f} | microF1 {mf:.4f} | {time.time()-t0:.0f}s", flush=True)
        if mf > best:
            best = mf
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            np.savez(OUT / f"oof{args.tag}_fold{fo}.npz", idx=va, pj=torch.cat(PJ).numpy(),
                     pk=torch.cat(PK).numpy())
    torch.save(best_state, OUT / f"effnet{args.tag}_fold{fo}.pth")
    print(f"fold{fo} SELESAI - microF1 terbaik {best:.4f}", flush=True)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="0", help="mis. 0 atau 0,1,2,3,4")
    p.add_argument("--arch", default="efficientnet_b0")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--tag", default="", help="akhiran nama file keluaran")
    p.add_argument("--jenis", type=int, default=0,
                   help="latih model khusus satu jenis (1=BANJIR 2=GEMPA 3=KEBAKARAN)")
    a = p.parse_args()

    OUT.mkdir(exist_ok=True)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(ROOT / "output_local" / "train_events.csv")
    folds = make_folds(df)
    np.save(OUT / "folds_event.npy", folds)  # fold identik di semua percobaan
    print(f"device {dev} | {len(df)} gambar | {df.event.nunique()} kejadian | "
          f"{a.arch} @ {a.size}px bs{a.bs} ep{a.epochs}", flush=True)

    hasil = {}
    for fo in [int(x) for x in a.folds.split(",")]:
        if (OUT / f"effnet{a.tag}_fold{fo}.pth").exists():
            print(f"fold{fo} sudah ada checkpoint-nya, dilewati", flush=True)
            continue
        hasil[fo] = run_fold(df, folds, fo, a, dev)
    (OUT / f"hasil{a.tag}.json").write_text(json.dumps(hasil, indent=2))
    print("\nringkasan:", {k: round(v, 4) for k, v in hasil.items()})


if __name__ == "__main__":
    main()
