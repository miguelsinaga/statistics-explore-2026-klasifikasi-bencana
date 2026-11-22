"""Ekstrak fitur DINOv3 ViT-L (backbone DIBEKUKAN) untuk data latih dan uji.

Fitur dipakai di tiga tempat:
  - build_events.py   : klaster kejadian (kemiripan cosine) untuk fold yang jujur
  - rangkum_oof.py    : kepala logistik 3-arah dan ordinal (OOF)
  - predict_all.py    : kepala yang sama, dilatih pada seluruh data latih

Pakai:
    python scripts/dinov3_features.py          # sekali saja, ~10 menit di RTX 3050
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import se_lib as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "big-data-competition-statistics-explore-2026" / "SE"
CACHE = ROOT / "cache_640"
OUT = ROOT / "output_dino"
FOLDS = ROOT / "output_local" / "train_folds.csv"


class PlainDS(Dataset):
    """Hanya baca gambar + transform timm. Tidak ada augmentasi - fitur harus deterministik."""

    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def extract(model_name, batch, workers, img_size=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FOLDS)
    test_files = L.list_test_files(DATA / "TEST")
    test_cpaths = [str(CACHE / "test" / f"{f.stem}.jpg") for f in test_files]

    print(f"memuat {model_name} (dibekukan)...", flush=True)
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(dev)
    cfg = timm.data.resolve_data_config({}, model=model)
    if img_size:
        cfg["input_size"] = (3, img_size, img_size)
        cfg["crop_pct"] = 1.0
    print(f"  input {cfg['input_size']}, mean {np.round(cfg['mean'],3)}")
    tf = timm.data.create_transform(**cfg, is_training=False)
    dim = model.num_features
    print(f"  dimensi fitur: {dim}")

    for nama, paths in (("train", df.cpath.tolist()), ("test", test_cpaths)):
        suf = f"_{img_size}" if img_size else ""
        dst = OUT / f"feat_{nama}{suf}.npy"
        if dst.exists():
            print(f"{nama}: {dst.name} sudah ada, dilewati")
            continue
        dl = DataLoader(PlainDS(paths, tf), batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True)
        feats, t0 = [], time.time()
        with torch.no_grad():
            for i, x in enumerate(dl, 1):
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                    f = model(x.to(dev, non_blocking=True))
                feats.append(f.float().cpu().numpy())
                done = i * batch
                if i % 40 == 0 or i == len(dl):
                    el = time.time() - t0
                    print(f"  {nama} {min(done,len(paths))}/{len(paths)}  "
                          f"({el:.0f}s, {done/max(el,1e-9):.0f} img/s)", flush=True)
        arr = np.concatenate(feats)[:len(paths)]
        np.save(dst, arr)
        print(f"{nama}: tersimpan {arr.shape} -> {dst.name}")

    del model
    if dev == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    extract(a.model, a.batch, a.workers)


if __name__ == "__main__":
    main()
