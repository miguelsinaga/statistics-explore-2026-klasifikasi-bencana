"""Prediksi data uji dari seluruh model, digabung dengan bobot hasil cari_bobot.py.

TTA: flip horizontal x dua skala (ukuran latih dan 1.15x). Bobot penggabungan
diambil apa adanya dari pencarian di validasi silang data latih - tidak ada satu
pun keputusan di sini yang memakai label data uji.

Urutan model HARUS sama dengan rangkum_oof.py: CNN global, DINOv3, lalu model
khusus per jenis (kalau lengkap).

Jalankan: python scripts/predict_all.py [--out NAMA.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import se_lib as L  # noqa: E402
from rangkum_oof import jenis_dasar  # noqa: E402
from train_effnet import MEAN, STD, HierNet  # noqa: E402

O = ROOT / "output_effnet"
ALF = {1: 2, 2: 3, 3: 1}
NJ = {1: "BANJIR", 2: "GEMPA BUMI", 3: "KEBAKARAN"}
NK = {1: "RINGAN", 2: "SEDANG", 3: "BERAT"}
SPEC = [("", "efficientnet_b0", 224),
        ("_r320", "efficientnet_b0", 320),
        ("_cnx", "convnext_tiny", 224)]


class TestData(Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, np.uint8).copy()).permute(2, 0, 1).float() / 255
        return (x - MEAN) / STD


def prediksi_cnn(paths, tag, arch, size, dev):
    """Rata-rata 5 fold x flip x 2 skala."""
    PJ = np.zeros((len(paths), 3))
    PK = np.zeros((len(paths), 3))
    ck = sorted(O.glob(f"effnet{tag}_fold*.pth"))
    skala = [size, int(size * 1.15) // 32 * 32]
    for c in ck:
        net = HierNet(arch).to(dev)
        net.load_state_dict(torch.load(c, map_location=dev))
        net.eval()
        for s in skala:
            dl = DataLoader(TestData(paths, s), 32, shuffle=False, num_workers=2)
            aj, ak = [], []
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for x in dl:
                    x = x.to(dev, non_blocking=True)
                    lj, lk = net(x)
                    lj2, lk2 = net(torch.flip(x, [3]))
                    aj.append(((lj.float().softmax(1) + lj2.float().softmax(1)) / 2).cpu())
                    ak.append(((lk.float().softmax(1) + lk2.float().softmax(1)) / 2).cpu())
            PJ += torch.cat(aj).numpy()
            PK += torch.cat(ak).numpy()
    n = len(ck) * len(skala)
    print(f"  {tag or '(awal)':>8}: {len(ck)} fold x {len(skala)} skala x flip", flush=True)
    return PJ / n, PK / n


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv \
        else "submission.csv"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    DATA = ROOT / "big-data-competition-statistics-explore-2026" / "SE"
    files = L.list_test_files(DATA / "TEST")
    ids = [f.stem for f in files]
    cache = ROOT / "cache_640" / "test"
    paths = [str(cache / f"{i}.jpg") if (cache / f"{i}.jpg").exists() else str(p)
             for i, p in zip(ids, files)]

    print("prediksi data uji, TTA flip x 2 skala:")
    TJ, TK = [], []
    for tag, arch, size in SPEC:
        if len(list(O.glob(f"effnet{tag}_fold*.pth"))) < 5:
            print(f"  lewati {tag or '(awal)'}: checkpoint belum lengkap")
            continue
        j, k = prediksi_cnn(paths, tag, arch, size, dev)
        TJ.append(j); TK.append(k)

    # DINOv3 beku - dilatih di seluruh data latih
    df = pd.read_csv(ROOT / "output_local" / "train_events.csv")
    sc = StandardScaler().fit(np.load(ROOT / "output_dino" / "feat_train.npy"))
    X = sc.transform(np.load(ROOT / "output_dino" / "feat_train.npy"))
    Xt = sc.transform(np.load(ROOT / "output_dino" / "feat_test.npy"))
    lin = lambda: LogisticRegression(max_iter=2000, C=0.03)  # noqa: E731
    jen, y = df.jenis.values, df.kerusakan.values
    for nm, ordin in (("DINOv3 3-arah", False), ("DINOv3 ordinal", True)):
        TJ.append(lin().fit(X, jen).predict_proba(Xt))
        if ordin:
            p1 = lin().fit(X, (y > 1).astype(int)).predict_proba(Xt)[:, 1]
            p2 = lin().fit(X, (y > 2).astype(int)).predict_proba(Xt)[:, 1]
            p2 = np.minimum(p2, p1)
            TK.append(np.c_[1 - p1, p1 - p2, p2])
        else:
            TK.append(lin().fit(X, y).predict_proba(Xt))
        print(f"  {nm}: selesai")

    # spesialis per jenis - diarahkan dengan aturan yang sama seperti di OOF.
    # pjd dihitung dari model DASAR saja, sebelum spesialis ikut ditambahkan.
    pjd = jenis_dasar(TJ)
    for tag, nm in (("_cnj", "per jenis"),):
        if not all(len(list(O.glob(f"effnet{tag}{j}_fold*.pth"))) == 5
                   for j in (1, 2, 3)):
            print(f"  lewati ConvNeXt-T {nm}: checkpoint belum lengkap")
            continue
        K = np.full((len(paths), 3), 1 / 3)
        for j in (1, 2, 3):
            _, kj = prediksi_cnn(paths, f"{tag}{j}", "convnext_tiny", 224, dev)
            K[pjd == j] = kj[pjd == j]
        TJ.append(np.full((len(paths), 3), 1 / 3)); TK.append(K)
        print(f"  ConvNeXt-T {nm}: selesai")

    b = np.load(O / "bobot.npz", allow_pickle=True)
    wj, wk = b["wj"], b["wk"]
    assert len(wj) == len(TJ), "jumlah model tidak cocok dengan bobot"
    pj = sum(w * P for w, P in zip(wj, TJ)).argmax(1) + 1
    pk = sum(w * P for w, P in zip(wk, TK)).argmax(1) + 1
    pk_out = np.array([ALF[v] for v in pk])

    sol, _ = L.read_solution(next(DATA.rglob("*olution*.csv")))
    sub = L.build_submission(sol, ids, pj, pk_out)
    L.verify_submission(sub, sol)
    dest = ROOT / "output_final" / out
    dest.parent.mkdir(exist_ok=True)
    L.write_submission(sub, dest, sep=",")
    np.savez(O / "test_probs_all.npz", ids=ids,
             **{f"{i}_{a}": v for i, (jj, kk) in enumerate(zip(TJ, TK))
                for a, v in (("J", jj), ("K", kk))})

    print("\nsebaran prediksi:")
    print("  _jenis    : " + "  ".join(
        f"{NJ[v]}={v} {float((pj == v).mean()):.3f}" for v in (1, 2, 3)))
    print("  _kerusakan: " + "  ".join(
        f"{NK[k]}={ALF[k]} {float((pk == k).mean()):.3f}" for k in (1, 2, 3)))
    print(f"\nditulis: {dest}")


if __name__ == "__main__":
    main()
