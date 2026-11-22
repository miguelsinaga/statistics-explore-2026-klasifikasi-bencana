"""Kumpulkan OOF semua cabang pipeline, bandingkan, simpan ke oof_semua.npz.

Cabang:
  - Anchor global: EfficientNet-B0 @224/@320, ConvNeXt-T @224
  - DINOv3 beku: kepala 3-arah dan kepala ordinal
  - Spesialis per jenis bencana (_cnj1/2/3): tiga ConvNeXt-T digabung jadi satu
    "model" - tiap gambar dinilai oleh spesialis jenisnya
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "output_effnet"
TAG = [("", "EffNet B0@224"), ("_r320", "EffNet B0@320"), ("_cnx", "ConvNeXt-T@224")]
VARIAN = [("_cnj", "per jenis")]


def lin():
    return LogisticRegression(max_iter=2000, C=0.03)


def prob_ordinal(Xtr, ytr, Xte):
    p1 = lin().fit(Xtr, (ytr > 1).astype(int)).predict_proba(Xte)[:, 1]
    p2 = lin().fit(Xtr, (ytr > 2).astype(int)).predict_proba(Xte)[:, 1]
    p2 = np.minimum(p2, p1)
    return np.c_[1 - p1, p1 - p2, p2]


def jenis_dasar(J):
    """Aturan pengarah ke spesialis: rata-rata probabilitas jenis model dasar.
    Aturan yang SAMA dipakai di predict_all.py untuk data uji."""
    return (sum(J) / len(J)).argmax(1) + 1


def main():
    df = pd.read_csv(ROOT / "output_local" / "train_events.csv")
    y, jen = df.kerusakan.values, df.jenis.values
    folds = np.load(O / "folds_event.npy")

    nama, J, K = [], [], []
    for tag, nm in TAG:
        ada = [f for f in range(5) if (O / f"oof{tag}_fold{f}.npz").exists()]
        if len(ada) < 5:
            print(f"  lewati {nm}: baru {len(ada)}/5 fold")
            continue
        a = np.zeros((len(df), 3)); b = np.zeros((len(df), 3))
        for f in ada:
            d = np.load(O / f"oof{tag}_fold{f}.npz")
            a[d["idx"]] = d["pj"]; b[d["idx"]] = d["pk"]
        nama.append(nm); J.append(a); K.append(b)

    X = StandardScaler().fit_transform(np.load(ROOT / "output_dino" / "feat_train.npy"))
    for nm, fn in (("DINOv3 3-arah", None), ("DINOv3 ordinal", prob_ordinal)):
        a = np.zeros((len(df), 3)); b = np.zeros((len(df), 3))
        for f in range(5):
            tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
            a[va] = lin().fit(X[tr], jen[tr]).predict_proba(X[va])
            b[va] = (lin().fit(X[tr], y[tr]).predict_proba(X[va]) if fn is None
                     else fn(X[tr], y[tr], X[va]))
        nama.append(nm); J.append(a); K.append(b)

    # ------------------------------------------------ spesialis per jenis
    pjd = jenis_dasar(J)                      # dihitung dari model DASAR saja
    pj_nama = []
    for tag, nm in VARIAN:
        if not all((O / f"oof{tag}{j}_fold{f}.npz").exists()
                   for j in (1, 2, 3) for f in range(5)):
            print(f"  lewati ConvNeXt-T {nm}: checkpoint belum lengkap")
            continue
        b = np.full((len(df), 3), 1 / 3)
        for j in (1, 2, 3):
            for f in range(5):
                d = np.load(O / f"oof{tag}{j}_fold{f}.npz")
                b[d["idx"]] = d["pk"]
        # Konservatif: di data uji gambar diarahkan memakai jenis PREDIKSI, jadi
        # baris yang jenisnya salah ditebak tidak boleh ikut dibantu di OOF.
        salah = pjd != jen
        b[salah] = 1 / 3
        full = f"ConvNeXt-T {nm}"
        print(f"  {full}: {int(salah.sum())} baris jenis salah-tebak dinetralkan")
        # Sumbu jenis tidak dinilai model ini: nilai rata tidak mengubah argmax.
        nama.append(full); J.append(np.full((len(df), 3), 1 / 3)); K.append(b)
        pj_nama.append(full)

    print(f"\n{'model':>28} | {'_jenis':>8} | {'_kerusakan':>11}")
    print("-" * 54)
    for nm, a, b in zip(nama, J, K):
        u = (a.argmax(1) + 1 == jen).mean()
        us = f"{u:>8.4f}" if "per jenis" not in nm else f"{'-':>8}"
        print(f"{nm:>28} | {us} | {(b.argmax(1)+1 == y).mean():>11.4f}")
    print("-" * 54)

    if pj_nama and "ConvNeXt-T@224" in nama:
        print("\nSPESIALIS vs ConvNeXt-T GLOBAL (baris yang sama)")
        g = K[nama.index("ConvNeXt-T@224")]
        for j, nj in ((1, "BANJIR"), (2, "GEMPA"), (3, "KEBAKARAN")):
            m = jen == j
            baris = f"  {nj:>10}: global {(g[m].argmax(1)+1 == y[m]).mean():.4f}"
            for full in pj_nama:
                s_ = K[nama.index(full)]
                baris += f" | {full.split('per jenis')[1].strip() or 'khusus'} " \
                         f"{(s_[m].argmax(1)+1 == y[m]).mean():.4f}"
            print(baris)

    np.savez(O / "oof_semua.npz", nama=np.array(nama),
             **{f"{i}_{ax}": (J[i] if ax == "J" else K[i])
                for i in range(len(nama)) for ax in "JK"})
    print(f"\ndisimpan: {len(nama)} model")


if __name__ == "__main__":
    main()
