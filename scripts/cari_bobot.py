"""Cari bobot penggabungan model, dinilai HANYA pada validasi silang data latih.

Bobot non-negatif berjumlah 1, dicari acak (Dirichlet) lalu diperbaiki setempat.
Sumbu jenis dan kerusakan dicari terpisah karena kekuatan tiap model berbeda
pada keduanya.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "output_effnet"


def cari(P, t, iters=4000, seed=0):
    rng = np.random.default_rng(seed)
    best = (0.0, None)
    for _ in range(iters):
        w = rng.dirichlet(np.ones(len(P)) * 0.7)
        a = (sum(wi * Pi for wi, Pi in zip(w, P)).argmax(1) + 1 == t).mean()
        if a > best[0]:
            best = (a, w)
    a, w = best
    for _ in range(300):                                   # perbaikan setempat
        w2 = np.clip(w + rng.normal(0, 0.05, len(w)), 0, None)
        w2 /= w2.sum()
        a2 = (sum(wi * Pi for wi, Pi in zip(w2, P)).argmax(1) + 1 == t).mean()
        if a2 > a:
            a, w = a2, w2
    return a, w


def main():
    df = pd.read_csv(ROOT / "output_local" / "train_events.csv")
    y, jen = df.kerusakan.values, df.jenis.values
    z = np.load(O / "oof_semua.npz", allow_pickle=True)
    nama = [str(v) for v in z["nama"]]
    J = [z[f"{i}_J"] for i in range(len(nama))]
    K = [z[f"{i}_K"] for i in range(len(nama))]

    aj, wj = cari(J, jen, seed=1)
    ak, wk = cari(K, y, seed=2)

    print("BOBOT GABUNGAN (dicari pada validasi silang data latih)")
    print(f"{'model':>24} | {'_jenis':>8} | {'_kerusakan':>11}")
    print("-" * 50)
    for i, nm in enumerate(nama):
        print(f"{nm:>24} | {wj[i]:>8.2f} | {wk[i]:>11.2f}")
    print("-" * 50)
    print(f"{'akurasi CV':>24} | {aj:>8.4f} | {ak:>11.4f}")
    # model tunggal terbaik per sumbu, dihitung - bukan diketik tangan
    bj = max((P.argmax(1) + 1 == jen).mean() for P in J)
    bk = max((P.argmax(1) + 1 == y).mean() for P in K)
    print(f"\nmicro F1 CV gabungan          : {(aj+ak)/2:.4f}")
    print(f"terbaik tunggal per sumbu     : jenis {bj:.4f} | kerusakan {bk:.4f}")
    np.savez(O / "bobot.npz", wj=wj, wk=wk, nama=np.array(nama))


if __name__ == "__main__":
    main()
