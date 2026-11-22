"""Kelompokkan gambar train per KEJADIAN, bukan per near-duplicate.

Masalah: pHash hanya menangkap crop/resize yang nyaris identik. Foto berbeda dari
peristiwa yang SAMA (sudut lain, frame lain, banjir yang sama) lolos, padahal
label kerusakannya identik. Validasi jadi menilai kemampuan MENGHAFAL kejadian.

Solusi: klaster berdasarkan kemiripan fitur DINOv3 (cosine), lalu gabungkan dengan
klaster pHash lama. Satu kejadian = satu grup = tidak boleh terpisah antar fold.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
THR = 0.92           # disapu 0.86-0.96; di bawah 0.92 klaster berantai jadi gumpalan
                     # raksasa (5289 gambar), di atasnya kejadian mulai terpecah


def main():
    df = pd.read_csv(ROOT / "output_local" / "train_folds.csv")
    F = normalize(np.load(ROOT / "output_dino" / "feat_train.npy")).astype(np.float32)
    n = len(F)
    rows, cols = [], []
    for i in range(0, n, 2048):
        s = F[i:i + 2048] @ F.T
        r, c = np.where(s >= THR)
        rows.append(r + i); cols.append(c)
        print(f"  {min(i+2048,n)}/{n}", end="\r", flush=True)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    print(f"\nklaster kejadian (cosine>={THR}): {k}")

    # gabungkan dengan grup pHash lama supaya tidak ada yang longgar
    if "group" in df.columns:
        pair = pd.DataFrame({"a": df.group.values, "b": lab})
        m = pair.groupby("a").b.transform("min")
        lab = pd.factorize(pd.Series(list(zip(m, lab))))[0]
        print(f"setelah digabung grup pHash          : {len(np.unique(lab))}")

    df["event"] = lab
    sz = pd.Series(lab).value_counts()
    print(f"\nukuran klaster: median {sz.median():.0f}  terbesar {sz.max()}  "
          f"tunggal {(sz==1).sum()} ({(sz==1).sum()/len(sz)*100:.0f}%)")
    print(f"gambar yang berbagi kejadian dgn gambar lain: "
          f"{(sz[lab].values > 1).mean()*100:.1f}%  (pHash lama: 26.8%)")
    df.to_csv(ROOT / "output_local" / "train_events.csv", index=False)
    print(f"\nditulis: {ROOT/'output_local'/'train_events.csv'}")


if __name__ == "__main__":
    main()
