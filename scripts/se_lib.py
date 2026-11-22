"""Library inti pipeline Statistics Explore 2026 - klasifikasi citra bencana.

Indeks dataset, deduplikasi (md5 + perceptual hash), dan penulisan submission.
Bebas dependensi GPU (numpy / pandas / PIL / scipy / sklearn).

Kontrak label (dari QnA panitia):
    _jenis     : BANJIR=1, GEMPA BUMI=2, KEBAKARAN=3
    _kerusakan : RINGAN=1, SEDANG=2, BERAT=3
    label9     : (jenis-1) * 3 + (kerusakan-1)   -> 0..8
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct

Image.MAX_IMAGE_PIXELS = None

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}

JENIS_MAP = {"banjir": 1, "gempa bumi": 2, "kebakaran": 3}
KERUSAKAN_MAP = {"ringan": 1, "sedang": 2, "berat": 3}
JENIS_NAMES = {1: "BANJIR", 2: "GEMPA BUMI", 3: "KEBAKARAN"}
KERUSAKAN_NAMES = {1: "RINGAN", 2: "SEDANG", 3: "BERAT"}


# --------------------------------------------------------------------------- #
# 1. Index dataset                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def to_label9(jenis: int, kerusakan: int) -> int:
    return (jenis - 1) * 3 + (kerusakan - 1)


def build_index(train_dir: str | Path) -> pd.DataFrame:
    """Ubah struktur folder TRAIN/<JENIS>/<KERUSAKAN X>/*.jpg jadi tabel datar."""
    train_dir = Path(train_dir)
    rows = []
    for f in sorted(train_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMG_EXT:
            continue
        rel = f.relative_to(train_dir).parts
        if len(rel) != 3:
            continue
        jenis = JENIS_MAP[_norm(rel[0])]
        kname = _norm(rel[1])
        kerusakan = next((v for k, v in KERUSAKAN_MAP.items() if k in kname), None)
        if kerusakan is None:
            raise ValueError(f"folder kerusakan tak dikenali: {rel[1]}")
        rows.append(
            {
                "path": str(f),
                "fname": f.name,
                "jenis": jenis,
                "kerusakan": kerusakan,
                "label": to_label9(jenis, kerusakan),
            }
        )
    return pd.DataFrame(rows)


def list_test_files(test_dir: str | Path) -> list[Path]:
    """File test diurutkan numerik sesuai ID (1.jpg, 2.jpg, ... 450.jpg)."""
    test_dir = Path(test_dir)
    files = [f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]
    return sorted(files, key=lambda f: (not f.stem.isdigit(), int(f.stem) if f.stem.isdigit() else 0, f.stem))


# --------------------------------------------------------------------------- #
# 2. Deduplikasi: md5 (persis sama) + perceptual hash (mirip)                  #
# --------------------------------------------------------------------------- #
def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def phash_u64(im: Image.Image, hash_size: int = 8, highfreq: int = 4) -> np.uint64:
    """Perceptual hash 64-bit, algoritma sama dengan library `imagehash`."""
    size = hash_size * highfreq
    g = im.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    out = np.uint64(0)
    for b in bits:
        out = np.uint64(out << np.uint64(1)) | np.uint64(bool(b))
    return out


_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming(a: np.uint64, b: np.uint64) -> int:
    x = int(a) ^ int(b)
    return int(_POPCNT8[np.frombuffer(x.to_bytes(8, "little"), dtype=np.uint8)].sum())


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def group_near_duplicates(
    hashes: np.ndarray, md5s: list[str] | None = None, thresh: int = 5
) -> np.ndarray:
    """Kelompokkan gambar duplikat / near-duplicate menjadi group id.

    Pakai banded LSH: hash 64-bit dipecah jadi 8 band x 8 bit. Dua gambar dengan
    jarak hamming <= 7 pasti punya minimal satu band identik (pigeonhole), jadi
    untuk thresh<=7 pencarian ini lengkap tanpa perlu bandingkan semua pasangan.
    """
    n = len(hashes)
    dsu = _DSU(n)

    if md5s is not None:
        seen: dict[str, int] = {}
        for i, m in enumerate(md5s):
            if m in seen:
                dsu.union(seen[m], i)
            else:
                seen[m] = i

    ints = [int(h) for h in hashes]
    for band in range(8):
        shift = band * 8
        buckets: dict[int, list[int]] = {}
        for i, v in enumerate(ints):
            buckets.setdefault((v >> shift) & 0xFF, []).append(i)
        for idxs in buckets.values():
            if len(idxs) < 2 or len(idxs) > 4000:  # bucket raksasa = hash datar, lewati
                continue
            for a in range(len(idxs)):
                ia = idxs[a]
                for b in range(a + 1, len(idxs)):
                    ib = idxs[b]
                    if dsu.find(ia) == dsu.find(ib):
                        continue
                    if _hamming(hashes[ia], hashes[ib]) <= thresh:
                        dsu.union(ia, ib)

    roots = {}
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = dsu.find(i)
        if r not in roots:
            roots[r] = len(roots)
        out[i] = roots[r]
    return out


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """StratifiedGroupKFold: stratifikasi 9 kelas, group = klaster duplikat."""
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, vi) in enumerate(sgkf.split(df, df.label, groups=df.group)):
        df.iloc[vi, df.columns.get_loc("fold")] = k
    return df


# --------------------------------------------------------------------------- #
# 3. Penulisan submission                                                     #
# --------------------------------------------------------------------------- #
def read_solution(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Baca Solution.csv dan deteksi separatornya (file resmi pakai ';')."""
    path = Path(path)
    head = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    sep = ";" if head.count(";") >= head.count(",") else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False), sep


def build_submission(
    sol: pd.DataFrame, ids: list[str] | np.ndarray, jenis: np.ndarray, kerusakan: np.ndarray
) -> pd.DataFrame:
    """Isi kolom Target mengikuti URUTAN BARIS Solution.csv persis (wajib lomba).

    `ids` adalah stem file test ('1','2',...), disandingkan dengan baris
    '<id>_jenis' dan '<id>_kerusakan' di template.
    """
    id_col, tgt_col = sol.columns[0], sol.columns[1]
    lookup: dict[str, str] = {}
    for i, j, k in zip(ids, jenis, kerusakan):
        lookup[f"{i}_jenis"] = str(int(j))
        lookup[f"{i}_kerusakan"] = str(int(k))

    missing = [x for x in sol[id_col] if x not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} ID di Solution.csv tidak punya prediksi, contoh: {missing[:5]}")

    out = sol.copy()
    out[tgt_col] = [lookup[x] for x in sol[id_col]]  # urutan baris template dipertahankan
    return out


def verify_submission(sub: pd.DataFrame, sol: pd.DataFrame) -> None:
    id_col, tgt_col = sol.columns[0], sol.columns[1]
    assert list(sub.columns) == list(sol.columns), f"kolom beda: {list(sub.columns)}"
    assert len(sub) == len(sol), f"jumlah baris beda: {len(sub)} vs {len(sol)}"
    assert list(sub[id_col]) == list(sol[id_col]), "URUTAN BARIS berubah - dilarang panitia"
    assert sub[tgt_col].isna().sum() == 0 and (sub[tgt_col].astype(str).str.len() > 0).all(), "Target kosong"
    assert set(sub[tgt_col].unique()).issubset({"1", "2", "3"}), f"nilai aneh: {sorted(set(sub[tgt_col]))}"
    n_j = sub[id_col].str.endswith("_jenis").sum()
    n_k = sub[id_col].str.endswith("_kerusakan").sum()
    assert n_j == n_k, f"jumlah baris _jenis ({n_j}) != _kerusakan ({n_k})"
    print(f"OK  {len(sub)} baris ({n_j} _jenis + {n_k} _kerusakan), urutan & nilai valid.")


def write_submission(sub: pd.DataFrame, path: str | Path, sep: str = ";") -> Path:
    path = Path(path)
    sub.to_csv(path, index=False, sep=sep, lineterminator="\r\n")
    return path
