"""Tahap 0: indeks data latih, cache gambar 640 px, dan grup near-duplicate.

    TRAIN/<JENIS>/<KERUSAKAN X>/*.jpg  ->  output_local/train_folds.csv
                                           cache_640/{train,test}/<n>.jpg

- Cache: tiap gambar di-decode sekali dan dikecilkan ke sisi terpanjang 640 px
  (dataset memuat gambar sampai 9232x6541 px; decode tiap epoch akan jadi bottleneck).
- Grup: md5 (identik byte) + perceptual hash (hamming <= 5) dari gambar ASLI,
  digabung union-find. Kolom `group` ini lalu diperluas menjadi klaster KEJADIAN
  oleh build_events.py.

Pakai:
    python scripts/prepare_data.py [--workers 6]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import se_lib as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "big-data-competition-statistics-explore-2026" / "SE"
OUT = ROOT / "output_local"
CACHE = ROOT / "cache_640"
SIDE = 640


def _cache_one(job):
    src, dst = job
    if os.path.exists(dst):
        return 1
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > SIDE:
                s = SIDE / max(w, h)
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                               Image.Resampling.BILINEAR)
            im.save(dst, format="JPEG", quality=92)
        return 1
    except Exception as e:  # noqa: BLE001
        print("  GAGAL cache:", src, repr(e))
        return 0


def _hash_one(p):
    with Image.open(p) as im:
        return int(L.phash_u64(im)), L.file_md5(p)


def build_cache(df, test_files, workers):
    (CACHE / "train").mkdir(parents=True, exist_ok=True)
    (CACHE / "test").mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(CACHE / "train" / f"{i}.jpg")) for i, p in enumerate(df.path)]
    jobs += [(str(f), str(CACHE / "test" / f"{f.stem}.jpg")) for f in test_files]
    todo = sum(1 for _, d in jobs if not os.path.exists(d))
    if todo:
        print(f"cache resize {todo} gambar dengan {workers} proses...", flush=True)
        t0, ok = time.time(), 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(_cache_one, jobs, chunksize=64), 1):
                ok += r
                if i % 2000 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"cache: {len(jobs)} gambar sudah ada, dilewati.")
    df["cpath"] = [str(CACHE / "train" / f"{i}.jpg") for i in range(len(df))]
    missing = [d for _, d in jobs if not os.path.exists(d)]
    assert not missing, f"{len(missing)} file cache gagal dibuat, contoh: {missing[:3]}"
    return df


def build_groups(df, workers):
    print(f"hashing {len(df)} gambar latih (md5 + pHash)...", flush=True)
    t0, res = time.time(), []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_hash_one, df.path.tolist(), chunksize=64), 1):
            res.append(r)
            if i % 3000 == 0 or i == len(df):
                print(f"  {i}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)
    hashes = np.array([r[0] for r in res], dtype=np.uint64)
    md5s = [r[1] for r in res]
    df["group"] = L.group_near_duplicates(hashes, md5s, thresh=5)
    n = df.group.nunique()
    print(f"dedup: {n} grup dari {len(df)} gambar ({100*(1-n/len(df)):.1f}% duplikat)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    assert DATA.exists(), f"folder data tidak ketemu: {DATA}"

    df = L.build_index(DATA / "TRAIN")
    test_files = L.list_test_files(DATA / "TEST")
    print(f"latih {len(df)} gambar | uji {len(test_files)} gambar")

    df = build_cache(df, test_files, a.workers)
    df = build_groups(df, a.workers)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "train_folds.csv", index=False)
    print(f"ditulis: {OUT / 'train_folds.csv'}")


if __name__ == "__main__":
    main()
