"""Pipeline lengkap: dari data resmi panitia sampai berkas submission (LB 0.844).

Setiap tahap punya keluaran sendiri, sehingga bisa diaudit, dilewati kalau sudah
selesai, dan dilanjutkan setelah terputus.

    python scripts/pipeline.py --dry-run      lihat status tiap tahap
    python scripts/pipeline.py                jalankan, lewati tahap yang sudah selesai
    python scripts/pipeline.py --only bobot   jalankan satu tahap saja
    python scripts/pipeline.py --force        ulangi walau keluarannya sudah ada

Pipeline tidak pernah membaca label data uji. Seluruh pemilihan bobot dilakukan
pada validasi silang data latih.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "scripts"
O = ROOT / "output_effnet"
PY = sys.executable


def py(script, *args):
    return [PY, "-u", str(S / script), *args]


def latih(tag, *extra):
    return py("train_effnet.py", "--folds", "0,1,2,3,4", *extra, "--tag", tag)


CNX = ("--epochs", "12", "--arch", "convnext_tiny", "--bs", "24", "--lr", "2e-4")

TAHAP = [
    ("data", "Indeks data, cache 640 px, grup near-duplicate",
     [ROOT / "output_local" / "train_folds.csv"],
     [py("prepare_data.py")]),

    ("dino", "Fitur DINOv3 ViT-L beku untuk data latih dan uji",
     [ROOT / "output_dino" / "feat_train.npy", ROOT / "output_dino" / "feat_test.npy"],
     [py("dinov3_features.py")]),

    ("kejadian", "Klaster kejadian (cosine >= 0.92) untuk fold yang jujur",
     [ROOT / "output_local" / "train_events.csv"],
     [py("build_events.py")]),

    ("anchor", "3 CNN global di-fine-tune: EffNet-B0 @224/@320, ConvNeXt-T @224",
     [O / f"effnet{t}_fold4.pth" for t in ("", "_r320", "_cnx")],
     [latih("", "--epochs", "8"),
      latih("_r320", "--epochs", "12", "--size", "320", "--bs", "16"),
      latih("_cnx", *CNX)]),

    ("spesialis", "ConvNeXt-T khusus tiap jenis bencana (BANJIR, GEMPA, KEBAKARAN)",
     [O / f"effnet_cnj{j}_fold4.pth" for j in (1, 2, 3)],
     [latih(f"_cnj{j}", *CNX, "--jenis", str(j)) for j in (2, 1, 3)]),

    ("oof", "Kumpulkan OOF keenam cabang (termasuk 2 kepala DINOv3)",
     [O / "oof_semua.npz"],
     [py("rangkum_oof.py")]),

    ("bobot", "Bobot penggabungan per sumbu, dicari di validasi silang",
     [O / "bobot.npz"],
     [py("cari_bobot.py")]),

    ("uji", "Inferensi data uji + TTA, tulis submission",
     [ROOT / "output_final" / "submission.csv"],
     [py("predict_all.py", "--out", "submission.csv")]),
]

# Kalau masukan sebuah tahap LEBIH BARU dari keluarannya, tahap itu dianggap basi.
DEP = {
    "kejadian": [ROOT / "output_dino" / "feat_train.npy"],
    "oof": [O / f"effnet{t}_fold4.pth" for t in ("", "_r320", "_cnx")] +
           [O / f"effnet_cnj{j}_fold4.pth" for j in (1, 2, 3)],
    "bobot": [O / "oof_semua.npz"],
    "uji": [O / "bobot.npz"],
}


def selesai(nama, out):
    if not all(p.exists() for p in out):
        return False
    dep = [p for p in DEP.get(nama, []) if p.exists()]
    return not dep or max(p.stat().st_mtime for p in dep) <= min(p.stat().st_mtime for p in out)


def jalankan(cmd, nama):
    """Tampilkan ke layar sekaligus simpan ke output_effnet/log_<tag>.txt."""
    tag = cmd[cmd.index("--tag") + 1] if "--tag" in cmd else nama
    log = O / f"log_{str(tag).lstrip('_') or 'b0'}.txt"
    O.mkdir(exist_ok=True)
    print(f"  (log: {log.name})", flush=True)
    with log.open("a", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        for ln in p.stdout:
            print(ln, end="", flush=True)
            fh.write(ln)
        p.wait()
    return p.returncode


def main():
    dry = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None

    print(f"{'tahap':>10} | {'status':>8} | keterangan")
    print("-" * 78)
    for nama, ket, out, _ in TAHAP:
        print(f"{nama:>10} | {'SELESAI' if selesai(nama, out) else 'belum':>8} | {ket}")
    print("-" * 78)
    if dry:
        return

    for nama, ket, out, cmds in TAHAP:
        if only and nama != only:
            continue
        if selesai(nama, out) and not force:
            print(f"\n[{nama}] sudah selesai, dilewati")
            continue
        print(f"\n[{nama}] {ket}", flush=True)
        for c in cmds:
            print("  $ " + " ".join(Path(x).name if str(x).endswith(".py") else str(x)
                                    for x in c[2:]), flush=True)
            if jalankan(c, nama):
                raise SystemExit(f"tahap '{nama}' gagal")
        if not selesai(nama, out):
            raise SystemExit(f"tahap '{nama}' selesai tapi keluarannya tidak ada")
    print("\nSEMUA TAHAP SELESAI")


if __name__ == "__main__":
    main()
