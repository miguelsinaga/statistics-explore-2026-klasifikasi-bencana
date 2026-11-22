# Klasifikasi Jenis Bencana & Tingkat Kerusakan — Statistics Explore 2026

🥉 **Juara 3, Big Data Competition — Statistics Explore 2026**

Dari sebuah foto, model menebak **jenis bencana** (banjir / gempa bumi / kebakaran) dan
**tingkat kerusakannya** (ringan / sedang / berat).

| | Skor |
|---|---|
| **Leaderboard (Micro F1)** | **0,844** |
| Validasi silang `_jenis` | 0,9942 |
| Validasi silang `_kerusakan` | 0,8223 |
| Validasi silang Micro F1 | 0,9082 |

Laporan lengkap beserta output setiap langkah ada di
[`notebooks/model_final.ipynb`](notebooks/model_final.ipynb).

---

## Tugas dan metrik

- **Data latih:** 17.482 gambar dalam folder `TRAIN/<JENIS>/<KERUSAKAN>/`, 9 kelas yang
  cukup seimbang (rasio terbesar/terkecil 1,96×).
- **Data uji:** 450 gambar. Setiap gambar dinilai di dua baris (`<id>_jenis` dan
  `<id>_kerusakan`), jadi totalnya 900 baris.
- **Metrik:** Micro F1 atas 900 baris itu. Untuk label tunggal, Micro F1 sama dengan
  akurasi, sehingga skornya adalah rata-rata akurasi kedua sumbu.

## Tiga keputusan yang paling menentukan

Tidak ada satu pun yang soal memperbesar model.

1. **Validasi dikelompokkan per kejadian.** Dataset ini berisi banyak potongan frame video
   dan crop dari kejadian yang sama. Dengan fold per near-duplicate (pHash), skor validasi
   menggelembung **0,097**. Tetangga terdekat tanpa pelatihan apa pun hampir menyamai model
   terlatih. Klaster kejadian dibentuk dari kemiripan fitur DINOv3 (cosine ≥ 0,92) yang
   digabung dengan grup pHash, lalu dipakai sebagai grup `StratifiedGroupKFold`.
2. **Penomoran kelas mengikuti urutan folder.** Panitia menomori `_kerusakan` menurut urutan
   alfabetis folder (BERAT=1, RINGAN=2, SEDANG=3), bukan menurut tingkat keparahan. Koreksi ini
   saja bernilai **+0,167** di leaderboard. Pelatihan memakai urutan semantik, dan konversi
   dilakukan sekali saat menulis submission.
3. **Satu model khusus untuk tiap jenis bencana.** Setelah `_jenis` mencapai 0,994, 91,5% sisa
   kesalahan adalah tertukarnya dua tingkat yang bersebelahan, dan polanya berbeda per bencana
   (ketinggian air, keruntuhan struktur, luas area terbakar). ConvNeXt-Tiny khusus per jenis
   menaikkan akurasi kerusakan OOF dari 0,7814 ke 0,8169, dan leaderboard dari **0,807 ke 0,844**.

## Arsitektur

```text
Data latih resmi (17.482 gambar)
   │  cache 640 px · grup pHash + md5 · klaster kejadian (DINOv3, cosine ≥ 0,92)
   │  StratifiedGroupKFold 5 fold, seed 42
   ▼
┌─────────────────────── 6 cabang model ────────────────────────┐
│ EfficientNet-B0 @224   ┐                                      │
│ EfficientNet-B0 @320   ├ fine-tune penuh, kepala hierarchical │
│ ConvNeXt-Tiny  @224    ┘                                      │
│ DINOv3 ViT-L beku → logistic 3-arah                           │
│ DINOv3 ViT-L beku → logistic ordinal (ringan < sedang < berat)│
│ ConvNeXt-Tiny khusus BANJIR / GEMPA / KEBAKARAN               │
└───────────────────────────────────────────────────────────────┘
   ▼
Bobot gabungan per sumbu (Dirichlet + perbaikan lokal, di OOF)
   ▼
TTA (flip × 2 skala × 5 fold) → submission
```

**Kepala hierarchical.** Tahap 1 memprediksi jenis. Tahap 2 memprediksi kerusakan dari fitur
backbone yang disambung dengan softmax jenis.

**Pelatihan.** RandomResizedCrop, flip horizontal, rotasi ±15°, dan perubahan
brightness/contrast/saturation. Mixup dan CutMix bergantian, label smoothing 0,1, dropout 0,3,
AdamW (weight decay 0,05) dengan OneCycleLR, mixed precision.

**Spesialis per jenis.** Dilatih hanya pada gambar jenis tersebut, dengan fold yang sama persis.
Pada data uji, gambar diarahkan ke spesialis sesuai jenis *prediksi*. Pada OOF, baris yang
jenisnya salah ditebak dinetralkan agar skor validasi tidak lebih optimistis daripada
kondisi data uji.

### Kontribusi tiap cabang (validasi silang tingkat kejadian)

| Cabang | `_jenis` | `_kerusakan` | Bobot `_jenis` | Bobot `_kerusakan` |
|---|---|---|---|---|
| EfficientNet-B0 @224 | 0,9854 | 0,7684 | 0,03 | 0,13 |
| EfficientNet-B0 @320 | 0,9881 | 0,7588 | 0,00 | 0,12 |
| ConvNeXt-Tiny @224 | 0,9847 | 0,7814 | 0,49 | 0,12 |
| DINOv3 beku, 3-arah | 0,9928 | 0,6743 | 0,40 | 0,00 |
| DINOv3 beku, ordinal | 0,9928 | 0,6757 | 0,07 | 0,00 |
| ConvNeXt-Tiny per jenis | — | 0,8169 | 0,00 | 0,64 |
| **Gabungan** | **0,9942** | **0,8223** | | |

Fitur beku unggul di `_jenis`, sedangkan model yang di-fine-tune unggul jauh di `_kerusakan`.
Karena itu bobot dicari terpisah untuk tiap sumbu.

## Perjalanan skor

| # | Submission | Leaderboard |
|---|---|---|
| 1 | ConvNeXt-Tiny fine-tune, fold per near-duplicate | 0,592 |
| 2 | DINOv3 beku, satu kepala | 0,614 |
| 3 | Gabungan 2 model + **koreksi penomoran kelas** | 0,781 |
| 4 | Ensemble 5 model + kepala ordinal + TTA multi-skala | 0,807 |
| 5 | **+ ConvNeXt-Tiny khusus per jenis bencana** | **0,844** |

### Yang dicoba lalu ditolak

| Gagasan | Hasil |
|---|---|
| Membuang label yang diragukan model | +0,030 di CV, **−0,027** di kenyataan. Melingkar: label "meragukan" ditentukan oleh model yang kemudian menilai dirinya sendiri. |
| Membuang sampel berdasarkan konsensus antar arsitektur | Naik di probe, tidak menambah apa pun di atas spesialis |
| Boundary expert (biner untuk batas tingkat bersebelahan) | Hanya 1 dari 5 yang lolos guard, tidak berpengaruh pada model final |
| ConvNeXt-Small | 0,8760 < ConvNeXt-Tiny 0,8841, karena batch kecil di GPU 4 GB |
| Perataan prediksi antar gambar uji yang mirip | −0,03 s/d −0,13 |
| Pseudo-labeling, DANN, CORAL, kalibrasi prior, resolusi 448 px | Tanpa efek atau negatif |

Pelajaran yang berulang: kenaikan di validasi silang saja tidak cukup. Sebuah gagasan hanya
diterima kalau validasi dan bukti independen bergerak ke arah yang sama.

---

## Struktur repo

```text
├── notebooks/
│   └── model_final.ipynb      laporan model final + output, membuktikan submission identik
├── scripts/
│   ├── pipeline.py            menjalankan semua tahap secara berurutan (bisa dilanjutkan)
│   ├── prepare_data.py        indeks data, cache 640 px, grup near-duplicate
│   ├── dinov3_features.py     fitur DINOv3 ViT-L beku
│   ├── build_events.py        klaster kejadian → fold yang jujur
│   ├── train_effnet.py        fine-tune CNN hierarchical (global & per jenis)
│   ├── rangkum_oof.py         OOF keenam cabang
│   ├── cari_bobot.py          bobot gabungan per sumbu
│   ├── predict_all.py         inferensi data uji + TTA → submission
│   └── se_lib.py              utilitas: indeks, dedup, format submission
├── output_final/
│   └── final_perjenis.csv     submission yang dikumpulkan (LB 0,844)
└── requirements.txt
```

## Menjalankan ulang

**1. Lingkungan** (Python 3.9, GPU CUDA disarankan)

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**2. Data.** Unduh dataset dari halaman kompetisi, lalu letakkan di root repo dengan susunan
berikut:

```text
big-data-competition-statistics-explore-2026/SE/
├── TRAIN/
│   ├── BANJIR/KERUSAKAN {RINGAN,SEDANG,BERAT}/...
│   ├── GEMPA BUMI/...
│   ├── KEBAKARAN/...
│   └── Solution.csv
└── TEST/1.jpg … 450.jpg
```

**3. Pipeline**

```bash
python scripts/pipeline.py --dry-run    # lihat status tiap tahap
python scripts/pipeline.py              # jalankan; tahap yang sudah selesai dilewati
```

| Tahap | Keluaran |
|---|---|
| `data` | `output_local/train_folds.csv`, `cache_640/` |
| `dino` | `output_dino/feat_{train,test}.npy` |
| `kejadian` | `output_local/train_events.csv` |
| `anchor` | 3 CNN global × 5 fold → `output_effnet/` |
| `spesialis` | 3 ConvNeXt-T per jenis × 5 fold |
| `oof` | `output_effnet/oof_semua.npz` |
| `bobot` | `output_effnet/bobot.npz` |
| `uji` | `output_final/submission.csv` |

Semua tahap dikerjakan di laptop dengan GPU **RTX 3050 4 GB**, dan karena itu ukuran model
dibatasi ke kelas B0/Tiny pada 224–320 px. Tahap pelatihan memakan waktu beberapa jam dan bisa
dilanjutkan bila terputus: fold yang checkpoint-nya sudah ada otomatis dilewati.

Setelah pipeline selesai, `notebooks/model_final.ipynb` bisa dijalankan ulang. Sel terakhirnya
membangun ulang submission dari probabilitas data uji dan memeriksa bahwa hasilnya identik
dengan `output_final/final_perjenis.csv`.

**Reproduksibilitas.** Seed 42 untuk pembagian fold dan seluruh proses acak. Hasil yang identik
bit-per-bit lintas versi driver CUDA tidak dijamin, tetapi pembagian fold, OOF, bobot, dan
submission dapat dibangun ulang secara deterministik dari checkpoint yang sama.

## Keterbatasan

- **Label kerusakan tidak selalu konsisten.** Dari 6.682 pasangan gambar identik, 725 (10,9%)
  diberi label kerusakan berbeda.
- **Batas antar tingkat memang kabur.** Beda antara "sedang" dan "berat" sering menjadi soal
  penilaian, bahkan bagi manusia.
- **`_kerusakan` (0,82) masih jauh di bawah `_jenis` (0,99).** Jenis bencana cukup dikenali dari
  pemandangan, sedangkan tingkat keparahan menuntut model membaca objek di dalamnya.

## Kepatuhan

- Hanya dataset resmi panitia. Tidak ada data eksternal, tidak ada API klasifikasi eksternal,
  dan seluruh komputasi dilakukan secara lokal.
- Bobot pretrained (diizinkan oleh ketentuan lomba): `efficientnet_b0` dan `convnext_tiny`
  (timm, Apache-2.0), serta `vit_large_patch16_dinov3.lvd1689m` (DINOv3, Meta).
