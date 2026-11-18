# Klasifikasi Motif Batik: GLCM, LBP, dan Fusinya dengan SVM

Tugas EAS mata kuliah **Pengolahan Citra Digital** — S1 Informatika, UPN "Veteran" Jawa Timur.

*Analisis Komparatif Ekstraksi Fitur GLCM, LBP, dan Kombinasinya untuk Klasifikasi Motif Batik Menggunakan Support Vector Machine.*

Anggota kelompok: _(belum diisi)_

## Ringkasan

Empat skenario fitur (GLCM 0°, GLCM 4 arah, uniform LBP, dan fusi GLCM+LBP) × tiga kernel SVM (linear, RBF, polynomial) diuji pada 3 kelas motif batik: Kawung, Megamendung, dan Parang.

| Skenario | Fitur | Dimensi |
|---|---|---|
| S1 | GLCM sudut 0° (4 properti × d = 1, 3, 5) | 12 |
| S2 | GLCM 4 arah, dirata-rata per jarak | 12 |
| S3 | Uniform LBP (P=8, R=1) | 59 |
| S4 | S2 + S3 | 71 |

## Dataset

*Batik Kawung, Megamendung, Parang, Sekarjagad, and Truntum Image Dataset* (Putra & Zamzami, Mendeley Data, 2025), CC BY 4.0 — [doi:10.17632/5hp2dg7n3m.1](https://data.mendeley.com/datasets/5hp2dg7n3m/1).

Dataset tidak disertakan di repo. Unduh ZIP dari Mendeley dan taruh di `dataset/`. Notebook 01 akan mengekstraknya ke `data/raw/` secara otomatis.

## Menjalankan

Memakai [uv](https://docs.astral.sh/uv/) dan Python 3.13.

```bash
uv sync
uv run pytest
uv run jupyter lab
```

Jalankan notebook di `notebooks/` secara berurutan.

## Struktur

```
src/project/   config, data (ekstraksi + audit), preprocess, features (GLCM, LBP)
notebooks/     01_audit_dan_split, 02_preprocessing_dan_fitur, ...
tests/         uji kebenaran fitur GLCM/LBP
figures/       gambar untuk laporan dan presentasi
results/       hasil audit, split, CV, dan evaluasi
```

## Status

- [x] Audit data dan visualisasi
- [x] Visualisasi preprocessing, LBP, dan GLCM
- [ ] Cek duplikat (dHash) dan split group-aware
- [ ] Ekstraksi fitur seluruh dataset
- [ ] Eksperimen SVM (12 model) dan evaluasi
- [ ] Pembahasan hipotesis
