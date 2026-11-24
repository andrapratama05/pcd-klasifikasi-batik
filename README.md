# Klasifikasi Motif Batik Banyumasan dengan GLCM, LBP, dan SVM

Tugas Evaluasi Akhir Semester (EAS) mata kuliah **Pengolahan Citra Digital**, Program Studi S1 Informatika, UPN "Veteran" Jawa Timur.

Desain eksperimen ditetapkan pada Evaluasi Tengah Semester (ETS) dalam bentuk proposal penelitian. Repositori ini berisi implementasi proposal tersebut untuk EAS.

**Judul kerja:** *Analisis Komparatif Ekstraksi Fitur GLCM, LBP, dan Kombinasinya untuk Klasifikasi Motif Batik Banyumasan Menggunakan Support Vector Machine.*

**Anggota kelompok:**

| Nama | NPM |
|---|---|
| Theresia Akwila Anastasya S. | 23081010010 |
| Andra Pratama | 23081010112 |
| Hardy Gustino | 23081010160 |
| Mohammad Sahrul Faza | 23081010242 |

## Ringkasan riset

Penelitian ini membandingkan fitur tekstur GLCM, LBP, dan gabungan keduanya untuk mengklasifikasikan 11 motif Batik Banyumasan dengan SVM. Ada empat skenario fitur, masing-masing diuji dengan tiga kernel SVM (linear, RBF, polynomial), sehingga terdapat 12 model.

| Skenario | Fitur | Dimensi |
|---|---|---|
| S1 | GLCM sudut 0° (kontras, korelasi, energi/ASM, homogenitas × d = 1, 3, 5) | 12 |
| S2 | GLCM 4 arah (0°, 45°, 90°, 135°), dirata-rata per jarak | 12 |
| S3 | Uniform LBP (P = 8, R = 1) | 59 |
| S4 | Gabungan S2 + S3 | 71 |

Hipotesis:

- **H1**: gabungan GLCM+LBP (S4) lebih baik daripada fitur tunggal (S2 dan S3).
- **H2**: kernel non-linear (RBF/polynomial) lebih baik daripada kernel linear pada fitur yang sama.
- **H3**: GLCM 4 arah (S2) lebih baik daripada GLCM 0° saja (S1), dilihat dari rata-rata dan simpangan baku antar-fold, serta dari recall kelas yang polanya paling berarah.

**Mengapa split per foto asli?** Pada dataset ini, 8 dari setiap 9 citra adalah hasil augmentasi dari satu foto asli. Jika data dibagi acak per file, salinan dari foto yang sama dapat masuk ke data latih dan data uji sekaligus, sehingga model cukup "mengenali kain yang sama". Karena itu data dibagi per foto asli (`origin_id`), dan data uji hanya berisi foto asli. Akurasi yang dihasilkan mengukur kemampuan mengenali motif.

## Dataset

Syafi'i. (2025). *Batik Banyumasan* (Versi 2) [Dataset]. Mendeley Data. https://doi.org/10.17632/vsjrkhs9km.2

- Lisensi: CC BY 4.0
- 11 kelas: Angguran, Ayam Puger, Jahe Lumbon, Jahe Puger, Jahe Srimpang, Lumbon, Madu Bronto, Pring Sedapur, Puger Galar, Puger Telu Bal, Wit Lumbon
- 572 foto asli (52 per kelas), masing-masing dengan 8 salinan augmentasi, total 5.148 citra (468 per kelas). Jumlah ini diverifikasi ulang di notebook 01.

Data tidak disimpan di repositori. Gunakan skrip pengunduh di bawah.

## Struktur repo

```
data/          dataset (tidak di-commit): data/zip/ untuk ZIP, data/raw/ untuk hasil ekstraksi, data/demo/ untuk foto baru
src/           kode yang dipakai ulang: konfigurasi, pengunduh data, split, preprocessing, fitur, evaluasi, prediksi
notebooks/     alur eksperimen 01–05 dan demo klasifikasi 06, dijalankan berurutan
tests/         uji kebenaran (pytest) untuk parsing data, split, fitur, dan prediksi
features/      cache fitur hasil ekstraksi (tidak di-commit)
models/        model terbaik (best_model.joblib) dan metadatanya (best_model.json), dibuat notebook 04
results/       tabel hasil (split, cross-validation, data uji, ablasi)
figures/       gambar untuk laporan dan slide, per notebook
```

## Cara menjalankan

### 1. Prasyarat

- Python ≥ 3.10 (proyek ini diuji dengan 3.12)
- [uv](https://docs.astral.sh/uv/)
- Git

### 2. Clone dan pasang dependensi

```bash
git clone <url-repo>
cd <nama-folder-repo>
uv sync
```

### 3. Unduh dataset

```bash
uv run python -m src.download_data
```

Skrip ini mengunduh ZIP (sekitar 1,1 GB) ke `data/zip/batik_banyumasan_v2.zip`, mengekstraknya ke `data/raw/batik_banyumasan/`, lalu memverifikasi jumlah file. Jika dijalankan lagi dan data sudah lengkap, skrip tidak mengunduh ulang. Untuk mengulang ekstraksi dari awal:

```bash
uv run python -m src.download_data --force
```

**Menjeda unduhan**: tekan Ctrl+C (atau koneksi terputus). Jalankan lagi perintah yang sama untuk melanjutkan dari posisi terakhir; unduhan tidak diulang dari awal.

**Jika unduhan otomatis gagal**: buka https://data.mendeley.com/datasets/vsjrkhs9km/2, klik **Download All**, letakkan file ZIP di `data/zip/`, lalu jalankan perintah yang sama.

### 4. Jalankan uji

```bash
uv run pytest
```

### 5. Jalankan notebook

Buka dengan Jupyter Lab:

```bash
uv run jupyter lab
```

atau buka folder ini di VS Code dan pilih kernel dari `.venv`. Jalankan notebook berurutan:

1. `01_dataset_dan_eda.ipynb`
2. `02_preprocessing_dan_split.ipynb`
3. `03_skenario_dan_training.ipynb`
4. `04_evaluasi_dan_hipotesis.ipynb`
5. `05_ablasi_augmentasi.ipynb` (tambahan)
6. `06_demo_klasifikasi.ipynb` (uji coba model terbaik pada citra satuan)

Untuk menjalankan ulang semua notebook sekaligus dari kernel bersih:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

### Perkiraan waktu jalan

| Notebook | Waktu |
|---|---|
| 01 Dataset dan EDA | ± 30 detik |
| 02 Preprocessing dan split | ± 10 detik |
| 03 Skenario dan training | ± 40 detik (termasuk ekstraksi fitur 4.113 citra; lebih cepat bila cache `features/` sudah ada) |
| 04 Evaluasi dan hipotesis | ± 10 detik |
| 05 Ablasi augmentasi | ± 50 detik |
| 06 Demo klasifikasi | ± 10 detik |

Diukur pada 24/25-11-2026 di laptop tanpa GPU (eksekusi ulang semua notebook dari kernel bersih).

### 6. Uji coba klasifikasi (model terbaik)

Notebook 04 menyimpan model terbaik menurut validasi silang (saat ini S2 GLCM 4 arah + SVM linear) ke `models/best_model.joblib`, beserta metadata di `models/best_model.json` (skenario, kernel, parameter, urutan fitur, `IMG_SIZE`, kelas, versi scikit-learn). Model ini adalah model yang sama dengan yang dievaluasi di notebook 04, tanpa training ulang.

Klasifikasikan satu citra atau seluruh isi folder dari terminal:

```bash
uv run python -m src.predict "data/raw/batik_banyumasan/Puger Galar/Puger Galar (20).jpg"
uv run python -m src.predict data/demo/
```

Keluarannya berupa label dan 3 kandidat teratas beserta skor `decision_function` SVM. Skor ini untuk mengurutkan kandidat, bukan probabilitas.

Untuk mencoba foto sendiri, letakkan fotonya di `data/demo/` (tidak harus persegi), lalu jalankan perintah di atas atau notebook `06_demo_klasifikasi.ipynb`. Preprocessing dan ekstraksi fitur yang dipakai sama persis dengan saat training. Catatan: model dilatih dari kain-kain di satu rumah batik, jadi foto dari kain atau kondisi pemotretan lain bisa diklasifikasikan lebih buruk daripada angka di data uji.

Dari Python:

```python
from src.predict import predict_image
predict_image("data/demo/foto_saya.jpg")   # {'label': ..., 'top': [(label, skor), ...]}
```

Kalau versi scikit-learn berbeda dengan yang tercatat di metadata, akan muncul peringatan; jalankan ulang notebook 04 untuk membuat ulang model.

## Hasil

Eksperimen utama (Opsi A): SVM dilatih dengan 457 foto asli latih, hyperparameter dipilih lewat validasi silang 5-fold (`f1_macro`), lalu diuji sekali pada 115 foto asli uji. Sumber: `results/cv_results.csv` dan `results/test_results.csv`. ★ = kernel terbaik per skenario menurut validasi silang.

| Skenario | Kernel | F1 makro CV (mean ± std) | Akurasi CV (mean ± std) | F1 makro uji | Akurasi uji |
|---|---|---|---|---|---|
| S1 | linear | 0.927 ± 0.016 | 0.928 ± 0.015 | 0.923 | 0.922 |
| S1 | rbf ★ | 0.954 ± 0.007 | 0.954 ± 0.008 | 0.965 | 0.965 |
| S1 | poly | 0.930 ± 0.034 | 0.932 ± 0.032 | 0.983 | 0.983 |
| S2 | linear ★ | 0.980 ± 0.011 | 0.980 ± 0.011 | 1.000 | 1.000 |
| S2 | rbf | 0.973 ± 0.027 | 0.974 ± 0.026 | 1.000 | 1.000 |
| S2 | poly | 0.974 ± 0.018 | 0.974 ± 0.018 | 1.000 | 1.000 |
| S3 | linear | 0.916 ± 0.004 | 0.915 ± 0.004 | 0.916 | 0.922 |
| S3 | rbf ★ | 0.938 ± 0.033 | 0.939 ± 0.031 | 0.953 | 0.957 |
| S3 | poly | 0.892 ± 0.035 | 0.893 ± 0.035 | 0.896 | 0.896 |
| S4 | linear ★ | 0.950 ± 0.013 | 0.950 ± 0.013 | 0.961 | 0.965 |
| S4 | rbf | 0.949 ± 0.024 | 0.950 ± 0.022 | 0.982 | 0.983 |
| S4 | poly | 0.936 ± 0.036 | 0.937 ± 0.034 | 0.911 | 0.913 |

Ringkasan hipotesis (status dari validasi silang; selisih < std ditulis *tidak meyakinkan*; rincian di notebook 04):

| Hipotesis | Status |
|---|---|
| H1: S4 > S2 dan S3 | **Tidak didukung**: S4 lebih rendah daripada S2, dan setara dengan S3 |
| H2: non-linear > linear | **Sebagian**: didukung hanya pada S1, tidak meyakinkan pada S2–S4 |
| H3: S2 > S1 | **Didukung untuk rata-rata F1**; kestabilan (std) tidak konsisten; recall Puger Galar S2 ≥ S1 |

**Batasan penting.** Foto uji hampir selalu punya foto latih dari kain fisik yang sama (difoto berurutan; lihat `figures/04_evaluasi/foto_uji_vs_latih_terdekat.png`). Split per foto asli mencegah kebocoran augmentasi, tetapi tidak mencegah kebocoran kain. Akibatnya, angka absolut di atas mengukur kemampuan mengenali kain-kain di dataset ini, bukan generalisasi ke kain baru. Perbandingan antar-skenario tetap adil karena semua skenario terkena batasan yang sama.

Ablasi augmentasi (Opsi B, notebook 05, `results/ablation_aug.csv`): menambahkan 8 augmentasi bawaan ke data latih menaikkan F1 CV secara meyakinkan pada 9 dari 12 model, terutama LBP (S3, +0,049 sampai +0,060). GLCM 4 arah (S2) hampir tidak diuntungkan karena rotasi dan pencerminan tidak mengubah fiturnya.

Rangkuman lengkap hasil dan kesimpulan: [`results/ringkasan_hasil.md`](results/ringkasan_hasil.md).

Gambar utama:
- `figures/03_training/cv_per_skenario_kernel.png`: F1 dan akurasi validasi silang per skenario × kernel
- `figures/04_evaluasi/confusion_terbaik_per_skenario.png`: confusion matrix model terbaik per skenario
- `figures/04_evaluasi/f1_uji_vs_cv.png`: data uji vs validasi silang

Angka di sini tidak dapat dibandingkan langsung dengan paper rujukan karena dataset, jumlah kelas, ukuran citra, atau skema split berbeda.

## Referensi

1. Andono, P. N., & Rachmawanto, E. H. (2021). Evaluasi Ekstraksi Fitur GLCM dan LBP Menggunakan Multikernel SVM untuk Klasifikasi Batik. *Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)*, 5(1), 1–9.
2. Fatimah, & Agustin. (2025). Klasifikasi Citra Batik Menggunakan LBP dan SVM. *Jurnal Algoritma*, 22(1), 185–196.
3. Adeva, Gaisani, & Pratama. (2025). Klasifikasi Citra Batik Banyumasan Menggunakan Ekstraksi Fitur HOG, Reduksi Dimensi PCA, dan Klasifikasi SVM. *Jurnal Ilmiah Teknologi Informasi dan Robotika (JIFTI)*, 7(2). https://doi.org/10.33005/jifti.v7i2.193
4. Syafi'i. (2024). *Klasifikasi Citra Motif Batik Banyumasan menggunakan Algoritma CNN* [Skripsi]. Institut Teknologi Telkom Purwokerto.
