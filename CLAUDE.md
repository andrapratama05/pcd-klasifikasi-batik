# CLAUDE.md — Klasifikasi Motif Batik Banyumasan (GLCM + LBP + SVM)

## Konteks project

Ini tugas EAS mata kuliah **Pengolahan Citra Digital** (S1 Informatika, UPN "Veteran" Jawa Timur). Tugasnya mengimplementasikan dan mengembangkan program dari paper referensi. Desain eksperimen ditetapkan saat ETS (presentasi proposal ala tesis), jadi project ini adalah **pelaksanaan proposal tersebut**, bukan eksplorasi bebas.

Judul kerja: *Analisis Komparatif Ekstraksi Fitur GLCM, LBP, dan Kombinasinya untuk Klasifikasi Motif Batik Banyumasan Menggunakan Support Vector Machine.*

Riwayat singkat: dataset awal (5 motif, Mendeley 5hp2dg7n3m, 3 kelas dipakai) **ditinggalkan** karena dosen menilai 3 kelas terlalu sedikit dan motifnya terlalu berbeda. Dosen meminta kelas lebih banyak atau motif yang lebih mirip, serta dataset yang punya paper pendamping. Dataset Batik Banyumasan memenuhi ketiganya.

### Paper referensi metode
1. **Andono & Rachmawanto (2021)**, "Evaluasi Ekstraksi Fitur GLCM dan LBP Menggunakan Multikernel SVM untuk Klasifikasi Batik", Jurnal RESTI 5(1), 1–9. GLCM (4 properti, sudut 0°, d = 1, 3, 5) + LBP, SVM kernel polynomial/linear/gaussian, 8 kelas, 160 citra (uji hanya 32). Klaim: GLCM+LBP sampai 100%.
2. **Fatimah & Agustin (2025)**, "Klasifikasi Citra Batik Menggunakan LBP dan SVM", Jurnal Algoritma 22(1), 185–196. Grayscale → sharpening → adapthisteq → uniform LBP (P=8, R=1, 59 bin) → SVM linear One-vs-Rest. 3 kelas, 720 citra Kaggle, split 80/20. Akurasi 88,89%.

### Paper pada dataset yang sama
3. **Adeva, Gaisani, & Pratama (2025)**, "Klasifikasi Citra Batik Banyumasan Menggunakan Ekstraksi Fitur HOG, Reduksi Dimensi PCA, dan Klasifikasi SVM", Jurnal Ilmiah Teknologi Informasi dan Robotika (JIFTI) 7(2), UPN "Veteran" Jawa Timur. DOI: 10.33005/jifti.v7i2.193. Grayscale, resize 128×128, HOG (8×8 piksel/sel, 2×2 sel/blok, 9 orientasi), PCA 100 komponen, SVM RBF (C=10, gamma='scale'). Split stratified acak 80/20 atas 5.148 citra. Akurasi terbaik 91% (HOG+PCA+SVM). Paper tidak melaporkan apakah salinan augmentasi dipisahkan dari data uji.
4. **Syafi'i (2024)**, skripsi "Klasifikasi Citra Motif Batik Banyumasan menggunakan Algoritma CNN", Institut Teknologi Telkom Purwokerto. Pembuat dataset. Menyebut bahwa beberapa motif Banyumasan mengambil pola dasar dari motif lain sehingga sulit dibedakan.

### Celah yang dijawab
- Data uji Andono terlalu kecil (1 citra = 3,1% akurasi).
- GLCM Andono hanya diuji pada sudut 0°; Fatimah hanya memakai LBP dan kernel linear.
- Pada dataset Banyumasan, 8 dari 9 citra adalah hasil augmentasi dari foto asli. Split acak per file berisiko menaruh salinan foto yang sama di train dan test. Project ini memakai **split per foto asli** dan **data uji berisi foto asli saja**, sehingga akurasinya mengukur kemampuan mengenali motif, bukan mengenali kain yang sama.

## Pertanyaan riset, skenario, dan hipotesis

| Skenario | Fitur | Dimensi |
|---|---|---|
| S1 | GLCM sudut 0° saja (4 properti × d = 1, 3, 5) | 12 |
| S2 | GLCM 4 arah (0°, 45°, 90°, 135°), dirata-rata per jarak | 12 |
| S3 | Uniform LBP (P=8, R=1) | 59 |
| S4 | S2 + S3 (fusi) | 71 |

Setiap skenario diuji dengan 3 kernel SVM: **linear, RBF (gaussian), polynomial** → 12 model.

- **RM1 / H1**: Fusi GLCM+LBP (S4) lebih baik daripada fitur tunggal (S2 dan S3).
- **RM2 / H2**: Kernel non-linear (RBF/polynomial) lebih baik daripada linear pada fitur yang sama.
- **RM3 / H3**: GLCM 4 arah (S2) lebih baik daripada 0° saja (S1), dilihat dari rata-rata **dan** simpangan baku antar-fold. Pantau juga recall satu **kelas berpola paling berarah** (diagonal/garis dominan). Kelas ini dipilih secara visual di notebook 01 dan dicatat di bagian Status **sebelum** data uji dievaluasi.

Hipotesis boleh terbukti salah. Kalau S4 tidak lebih baik dari S3, itu tetap temuan yang sah. **Jangan pernah mengubah eksperimen supaya hipotesis terbukti.**

## Dataset

- Sumber: **Batik Banyumasan** (Syafi'i, Mendeley Data, **versi 2**, 2025), lisensi CC BY 4.0. https://data.mendeley.com/datasets/vsjrkhs9km/2 — DOI: 10.17632/vsjrkhs9km.2
- Foto diambil langsung di Rumah Batik R, Sokaraja, Banyumas (kamera ponsel 64 MP, rasio 1:1, kain dibentang di lantai, jarak 50 cm). Label dikonfirmasi pemilik rumah batik.
- **11 kelas**: Angguran, Ayam Puger, Jahe Lumbon, Jahe Puger, Jahe Srimpang, Lumbon, Madu Bronto, Pring Sedapur, Puger Galar, Puger Telu Bal, Wit Lumbon.
- 572 foto asli (52 per kelas menurut Adeva dkk.), masing-masing punya 8 salinan augmentasi → 5.148 citra, 468 per kelas (seimbang). Angka ini wajib diverifikasi saat audit.
- **Citra asli dan augmentasi berada di folder kelas yang sama**, dibedakan lewat nama file:
  - asli: `Angguran (24).jpg`
  - augmentasi: `Angguran (24)_<Jenis>.jpg`, dengan `<Jenis>` ∈ {`ChannelShuffle`, `ElasticTransform`, `FDA`, `Flip`, `RandomGridShuffle`, `RandomRotate90`, `Sharpen`, `Transpose`}
- Struktur di Mendeley: `Batik Banyumasan/<Motif>/<file>.jpg`. Hasil ekstraksi berada di `data/raw/batik_banyumasan/` **apa adanya**. Jangan memindahkan atau mengganti nama file asli.
- Ukuran citra belum diketahui. Audit wajib melaporkannya.
- Folder `data/` **tidak di-commit**.
- Angka hasil tidak bisa dibandingkan langsung dengan Andono, Fatimah, maupun Adeva dkk. (dataset, jumlah kelas, ukuran citra, atau skema split berbeda). Sebutkan ini di pembahasan.

### Pengambilan data otomatis (`src/download_data.py`)

Tujuannya: dataset cukup diunduh sekali, dan siapa pun yang meng-clone repo bisa menyiapkan data dengan satu perintah, tanpa langkah manual.

```
uv run python -m src.download_data            # unduh (bila perlu) + ekstrak + verifikasi
uv run python -m src.download_data --force    # ulangi ekstraksi dari awal
```

Perilaku yang diwajibkan (idempoten, hanya pustaka standar + `tqdm`):
1. **Kalau `data/raw/batik_banyumasan/` sudah ada dan lolos verifikasi → selesai**, tidak mengunduh atau mengekstrak ulang.
2. **Kalau ZIP sudah ada di `data/zip/` → langsung ekstrak**, tidak mengunduh ulang.
3. **Kalau belum ada → unduh** dari `DATASET_URL` di `config.py` ke `data/zip/` dengan progress bar. Unduh ke file `.part` lalu ganti nama setelah selesai, supaya unduhan yang terputus tidak dianggap lengkap.
4. **Ekstrak** ke folder sementara, tangani ZIP bertingkat (ZIP di dalam ZIP) bila ada, lalu pindahkan folder-folder motif ke `data/raw/batik_banyumasan/<Motif>/`. Abaikan file sampah seperti `__MACOSX/` dan `.DS_Store`.
5. **Verifikasi**: 11 folder motif, jumlah `.jpg` total dan per kelas, jumlah foto asli. Cetak ringkasan singkat. Kalau tidak sesuai harapan (5.148 citra, 572 asli), beri peringatan yang jelas, jangan diam-diam lanjut.
6. **Fallback manual**: kalau unduhan otomatis gagal (URL berubah, butuh login, jaringan), cetak instruksi: unduh "Download All" dari halaman Mendeley lalu letakkan ZIP di `data/zip/`, kemudian jalankan perintah yang sama.

Catatan untuk `DATASET_URL`: tautan langsung ZIP Mendeley belum dipastikan. Sebelum menulis skrip, coba cari tautan "Download All" yang valid (misalnya pola cache ZIP Mendeley `https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/vsjrkhs9km-2.zip`, atau endpoint public API Mendeley) dan uji dengan request `HEAD`. Kalau tidak ada yang berhasil, biarkan `DATASET_URL = None` dan skrip langsung memakai jalur fallback manual. Jangan mengarang URL.

### Parsing nama file → grup

```python
pattern = r"^(?P<motif>.+?) \((?P<idx>\d+)\)(?:_(?P<aug>\w+))?\.jpg$"  # case-insensitive untuk ekstensi
# origin_id = f"{motif}_{idx}"   (contoh: "Angguran_24")
# aug_type  = aug atau "original"
```

- Label diambil dari **nama folder**, lalu dicek konsisten dengan `motif` di nama file.
- Audit: setiap `origin_id` harus punya tepat 9 file (1 asli + 8 augmentasi). Laporkan pengecualian, file yang tidak cocok pola, dan file rusak.

## Pipeline (urutan wajib)

1. **Dataset dan EDA** (notebook 01) — parsing nama file, audit (jumlah per kelas, per `aug_type`, ukuran, mode warna, file rusak), contoh 1 foto asli + 8 augmentasinya, contoh 1 citra asli per kelas. Tentukan kelas berpola paling berarah untuk H3.
2. **Preprocessing dan split** (notebook 02):
   - **Split per `origin_id`**: 80% train / 20% test, stratified per kelas, `random_state=42` (cukup `train_test_split` pada tabel `origin_id` unik dengan `stratify=label`, karena satu `origin_id` hanya punya satu kelas). Hasilnya sekitar 42 foto asli per kelas untuk train dan 10 untuk test.
   - **Data uji hanya berisi foto asli.** Salinan augmentasi dari grup test dibuang seluruhnya.
   - Simpan `results/split.csv` (kolom: `filename, label, origin_id, aug_type, split`). Data uji dikunci sampai notebook 04.
   - **Tidak ada augmentasi tambahan.** Dataset sudah teraugmentasi.
   - Preprocessing (sama untuk semua skenario, dihitung per citra sehingga tidak membocorkan informasi antar-citra):
     - center-crop ke persegi bila perlu, lalu resize ke `IMG_SIZE` (default **256×256**, sesuai desain ETS; `skimage.transform.resize` dengan `anti_aliasing=True`). Kalau audit menemukan citra lebih kecil dari 256, tanyakan dulu sebelum menentukan ukuran;
     - grayscale dengan bobot 0,299R + 0,587G + 0,114B (sama dengan `rgb2gray` MATLAB dan persamaan (1) Fatimah);
     - **CLAHE**: `skimage.exposure.equalize_adapthist(clip_limit=0.01)` (setara default `adapthisteq` MATLAB), lalu kembali ke `uint8`;
     - **tanpa sharpening** (sharpening memperbesar noise yang langsung mengotori kode LBP). Catatan: salah satu augmentasi bawaan dataset adalah `Sharpen`; ini alasan tambahan menjaga augmentasi di luar data uji.
   - Visualisasi: asli → grayscale → CLAHE untuk satu citra per kelas; karakteristik data akhir (distribusi kelas train/test, histogram intensitas).
3. **Skenario dan training** (notebook 03):
   - Ekstraksi fitur S1–S3 (S4 = gabungan S2 + S3), di-cache ke `features/` (kolom: `filename, label, origin_id, aug_type, split, fitur...`).
   - **Eksperimen utama (Opsi A)**: training hanya dengan **foto asli** dari grup train.
   - `Pipeline([StandardScaler(), SVC()])` + `GridSearchCV` dengan `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, `groups=origin_id`. Fold yang sama dipakai untuk semua skenario (perbandingan berpasangan).
   - Visualisasi: skor per fold (mean ± std) per skenario × kernel; heatmap grid search C × gamma untuk RBF. SVM tidak punya kurva loss/epoch.
4. **Evaluasi dan hipotesis** (notebook 04) — evaluasi akhir di data uji, **sekali**, setelah semua hyperparameter dibekukan. Confusion matrix, metrik per kelas, pembahasan hipotesis. Perhatikan kebingungan antar-motif yang namanya serumpun (keluarga Lumbon dan keluarga Puger).
5. **Ablasi augmentasi (Opsi B, tambahan, notebook 05)** — training dengan foto asli + 8 augmentasinya dari grup train. Validasi tiap fold **hanya** memakai foto asli dari fold validasi: buat daftar `(train_idx, val_idx)` sendiri dari `StratifiedGroupKFold` atas `origin_id`, lalu buang indeks augmentasi dari `val_idx`, dan berikan daftar itu ke `GridSearchCV(cv=...)`. Bandingkan dengan Opsi A di data uji yang sama. Catatan untuk pembahasan: `RandomRotate90`, `Transpose`, dan `Flip` hanya menukar sudut GLCM (0°↔90°, 45°↔135°), sehingga hampir tidak mengubah fitur S2 yang dirata-rata 4 arah; LBP `nri_uniform` tidak invarian rotasi, jadi fitur S3 ikut berubah.

## Detail implementasi yang mudah salah

**GLCM** (`skimage.feature.graycomatrix` / `graycoprops`):
- Kuantisasi ke **8 level** sebelum membentuk GLCM (default `NumLevels=8` MATLAB, yang dipakai Andono). Jadikan `levels` parameter di `config.py`.
- `distances=[1, 3, 5]`, `angles=[0, π/4, π/2, 3π/4]`, `normed=True`, `symmetric=False` (default MATLAB).
- **Energi**: rumus paper (persamaan 7) adalah Σ P(i,j)², yaitu `'ASM'` di skimage. `'energy'` di skimage adalah akar kuadratnya. Pakai `'ASM'`.
- **Homogenitas**: paper dan MATLAB memakai 1/(1+|i−j|), sedangkan `'homogeneity'` skimage memakai 1/(1+(i−j)²). **Hitung manual** dari matriks GLCM.
- Kontras dan korelasi: `graycoprops` skimage sudah sesuai.
- S2: rata-ratakan tiap properti di 4 sudut, per jarak → tetap 12 fitur.
- Nama kolom jelas, misalnya `glcm_contrast_d1_a0`, `glcm_asm_d3_mean`.

**LBP** (`skimage.feature.local_binary_pattern`):
- `P=8, R=1, method='nri_uniform'` → 59 nilai (0–58), setara `extractLBPFeatures` MATLAB dengan `Upright=true`.
- Histogram 59 bin, `range=(0, 59)`, lalu normalisasi L2.

**SVM** (`sklearn.svm.SVC`):
- Grid: `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {0.001, 0.01, 0.1, 1}` (RBF dan poly), `degree ∈ {2, 3}` (poly).
- `class_weight='balanced'` untuk semua kernel (data seimbang, jadi efeknya netral; dipertahankan supaya konsisten dengan desain).
- Metrik pemilihan model: `f1_macro`; akurasi tetap dilaporkan.
- `SVC` multikelas memakai One-vs-One. Replikasi baseline Fatimah (S3 + linear + `OneVsRestClassifier`) boleh jadi satu baris pembanding tambahan.
- Scaler **harus di dalam Pipeline**, jangan di-fit di seluruh data sebelum CV.

## Output yang harus dihasilkan

- `results/split.csv` — pembagian train/test per file.
- `results/cv_results.csv` — 12 baris: skenario, kernel, best params, mean ± std akurasi dan F1 makro antar-fold, plus skor per fold.
- `results/test_results.csv` — 12 baris: akurasi, precision/recall/F1 per kelas, F1 makro di data uji.
- `results/ablation_aug.csv` — hasil Opsi B (kalau dikerjakan).
- `figures/01_dataset/`, `figures/02_preprocessing/`, `figures/03_training/`, `figures/04_evaluasi/`, `figures/05_ablasi/`:
  - distribusi kelas dan contoh 1 foto asli + 8 augmentasinya;
  - tahap preprocessing per kelas; peta LBP dan contoh matriks GLCM (untuk slide);
  - grouped bar chart akurasi/F1 per skenario × kernel dengan error bar std; heatmap grid search;
  - confusion matrix model terbaik per skenario di data uji.
- Bagian **Pembahasan Hipotesis** di notebook 04: tiap hipotesis diberi status *didukung / tidak didukung / tidak meyakinkan*, dengan angka dan alasannya. Karena hanya 5 fold, selisih yang lebih kecil dari std ditulis *tidak meyakinkan*.

## Struktur repo

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml          # dikelola uv
├── uv.lock
├── .python-version
├── .gitignore              # data/, features/, .venv/, __pycache__/, .ipynb_checkpoints/
├── data/                   # tidak di-commit
│   ├── zip/                # ZIP unduhan (disimpan supaya tidak mengunduh ulang)
│   ├── raw/batik_banyumasan/<Motif>/
│   └── demo/               # foto baru untuk uji coba (hanya README yang di-commit)
├── src/
│   ├── __init__.py
│   ├── config.py           # path, seed, DATASET_URL, CLASSES, IMG_SIZE, levels, distances, angles, P, R, grid
│   ├── download_data.py    # unduh + ekstrak + verifikasi (idempoten)
│   ├── data.py             # parse_filename(), audit, make_split()
│   ├── preprocess.py
│   ├── features.py         # glcm_features(), lbp_features()
│   ├── evaluate.py         # metrik, confusion matrix
│   └── predict.py          # predict_image() + CLI, memakai models/best_model.*
├── notebooks/
│   ├── 01_dataset_dan_eda.ipynb
│   ├── 02_preprocessing_dan_split.ipynb
│   ├── 03_skenario_dan_training.ipynb
│   ├── 04_evaluasi_dan_hipotesis.ipynb
│   ├── 05_ablasi_augmentasi.ipynb      # tambahan
│   ├── 06_demo_klasifikasi.ipynb       # uji coba model terbaik
│   └── 07_diagnostik_model.ipynb       # learning/validation curve, kebocoran data
├── tests/
│   ├── test_features.py
│   ├── test_data.py
│   └── test_predict.py
├── features/               # cache fitur, tidak di-commit
├── models/                 # best_model.joblib + best_model.json (dari notebook 04)
├── results/
└── figures/
```

Logika yang dipakai ulang ditaruh di `src/`, notebook mengimpor dari sana. Sel pertama setiap notebook menambahkan root repo ke `sys.path` (`Path.cwd().parent`) supaya `from src.features import ...` berjalan. Notebook berisi alur, visualisasi, dan penjelasan.

## README.md

Ditulis dalam Bahasa Indonesia baku, untuk dosen, anggota kelompok, dan orang yang meng-clone repo. Isi minimal:
1. **Judul dan deskripsi tugas**: mata kuliah Pengolahan Citra Digital, UPN "Veteran" Jawa Timur; konteks ETS (proposal) dan EAS (implementasi); judul kerja; anggota kelompok (placeholder dulu).
2. **Ringkasan riset**: tujuan, 4 skenario × 3 kernel, hipotesis H1–H3 (singkat), dan alasan split per foto asli.
3. **Dataset**: sumber, sitasi lengkap dengan DOI, lisensi CC BY 4.0, jumlah kelas dan citra.
4. **Struktur repo** (ringkas, satu baris penjelasan per folder).
5. **Cara menjalankan**, langkah demi langkah dan bisa di-copy:
   - prasyarat (Python ≥ 3.10, uv, Git);
   - `git clone ...` lalu `uv sync`;
   - `uv run python -m src.download_data` (beserta fallback manual);
   - `uv run pytest`;
   - membuka notebook (`uv run jupyter lab` atau VS Code dengan kernel `.venv`) dan menjalankan 01 → 05 berurutan;
   - menjalankan ulang semua notebook sekaligus dengan `nbconvert`;
   - perkiraan waktu jalan tiap notebook (isi setelah benar-benar dijalankan).
6. **Hasil**: tabel ringkas dari `results/` dan tautan ke gambar utama. Selama belum dijalankan, tulis placeholder, jangan mengarang angka.
7. **Referensi**: paper 1–4 di atas.

## Environment

- Dikelola dengan **uv**. Python ≥ 3.10.
- Dependensi: `uv add numpy pandas scikit-image scikit-learn matplotlib pillow tqdm`; dev: `uv add --dev jupyter ipykernel pytest`. Hindari dependensi lain kecuali benar-benar perlu.
- Semua perintah lewat `uv run` (misalnya `uv run pytest`).
- Dijalankan **lokal** (laptop, VS Code + Jupyter, kernel dari `.venv`). Tanpa GPU. Pakai `n_jobs=-1` di GridSearchCV dan cache fitur.
- Semua path relatif terhadap root repo (lewat `src/config.py`), tidak ada path absolut.
- Seed global `42` untuk semua hal yang acak.

## Uji kebenaran

`tests/test_features.py`:
- Citra konstan → kontras GLCM = 0, ASM = 1, homogenitas = 1.
- Citra garis vertikal berselang-seling → kontras sudut 0° jauh lebih besar dari sudut 90°.
- Histogram LBP punya tepat 59 bin dan norma L2 = 1.
- Homogenitas manual cocok dengan rumus 1/(1+|i−j|) pada matriks kecil yang dihitung tangan.

`tests/test_data.py`:
- `parse_filename()` benar untuk contoh `Angguran (24).jpg`, `Angguran (24)_RandomRotate90.jpg`, dan nama motif dua kata seperti `Puger Telu Bal (3)_FDA.jpg`.
- Tidak ada `origin_id` yang muncul di train dan test sekaligus.
- Data uji tidak berisi `aug_type` selain `original`.

## Aturan kerja untuk Claude Code

- **Bahasa**: diskusi dalam Bahasa Indonesia. Markdown di notebook ditulis dalam Bahasa Indonesia baku, istilah teknis tetap (dataset, fold, kernel, dll.). Nama variabel, fungsi, dan komentar kode dalam bahasa Inggris.
- **Level**: aku mahasiswa Informatika tingkat menengah. Jelaskan *kenapa* sebuah langkah dilakukan, tidak perlu definisi dasar Python/ML. Untuk konsep pengolahan citra yang lebih dalam, beri penjelasan singkat di markdown cell sebelum kodenya.
- **Jangan ubah desain riset** (skenario, preprocessing, parameter GLCM/LBP, grid, skema split) tanpa bertanya dulu. Kalau menemukan masalah di desain, jelaskan dan usulkan opsi.
- **Jangan menyentuh data uji** untuk keputusan apa pun (pemilihan fitur, parameter, threshold, kelas pantauan H3).
- **Jangan mengarang hasil.** Semua angka di markdown dan tabel harus berasal dari sel yang benar-benar dijalankan. Kalau belum dijalankan, tulis placeholder yang jelas.
- Kerjakan bertahap: selesaikan dan jalankan satu notebook sebelum lanjut ke berikutnya. Tunjukkan ringkasan hasil di akhir tiap tahap.
- Sebelum menyatakan selesai: `uv run pytest` dan eksekusi ulang semua notebook dari kernel bersih (`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`).
- Sel kode pendek dan fokus. Hindari output panjang (print ribuan baris, warning berulang).
- Commit kecil dengan pesan yang jelas per tahap, hanya kalau aku minta.

## Status dan yang masih perlu diisi

- Tahap sekarang: **notebook 01–05 selesai dan dijalankan (24-11-2026); rangkuman di `results/ringkasan_hasil.md`.** Berikutnya: keputusan soal batasan "kain yang sama" (lihat di bawah).
- Tahap uji coba klasifikasi (25-11-2026): notebook 04 menyimpan model terbaik menurut CV (S2 + SVM linear, C=100) **tanpa training ulang** ke `models/best_model.joblib` + `models/best_model.json` (skenario, kernel, params, urutan `feature_columns`, `IMG_SIZE`, kelas, versi scikit-learn), serta prediksi 12 model di data uji ke `results/test_predictions.csv`. `src/predict.py` berisi `predict_image(path)` (preprocessing + fitur yang sama persis, label + top-3 dari `decision_function`) dan CLI `uv run python -m src.predict <file|folder>`. Notebook `06_demo_klasifikasi.ipynb` mendemokan beberapa citra uji dan foto baru di `data/demo/` (foto tidak di-commit, hanya README-nya). `tests/test_predict.py` memastikan `predict_image` menghasilkan label yang sama dengan prediksi notebook 04 untuk ke-115 citra uji (di-skip bila data/model belum ada).
- Diagnostik model (notebook 07, 26-11-2026; hanya data latih): tidak underfitting; overfitting ringan (F1 latih ≈ 1 vs validasi 0,94–0,98), hyperparameter terpilih di luar daerah overfitting; learning curve belum mendatar. Permutation test (100×): F1 label acak ±0,08, p = 0,01 → pipeline tidak bocor. Dataset punya 5 pasangan foto asli identik (Lumbon 17=25, 18=26, 19=27, 24=28; Madu Bronto 41=42), semua di data latih; efek ke CV −0,009 s.d. +0,001. CV per blok nomor foto (1–13, 14–26, 27–39, 40–52) menurunkan F1 0,13–0,19 (S2 0,821 ± 0,102; S4 0,824 ± 0,074), mengonfirmasi kebocoran kain. Ringkasan di `results/diagnostik_model.csv`.
- Temuan validitas (notebook 04, bagian 5): foto uji hampir selalu punya foto latih dari kain fisik yang sama (tetangga terdekat selalu berlabel sama; selisih nomor foto bermedian 2). Split per foto asli mencegah kebocoran augmentasi, tetapi tidak mencegah kebocoran kain. Belum ada keputusan untuk mengubah desain.
- [ ] Konfirmasi ke dosen: Opsi A (foto asli saja) sebagai eksperimen utama, Opsi B sebagai ablasi
- [x] Ukuran citra asli: semua 5.148 citra 512×512 RGB (audit notebook 01). `IMG_SIZE` final = **256** (dikonfirmasi 24-11-2026: tanpa upscaling, jarak GLCM d = 1, 3, 5 menangkap pola motif alih-alih serat kain/noise, piksel 4× lebih sedikit; rincian di `note.md` lokal, tidak di-commit)
- [x] Kelas pantauan H3: **Puger Galar** (dikonfirmasi 24-11-2026, sebelum ada hasil model; `config.H3_CLASS`). Alasan: satu-satunya kelas dengan latar garis lurus sejajar (galar) yang arahnya berbeda antar-foto; indeks anisotropi GLCM (range/mean kontras 4 sudut, foto asli latih saja) terkecilnya 0,902 > nilai terbesar kelas lain 0,614. Cadangan yang tidak dipakai: Pring Sedapur.
- [ ] Deadline EAS dan format pengumpulan final
- [ ] Nama anggota kelompok (untuk README)