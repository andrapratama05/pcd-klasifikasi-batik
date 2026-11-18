# CLAUDE.md — Klasifikasi Motif Batik (GLCM + LBP + SVM)

## Konteks project

Ini tugas EAS mata kuliah **Pengolahan Citra Digital** (S1 Informatika, UPN "Veteran" Jawa Timur). Tugasnya mengimplementasikan dan mengembangkan program dari paper referensi. Desain eksperimennya sudah ditetapkan saat ETS (presentasi proposal ala tesis), jadi notebook ini adalah **pelaksanaan proposal tersebut**, bukan eksplorasi bebas.

Judul kerja: *Analisis Komparatif Ekstraksi Fitur GLCM, LBP, dan Kombinasinya untuk Klasifikasi Motif Batik Menggunakan Support Vector Machine.*

Paper referensi:
1. **Andono & Rachmawanto (2021)**, "Evaluasi Ekstraksi Fitur GLCM dan LBP Menggunakan Multikernel SVM untuk Klasifikasi Batik", Jurnal RESTI 5(1), 1–9. GLCM (4 properti, sudut 0°, d = 1, 3, 5) + LBP, SVM kernel polynomial/linear/gaussian, 8 kelas, 160 citra (uji hanya 32). Klaim: GLCM+LBP sampai 100%.
2. **Fatimah & Agustin (2025)**, "Klasifikasi Citra Batik Menggunakan LBP dan SVM", Jurnal Algoritma 22(1), 185–196. Grayscale → sharpening → adapthisteq → uniform LBP (P=8, R=1, 59 bin) → SVM linear One-vs-Rest. 3 kelas (Kawung, Megamendung, Parang), 720 citra Kaggle, split 80/20. Akurasi 88,89%.

Celah yang dijawab: data uji Andono terlalu kecil (1 citra = 3,1% akurasi) dan rawan kebocoran data; GLCM hanya diuji di sudut 0°; Fatimah hanya memakai LBP dan kernel linear.

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
- **RM3 / H3**: GLCM 4 arah (S2) lebih baik daripada 0° saja (S1), dilihat dari rata-rata **dan** simpangan baku antar-fold. Perhatikan juga recall kelas Parang (motif diagonal, paling sensitif terhadap arah).

Hipotesis boleh terbukti salah. Kalau S4 tidak lebih baik dari S3, itu tetap temuan yang sah. **Jangan pernah mengubah eksperimen supaya hipotesis terbukti.**

## Dataset

- Sumber: **Batik Kawung, Megamendung, Parang, Sekarjagad, and Truntum Image Dataset** (Putra & Zamzami, Mendeley Data, v1, 2025), lisensi CC BY 4.0. https://data.mendeley.com/datasets/5hp2dg7n3m/1 — DOI: 10.17632/5hp2dg7n3m.1
- Isi menurut halaman dataset: 2.448 citra JPG RGB, **sudah di-crop dan di-resize ke 224×224**. Tidak disebutkan ada augmentasi. Per kelas: Kawung 626, Megamendung 390, Parang 804, Sekarjagad 339, Truntum 289.
- **Eksperimen utama memakai 3 kelas saja: Kawung, Megamendung, Parang** (1.820 citra), sesuai desain ETS dan kelas yang sama dengan Fatimah. Sekarjagad dan Truntum diabaikan untuk 12 model utama; eksperimen 5 kelas boleh jadi tambahan di akhir kalau waktu cukup.
- Data **tidak seimbang** (Parang ≈ 2× Megamendung). Penanganannya: semua citra tetap dipakai, split stratified, `class_weight='balanced'` di SVM, dan F1 makro sebagai metrik pemilihan model. Jangan undersampling tanpa bertanya.
- Data diunduh manual dari Mendeley (ZIP), lalu diekstrak ke `data/raw/`. Struktur folder di dalam ZIP belum diketahui. Saat audit, periksa struktur aslinya dulu dan tulis loader yang memetakan nama folder ke label secara case-insensitive (misalnya `Kawung`, `kawung`, `Batik Kawung` → `kawung`). Jangan memindahkan atau mengganti nama file asli.
- Folder `data/` **tidak di-commit** (masukkan ke `.gitignore`).
- Angka hasil tidak bisa dibandingkan langsung dengan Fatimah (dataset Kaggle 720 citra, seimbang) maupun Andono (8 kelas, foto sendiri). Sebutkan ini di pembahasan.

## Pipeline (urutan wajib)

1. **Audit data** — hitung citra per kelas, cek file rusak, ukuran, mode warna (RGBA/palette → RGB).
2. **Cek duplikat dan near-duplicate** — dataset batik Kaggle sering berisi citra hasil augmentasi atau foto kain yang sama. Hitung perceptual hash (dHash 64-bit, cukup implementasi numpy + PIL) dan kelompokkan citra dengan jarak Hamming ≤ 5 ke dalam satu `group_id`. Laporkan jumlah grup duplikat.
3. **Split dulu, baru proses** — 80% train / 20% test, stratified per kelas dan **group-aware** (satu grup duplikat tidak boleh terbelah antara train dan test). Gunakan `random_state=42`. Simpan daftar file per split ke `results/split.csv`. Data uji dikunci sampai tahap evaluasi akhir.
4. **Preprocessing** (sama untuk semua skenario):
   - **Pertahankan ukuran asli 224×224, jangan di-resize** (desain ETS menyebut 256×256, tapi dataset ini sudah 224×224; memperbesar ke 256 berarti interpolasi yang menghaluskan tekstur dan mengubah nilai GLCM/LBP). Audit wajib memastikan semua citra memang 224×224. Kalau ada yang tidak, center-crop ke persegi lalu resize ke 224×224, dan laporkan jumlahnya.
   - Grayscale dengan bobot 0,299R + 0,587G + 0,114B (sama dengan `rgb2gray` MATLAB dan persamaan (1) Fatimah).
   - **CLAHE**: `skimage.exposure.equalize_adapthist` dengan `clip_limit=0.01` (setara default `adapthisteq` MATLAB). Konversi kembali ke `uint8`.
   - **Tanpa sharpening** (keputusan desain: sharpening memperbesar noise yang langsung mengotori kode LBP). Ablasi dengan vs tanpa sharpening boleh ditambahkan sebagai eksperimen tambahan, terpisah dari 12 model utama.
5. **Ekstraksi fitur** — hasilnya di-cache ke `features/*.csv` (kolom: `filename, label, group_id, split, fitur...`) supaya tidak dihitung ulang.
6. **Pelatihan + tuning** di data train saja: `Pipeline([StandardScaler(), SVC()])` + `GridSearchCV` dengan `StratifiedGroupKFold(n_splits=5)`. Fold yang sama dipakai untuk semua skenario (perbandingan berpasangan).
7. **Evaluasi akhir** di data uji, sekali, setelah semua hyperparameter dibekukan.
8. **Analisis hipotesis** dan tabel/grafik untuk laporan dan presentasi.

## Detail implementasi yang mudah salah

**GLCM** (`skimage.feature.graycomatrix` / `graycoprops`):
- Kuantisasi ke **8 level** sebelum membentuk GLCM (default `NumLevels=8` di MATLAB `graycomatrix`, yang dipakai Andono; nilai kontras di Tabel 2 Andono, sekitar 0,2–2,9, konsisten dengan 8 level). Jadikan `levels` parameter supaya bisa diubah.
- `distances=[1, 3, 5]`, `angles=[0, π/4, π/2, 3π/4]`, `normed=True`. `symmetric=False` (default MATLAB).
- **Energi**: rumus paper (persamaan 7) adalah Σ P(i,j)², itu `'ASM'` di skimage. `'energy'` di skimage adalah akar kuadratnya. Pakai `'ASM'`.
- **Homogenitas**: rumus paper (persamaan 8) dan MATLAB memakai 1/(1+|i−j|). `'homogeneity'` di skimage memakai 1/(1+(i−j)²). **Hitung manual** dari matriks GLCM supaya sesuai paper.
- Kontras dan korelasi: `graycoprops` skimage sudah sesuai.
- S2: rata-ratakan tiap properti di 4 sudut, per jarak → tetap 12 fitur.
- Beri nama kolom yang jelas, misalnya `glcm_contrast_d1_a0`, `glcm_asm_d3_mean`.

**LBP** (`skimage.feature.local_binary_pattern`):
- `P=8, R=1, method='nri_uniform'` → 59 nilai (0–58), setara `extractLBPFeatures` MATLAB dengan `Upright=true`.
- Histogram 59 bin, `range=(0, 59)`, lalu normalisasi L2 (default MATLAB, sesuai skala nilai di Tabel 1 Fatimah).

**SVM** (`sklearn.svm.SVC`):
- Grid: `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {0.001, 0.01, 0.1, 1}` (RBF dan poly), `degree ∈ {2, 3}` (poly).
- `class_weight='balanced'` untuk semua kernel (data tidak seimbang).
- Metrik pemilihan model: `f1_macro`; akurasi tetap dilaporkan.
- `SVC` multikelas secara internal memakai One-vs-One. Fatimah memakai One-vs-Rest. Untuk 12 model utama cukup `SVC` biasa; replikasi baseline Fatimah (S3 + linear + `OneVsRestClassifier`) boleh dijadikan satu baris pembanding tambahan.
- Scaler **harus di dalam Pipeline**, jangan di-fit di seluruh data sebelum CV (itu kebocoran data).

## Output yang harus dihasilkan

- `results/cv_results.csv` — 12 baris: skenario, kernel, best params, mean ± std akurasi dan F1 makro antar-fold, plus skor per fold.
- `results/test_results.csv` — 12 baris: akurasi, precision/recall/F1 per kelas, F1 makro di data uji.
- `figures/`:
  - contoh tiap tahap preprocessing (asli → grayscale → CLAHE) untuk satu citra per kelas;
  - visualisasi peta LBP dan contoh matriks GLCM (untuk slide presentasi);
  - grouped bar chart akurasi/F1 per skenario × kernel dengan error bar std;
  - confusion matrix model terbaik per skenario di data uji.
- Bagian **Pembahasan Hipotesis** di notebook: tiap hipotesis diberi status *didukung / tidak didukung / tidak meyakinkan*, dengan angka dan alasannya. Karena hanya 5 fold, selisih kecil (lebih kecil dari std) ditulis sebagai *tidak meyakinkan*, bukan *didukung*.
- Perbandingan dengan paper referensi ditulis hati-hati: dataset, jumlah kelas, dan split berbeda, jadi angka tidak bisa dibandingkan langsung.

## Struktur repo

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore              # data/, features/, __pycache__/, .ipynb_checkpoints/
├── data/raw/<kelas>/       # tidak di-commit
├── src/
│   ├── config.py           # path, seed, parameter (levels, distances, angles, P, R, grid)
│   ├── data.py             # load, audit, dHash, group-aware split
│   ├── preprocess.py
│   ├── features.py         # glcm_features(), lbp_features()
│   └── evaluate.py         # metrik, plot confusion matrix
├── notebooks/
│   ├── 01_audit_dan_split.ipynb
│   ├── 02_preprocessing_dan_fitur.ipynb
│   ├── 03_eksperimen_svm.ipynb
│   └── 04_hasil_dan_hipotesis.ipynb
├── tests/test_features.py  # uji kecil untuk GLCM/LBP
├── features/
├── results/
└── figures/
```

Logika yang dipakai ulang ditaruh di `src/`, notebook mengimpor dari sana. Notebook berisi alur, visualisasi, dan penjelasan.

## Environment

- Python ≥ 3.10. Library: `numpy`, `pandas`, `scikit-image`, `scikit-learn`, `matplotlib`, `pillow`, `tqdm`, `jupyter`. Hindari dependensi lain kecuali benar-benar perlu.
- Dijalankan **lokal** (laptop, VS Code + Jupyter). Buat virtual environment `.venv` dan pastikan komputasi tetap ringan: 1.820 citra 224×224 seharusnya cukup tanpa GPU. Pakai `n_jobs=-1` di GridSearchCV dan cache fitur supaya tidak dihitung ulang.
- Semua path relatif terhadap root repo (lewat `src/config.py`), tidak ada path absolut.
- Seed global `42` untuk semua hal yang acak.

## Uji kebenaran (tests/test_features.py)

- Citra konstan → kontras GLCM = 0, ASM = 1, homogenitas = 1.
- Citra garis vertikal berselang-seling → kontras sudut 0° jauh lebih besar dari sudut 90°.
- Histogram LBP punya tepat 59 bin dan norma L2 = 1.
- Bandingkan homogenitas manual dengan rumus 1/(1+|i−j|) pada matriks kecil yang dihitung tangan.

## Aturan kerja untuk Claude Code

- **Bahasa**: diskusi dengan aku dalam Bahasa Indonesia. Markdown di notebook ditulis dalam Bahasa Indonesia baku tapi istilah teknis tetap (dataset, fold, kernel, dll.). Nama variabel, fungsi, dan komentar kode dalam bahasa Inggris.
- **Level**: aku mahasiswa Informatika tingkat menengah. Jelaskan *kenapa* sebuah langkah dilakukan, tidak perlu definisi dasar Python/ML. Untuk konsep pengolahan citra yang lebih dalam, beri penjelasan singkat di markdown cell sebelum kodenya.
- **Jangan ubah desain riset** (skenario, preprocessing, parameter GLCM/LBP, grid, skema split) tanpa bertanya dulu. Kalau menemukan masalah di desain, jelaskan masalahnya dan usulkan opsi.
- **Jangan menyentuh data uji** untuk keputusan apa pun (pemilihan fitur, parameter, threshold).
- **Jangan mengarang hasil.** Semua angka di markdown dan tabel harus berasal dari sel yang benar-benar dijalankan. Kalau sesuatu belum dijalankan, tulis placeholder yang jelas.
- Kerjakan bertahap: selesaikan dan jalankan satu notebook sebelum lanjut ke berikutnya. Tunjukkan ringkasan hasil di akhir tiap tahap.
- Sebelum menyatakan selesai: jalankan `pytest` dan eksekusi ulang semua notebook dari kernel bersih (`jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`).
- Sel kode pendek dan fokus. Hindari output panjang yang tidak perlu (print ribuan baris, warning berulang).
- Commit kecil dengan pesan yang jelas per tahap, hanya kalau aku minta.

## Status dan yang masih perlu diisi

- Tahap sekarang: **cek progres**, bukan pengumpulan final. Prioritaskan pipeline yang jalan end-to-end dan hasil awal yang jujur dibanding tampilan yang rapi.
- [ ] Deadline EAS dan format pengumpulan final
- [ ] Nama anggota kelompok (untuk README)