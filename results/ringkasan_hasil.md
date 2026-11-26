# Ringkasan Hasil dan Kesimpulan

**Klasifikasi Motif Batik Banyumasan dengan GLCM, LBP, dan SVM**
Dijalankan 24-11-2026. Semua angka disalin dari file di folder `results/` (sumbernya disebut di tiap tabel) dan dibulatkan ke 3 desimal. Rincian lengkap ada di notebook 03–05.

## 1. Data dan skema evaluasi

| Hal | Nilai |
|---|---|
| Dataset | Batik Banyumasan v2 (Mendeley, CC BY 4.0), 11 kelas |
| Isi | 572 foto asli + 4.576 augmentasi = 5.148 citra, semua 512×512 RGB, tanpa file rusak |
| Split | Per foto asli (`origin_id`), 80/20 stratified, `random_state=42` |
| Data latih | 457 foto asli (Opsi A) / 4.113 citra termasuk augmentasi (Opsi B) |
| Data uji | 115 foto asli (10–11 per kelas), tanpa augmentasi |
| Preprocessing | resize 256×256 → grayscale → CLAHE (clip 0,01), tanpa sharpening |
| Validasi | `StratifiedGroupKFold` 5 fold (fold yang sama untuk semua model), metrik `f1_macro` |
| Aturan status | Selisih F1 lebih kecil dari std fold terbesar → *tidak meyakinkan* |

## 2. Hasil eksperimen utama (Opsi A: latih dengan foto asli saja)

Sumber: `cv_results.csv`, `test_results.csv`. ★ = kernel terbaik per skenario menurut validasi silang.

| Skenario | Fitur | Kernel | F1 makro CV (mean ± std) | F1 makro uji | Akurasi uji |
|---|---|---|---|---|---|
| S1 | GLCM 0° (12) | linear | 0,927 ± 0,016 | 0,923 | 0,922 |
| | | **RBF ★** | **0,954 ± 0,007** | 0,965 | 0,965 |
| | | poly | 0,930 ± 0,034 | 0,983 | 0,983 |
| S2 | GLCM 4 arah (12) | **linear ★** | **0,980 ± 0,011** | 1,000 | 1,000 |
| | | RBF | 0,973 ± 0,027 | 1,000 | 1,000 |
| | | poly | 0,974 ± 0,018 | 1,000 | 1,000 |
| S3 | LBP (59) | linear | 0,916 ± 0,004 | 0,916 | 0,922 |
| | | **RBF ★** | **0,938 ± 0,033** | 0,953 | 0,957 |
| | | poly | 0,892 ± 0,035 | 0,896 | 0,896 |
| S4 | S2 + S3 (71) | **linear ★** | **0,950 ± 0,013** | 0,961 | 0,965 |
| | | RBF | 0,949 ± 0,024 | 0,982 | 0,983 |
| | | poly | 0,936 ± 0,036 | 0,911 | 0,913 |

**Model terbaik:** S2 (GLCM 4 arah) dengan kernel linear. Validasi silangnya F1 0,980 ± 0,011, dan di data uji ia mengklasifikasikan seluruh 115 foto dengan benar.

## 3. Kesimpulan hipotesis

Sumber: `hypothesis_tests.csv`, `cv_oof_per_class.csv`. Status ditentukan dari validasi silang; data uji sebagai pembanding.

| Hipotesis | Perbandingan | Selisih F1 CV | Status |
|---|---|---|---|
| **H1** fusi lebih baik | S4 linear vs S2 linear | −0,030 | **Tidak didukung** (S4 lebih buruk) |
| | S4 linear vs S3 RBF | +0,012 | Tidak meyakinkan |
| **H2** non-linear lebih baik | S1: RBF vs linear | +0,026 | Didukung |
| | S2: poly vs linear | −0,006 | Tidak meyakinkan |
| | S3: RBF vs linear | +0,023 | Tidak meyakinkan |
| | S4: RBF vs linear | −0,001 | Tidak meyakinkan |
| **H3** 4 arah lebih baik dari 0° | S2 linear vs S1 RBF (terbaik) | +0,026 | **Didukung** |
| | per kernel: linear / RBF / poly | +0,053 / +0,020 / +0,044 | Didukung / tidak meyakinkan / didukung |

**H1: tidak didukung.** Menggabungkan LBP ke GLCM justru menurunkan skor dibanding GLCM 4 arah saja. Dugaannya: setelah normalisasi, 59 fitur LBP mendominasi 12 fitur GLCM, padahal LBP sendiri lebih lemah.

**H2: hanya didukung pada S1.** Pada fitur yang sudah kuat (S2), kelas-kelasnya sudah hampir terpisah secara linear, sehingga kernel non-linear tidak menambah apa-apa.

**H3: didukung untuk rata-rata F1, tetapi tidak untuk kestabilan.**
- Std antar-fold S2 lebih kecil pada linear (0,011 vs 0,016) dan poly (0,018 vs 0,034), tetapi lebih besar pada RBF (0,027 vs 0,007).
- Recall kelas pantauan **Puger Galar** (dipilih sebelum ada hasil karena polanya paling berarah):

| | Linear | RBF | Poly |
|---|---|---|---|
| Out-of-fold, 41 foto (S1 → S2) | 0,878 → 1,000 | 0,976 → 1,000 | 0,902 → 0,951 |
| Data uji, 11 foto (S1 → S2) | 0,818 → 1,000 | 1,000 → 1,000 | 1,000 → 1,000 |

Arahnya sesuai prediksi, tetapi buktinya lemah karena skor sudah mendekati maksimum.

**Motif serumpun.** Sebagian besar kesalahan di data uji jatuh ke kelas yang serumpun nama (keluarga Lumbon, Puger, Jahe):

| Model | Kesalahan ke kelas serumpun |
|---|---|
| S1 | 3 dari 4 |
| S3 | 3 dari 5 |
| S4 | 4 dari 4 |
| Seandainya acak | sekitar 22% |

Pasangan tersering adalah Jahe Lumbon → Lumbon. Jumlah kesalahannya kecil, jadi ini hanya indikasi.

## 4. Ablasi augmentasi (Opsi B: latih dengan foto asli + 8 augmentasi)

Sumber: `ablation_aug.csv`. Validasi tetap hanya memakai foto asli, dan fold sama dengan Opsi A.

| Skenario | Kenaikan F1 CV (B − A), linear / RBF / poly | F1 CV terbaik Opsi B |
|---|---|---|
| S1 | −0,001 / +0,033 / +0,047 | 0,987 (RBF) |
| S2 | 0,000 / +0,022 / +0,022 | 0,995 (RBF dan poly) |
| S3 | +0,060 / +0,049 / +0,056 | 0,987 (RBF) |
| S4 | +0,039 / +0,042 / +0,038 | 0,991 (RBF) |

- **Augmentasi membantu 9 dari 12 model** secara meyakinkan. Yang paling diuntungkan adalah LBP (S3).
- **S2 hampir tidak diuntungkan.** Bagi GLCM yang dirata-rata 4 arah, augmentasi `Flip`, `RandomRotate90`, dan `Transpose` praktis duplikat: jaraknya ke foto asli hanya 0,006 relatif terhadap jarak antar-foto berbeda. Sebaliknya, bagi LBP salinan ini berbeda nyata (0,36–0,58).
- **`Sharpen` mengubah fitur sejauh foto yang berbeda** (jarak relatif 1,04–1,10). Ini mendukung keputusan untuk tidak memakai sharpening dan tidak memasukkan augmentasi ke data uji.
- **Data uji sudah jenuh.** Dengan Opsi B, 6 dari 12 model mencapai F1 uji 1,000, sehingga perbandingan A vs B dibaca dari validasi silang. Selisih antar-skenario terbaik pada Opsi B semuanya lebih kecil dari std.

## 5. Diagnostik model: overfitting, underfitting, dan kebocoran data

Sumber: `diagnostik_model.csv`, notebook 07. Semua analisis hanya memakai data latih.

| Skenario | F1 latih | F1 validasi | Selisih | F1 label acak (permutation test) | F1 CV per blok nomor foto |
|---|---|---|---|---|---|
| S1 RBF | 0,995 | 0,954 | 0,041 | 0,083 | 0,764 ± 0,108 |
| S2 linear | 1,000 | 0,980 | 0,020 | 0,081 | 0,821 ± 0,102 |
| S3 RBF | 1,000 | 0,938 | 0,062 | 0,084 | 0,783 ± 0,035 |
| S4 linear | 1,000 | 0,950 | 0,050 | 0,085 | 0,824 ± 0,074 |

- **Tidak underfitting.** Pada validation curve, skor latih dan validasi baru turun bersama ketika C dikecilkan jauh di bawah nilai terpilih.
- **Overfitting ringan dan wajar.** Skor latih hampir 1 dengan selisih 0,02–0,06. C dan gamma terpilih berada di puncak skor validasi, sebelum daerah overfitting (pada RBF, gamma besar membuat skor validasi anjlok). Learning curve masih naik di ujung, jadi menambah data kemungkinan masih membantu.
- **Pipeline tidak bocor.** Dengan label diacak 100 kali, F1 turun ke tingkat tebakan (±0,08; 1/11 ≈ 0,09), dengan p = 0,01. Pemeriksaan split juga bersih: 0 `origin_id` tumpang tindih, 0 augmentasi di data uji, dan 0 augmentasi foto uji di data latih.
- **Ada 5 pasangan foto asli identik** di dataset: Lumbon 17 = 25, 18 = 26, 19 = 27, 24 = 28, dan Madu Bronto 41 = 42. Semuanya berada di data latih. Menggabungkan tiap pasangan ke fold yang sama hanya mengubah F1 CV sebesar −0,009 sampai +0,001.
- **Kebocoran kain terkonfirmasi.** Validasi silang yang memisahkan foto per blok nomor foto (1–13, 14–26, 27–39, 40–52) menurunkan F1 sebesar 0,13–0,19. Ini perkiraan yang lebih realistis untuk foto dari kain atau sesi pemotretan lain. Dengan skema ini, S4 (0,824) dan S2 (0,821) setara, dan semua selisih antar-skenario lebih kecil dari std.

## 6. Batasan penting

1. **Kain yang sama ada di data latih dan uji.** Setiap foto uji punya tetangga terdekat di data latih dengan label sama, dan selisih nomor fotonya bermedian 2. Secara visual keduanya adalah kain fisik yang sama yang difoto berurutan (`figures/04_evaluasi/foto_uji_vs_latih_terdekat.png`).
   - Split per foto asli mencegah kebocoran augmentasi, tetapi tidak mencegah kebocoran kain.
   - Angka absolut (sampai 100%) mengukur kemampuan mengenali kain-kain di dataset ini, bukan generalisasi ke kain baru.
   - Perbandingan antar-skenario tetap adil karena semua skenario terkena batasan yang sama.
2. **Ukuran data kecil.** Satu foto uji setara 0,87 poin persentase akurasi, dan validasi silang hanya 5 fold. Karena itu banyak selisih berstatus *tidak meyakinkan*.
3. **Tidak dapat dibandingkan langsung dengan paper rujukan.** Andono & Rachmawanto (2021), Fatimah & Agustin (2025), dan Adeva dkk. (2025) memakai dataset, jumlah kelas, ukuran citra, atau skema split yang berbeda. Adeva dkk. memakai dataset yang sama, tetapi dengan split acak per file, sehingga salinan augmentasi dari foto yang sama bisa berada di data latih dan uji.

## 7. Kesimpulan singkat

1. **GLCM 4 arah (S2) adalah fitur terbaik** dalam eksperimen ini: F1 CV 0,980 hanya dengan 12 fitur, dan unggul dari GLCM 0° saja (H3 didukung untuk rata-rata F1).
2. **Menggabungkan GLCM dengan LBP tidak meningkatkan hasil** (H1 tidak didukung), dan **kernel non-linear hanya membantu pada fitur yang lemah** (H2 sebagian).
3. **Augmentasi bawaan dataset membantu LBP, tetapi hampir tidak berguna bagi GLCM 4 arah**, karena rotasi dan pencerminan tidak mengubah fiturnya.
4. **Model tidak underfitting dan hanya overfitting ringan, dan pipeline-nya tidak bocor.** Namun skor yang sangat tinggi harus dibaca bersama batasan kain yang sama: bila foto dipisah per blok nomor foto, F1 turun ke sekitar 0,76–0,82. Untuk mengukur generalisasi ke kain baru, dibutuhkan label ID kain atau data dari kain lain.
