"""Central configuration: paths, seed, dataset source, and experiment parameters.

All paths are relative to the repo root so the project runs on any machine.
"""

from pathlib import Path

import numpy as np

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ZIP_DIR = DATA_DIR / "zip"
RAW_DIR = DATA_DIR / "raw" / "batik_banyumasan"
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
DEMO_DIR = DATA_DIR / "demo"  # put new photos here for notebook 06 / src.predict

# --- Reproducibility -----------------------------------------------------
SEED = 42

# --- Dataset source ------------------------------------------------------
# Batik Banyumasan (Syafi'i, Mendeley Data, v2, 2025), CC BY 4.0
# DOI: 10.17632/vsjrkhs9km.2
# Mendeley public API endpoint; it redirects (302) to a short-lived presigned
# S3 URL of the "Download All" ZIP. Verified with a ranged GET on 2026-11-24
# (application/zip, 1,063,487,295 bytes). A plain HEAD on the S3 URL returns
# 403 because the signature is only valid for GET.
DATASET_URL = "https://data.mendeley.com/public-api/zip/vsjrkhs9km/download/2"
DATASET_PAGE = "https://data.mendeley.com/datasets/vsjrkhs9km/2"
ZIP_NAME = "batik_banyumasan_v2.zip"  # readable local name for the Mendeley "Download All" ZIP

CLASSES = [
    "Angguran",
    "Ayam Puger",
    "Jahe Lumbon",
    "Jahe Puger",
    "Jahe Srimpang",
    "Lumbon",
    "Madu Bronto",
    "Pring Sedapur",
    "Puger Galar",
    "Puger Telu Bal",
    "Wit Lumbon",
]
AUG_TYPES = [
    "ChannelShuffle",
    "ElasticTransform",
    "FDA",
    "Flip",
    "RandomGridShuffle",
    "RandomRotate90",
    "Sharpen",
    "Transpose",
]
FILES_PER_ORIGIN = 1 + len(AUG_TYPES)  # 9
# "Angguran (24).jpg" / "Angguran (24)_RandomRotate90.jpg" (extension case-insensitive)
FILENAME_PATTERN = r"^(?P<motif>.+?) \((?P<idx>\d+)\)(?:_(?P<aug>\w+))?\.(?i:jpg)$"
EXPECTED_TOTAL = 5148
EXPECTED_ORIGINALS = 572

# --- Split ---------------------------------------------------------------
TEST_SIZE = 0.2
N_FOLDS = 5

# --- Preprocessing -------------------------------------------------------
IMG_SIZE = 256  # final (confirmed 2026-11-24); all source images are 512x512 RGB, rationale in note.md
CLAHE_CLIP_LIMIT = 0.01

# --- H3 monitored class -------------------------------------------------
# Most directional class, chosen visually in notebook 01 from training originals only,
# before any model results (confirmed 2026-11-24). Its recall is compared between S1 and S2.
H3_CLASS = "Puger Galar"

# --- GLCM ----------------------------------------------------------------
GLCM_LEVELS = 8
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_ANGLE_NAMES = ["a0", "a45", "a90", "a135"]
GLCM_PROPS = ["contrast", "correlation", "asm", "homogeneity"]

# --- LBP -----------------------------------------------------------------
LBP_P = 8
LBP_R = 1
LBP_METHOD = "nri_uniform"
LBP_BINS = 59

# --- SVM grid ------------------------------------------------------------
SVM_C = [0.1, 1, 10, 100]
SVM_GAMMA = [0.001, 0.01, 0.1, 1]
SVM_DEGREE = [2, 3]
PARAM_GRIDS = {
    "linear": {"svc__C": SVM_C},
    "rbf": {"svc__C": SVM_C, "svc__gamma": SVM_GAMMA},
    "poly": {"svc__C": SVM_C, "svc__gamma": SVM_GAMMA, "svc__degree": SVM_DEGREE},
}
SCORING = "f1_macro"
