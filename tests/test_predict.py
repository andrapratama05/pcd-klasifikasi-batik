"""predict_image must reproduce the notebook 04 predictions of the saved best model.

Needs the dataset (data/raw/), models/best_model.*, and results/test_predictions.csv;
skipped when any of them is missing (e.g. on a fresh clone before running notebooks).
"""

import json

import pandas as pd
import pytest

from src import config

MODEL = config.MODELS_DIR / "best_model.joblib"
META = config.MODELS_DIR / "best_model.json"
PREDS = config.RESULTS_DIR / "test_predictions.csv"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and META.exists() and PREDS.exists() and config.RAW_DIR.is_dir()),
    reason="butuh data/raw, models/best_model.* dan results/test_predictions.csv (jalankan notebook 04)",
)


@pytest.fixture(scope="module")
def reference():
    meta = json.loads(META.read_text(encoding="utf-8"))
    df = pd.read_csv(PREDS)
    return df, f"pred_{meta['scenario']}_{meta['kernel']}"


def test_predict_image_matches_notebook_04(reference):
    from src.predict import predict_image

    df, col = reference
    got = [predict_image(config.RAW_DIR / fn)["label"] for fn in df["filename"]]
    mismatches = [(fn, exp, g) for fn, exp, g in zip(df["filename"], df[col], got) if exp != g]
    assert not mismatches, f"{len(mismatches)} beda, contoh: {mismatches[:3]}"
    assert len(got) == 115


def test_top3_is_ranked_and_starts_with_label(reference):
    from src.predict import predict_image

    df, _ = reference
    res = predict_image(config.RAW_DIR / df["filename"].iloc[0])
    scores = [s for _, s in res["top"]]
    assert len(res["top"]) == 3
    assert scores == sorted(scores, reverse=True)
    assert res["top"][0][0] == res["label"]
    assert len({lbl for lbl, _ in res["top"]}) == 3
