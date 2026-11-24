import pandas as pd
import pytest

from src import config
from src.data import make_split, parse_filename


@pytest.mark.parametrize(
    "name, motif, idx, aug",
    [
        ("Angguran (24).jpg", "Angguran", 24, "original"),
        ("Angguran (24)_RandomRotate90.jpg", "Angguran", 24, "RandomRotate90"),
        ("Puger Telu Bal (3)_FDA.jpg", "Puger Telu Bal", 3, "FDA"),
        ("Puger Telu Bal (3).JPG", "Puger Telu Bal", 3, "original"),
    ],
)
def test_parse_filename(name, motif, idx, aug):
    p = parse_filename(name)
    assert p["motif"] == motif
    assert p["idx"] == idx
    assert p["origin_id"] == f"{motif}_{idx}"
    assert p["aug_type"] == aug


@pytest.mark.parametrize("name", ["Angguran 24.jpg", "Angguran (24).png", "notes.txt"])
def test_parse_filename_rejects(name):
    assert parse_filename(name) is None


@pytest.fixture
def synthetic_index():
    """Full synthetic dataset: 11 classes x 52 origins x 9 files."""
    rows = []
    for cls in config.CLASSES:
        for i in range(1, 53):
            for aug in ["original", *config.AUG_TYPES]:
                suffix = "" if aug == "original" else f"_{aug}"
                name = f"{cls} ({i}){suffix}.jpg"
                rows.append({"filename": f"{cls}/{name}", "label": cls, **parse_filename(name), "matched": True})
    return pd.DataFrame(rows)


def test_split_no_origin_in_both(synthetic_index):
    split = make_split(synthetic_index)
    train = set(split.loc[split["split"] == "train", "origin_id"])
    test = set(split.loc[split["split"] == "test", "origin_id"])
    assert train and test
    assert train.isdisjoint(test)


def test_split_test_is_originals_only(synthetic_index):
    split = make_split(synthetic_index)
    assert set(split.loc[split["split"] == "test", "aug_type"]) == {"original"}


def test_split_stratified_and_deterministic(synthetic_index):
    a, b = make_split(synthetic_index), make_split(synthetic_index)
    pd.testing.assert_frame_equal(a, b)
    test_per_class = a[a["split"] == "test"]["label"].value_counts()
    assert test_per_class.min() >= 10 and test_per_class.max() <= 11
