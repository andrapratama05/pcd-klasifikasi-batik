"""Metrics and plotting helpers shared by notebooks 03-05."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src import config


def summarize(y_true, y_pred, classes=config.CLASSES) -> dict:
    """Accuracy, macro F1, and per-class precision/recall/F1 as a flat dict."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
    }
    for c, pi, ri, fi in zip(classes, p, r, f):
        out[f"precision_{c}"] = pi
        out[f"recall_{c}"] = ri
        out[f"f1_{c}"] = fi
    return out


def plot_confusion(y_true, y_pred, ax, title="", classes=config.CLASSES, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    ax.imshow(shown, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("prediksi")
    ax.set_ylabel("label sebenarnya")
    ax.set_title(title, fontsize=10)
    thresh = shown.max() / 2 if shown.max() else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if shown[i, j] > thresh else "black")
    return cm


def compare(mean_a, std_a, mean_b, std_b) -> str:
    """Rule from CLAUDE.md: a difference smaller than the std is 'tidak meyakinkan'.

    Uses the larger of the two fold stds. Returns 'didukung' if a > b by more than
    that std, 'tidak didukung' if b > a by more than that std, else 'tidak meyakinkan'.
    """
    diff, tol = mean_a - mean_b, max(std_a, std_b)
    if abs(diff) <= tol:
        return "tidak meyakinkan"
    return "didukung" if diff > 0 else "tidak didukung"
