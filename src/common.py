"""Shared paths and helpers."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"            # images/, labels/, train.txt, val.txt, test.txt, split.csv
SESSIONS = ROOT / "data" / "sessions.csv"
MODELS = ROOT / "results" / "models"
RESULTS = ROOT / "results"
SPLITS = ("train", "val", "test")


def device():
    import torch
    if torch.cuda.is_available():
        return 0
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_split():
    """sessions.csv joined with split.csv: one row per photo, with split and condition columns."""
    return pd.read_csv(SESSIONS).merge(pd.read_csv(REAL / "split.csv"), on="photo")


def data_yaml():
    """Write the Ultralytics dataset file for our own photos and return its path."""
    p = REAL / "real.yaml"
    p.write_text(f"path: {REAL}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: pill\n")
    return p
