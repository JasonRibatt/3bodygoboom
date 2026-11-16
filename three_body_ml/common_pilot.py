"""Shared config, data loading, and locked splits for the three-body outcome classifier."""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
META_PATH = os.environ.get("META_PATH", os.path.join(DATA, "pilot_3body_metadata.xlsx"))
TRAJ_PATH = os.environ.get("TRAJ_PATH", os.path.join(DATA, "pilot_3body_trajectories.npz"))

SEED = 42
CLASSES = ["Ejection", "Stable Hierarchical", "Collision"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Columns that are computed from the *outcome* of the integration -> never features.
LEAKY_COLS = {"rel_dE", "traj_complete"}

# Held-out contiguous regions of (a_out, e_out) space (inclusive bounds).
# Outcomes are nearly deterministic in (a_out, e_out): Stable occupies wide/low-e orbits,
# Collision tight/high-e orbits, Ejection everything between. A single compact block
# cannot contain all three classes, so we hold out one block on each decision boundary.
HELDOUT_BLOCKS = {
    "block_ES": {"a_out": (1.45, 1.95), "e_out": (0.00, 0.20)},  # Ejection <-> Stable boundary
    "block_EC": {"a_out": (0.95, 1.30), "e_out": (0.50, 0.75)},  # Ejection <-> Collision boundary
}


def load_meta():
    if META_PATH.endswith(".csv"):
        df = pd.read_csv(META_PATH)
    else:
        df = pd.read_excel(META_PATH)
    df = df.sort_values("run_id").reset_index(drop=True)
    assert (df.run_id.values == np.arange(len(df))).all(), "run_id must be 0..N-1"
    return df


def heldout_mask(df):
    m = np.zeros(len(df), bool)
    for b in HELDOUT_BLOCKS.values():
        (a0, a1), (e0, e1) = b["a_out"], b["e_out"]
        m |= df.a_out.between(a0, a1).values & df.e_out.between(e0, e1).values
    return m


def make_splits(df):
    """Locked splits, returned as arrays of run_id. Deterministic given SEED."""
    resolved = df.label.isin(CLASSES).values
    block = heldout_mask(df)
    pool = np.where(resolved & ~block)[0]
    y_pool = df.label.values[pool]
    tr, tmp = train_test_split(pool, test_size=0.30, stratify=y_pool, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=df.label.values[tmp], random_state=SEED)
    return {
        "train": np.sort(tr),
        "val": np.sort(va),
        "test": np.sort(te),
        "heldout_block": np.where(resolved & block)[0],
        "unresolved": np.where(df.label.values == "Unresolved")[0],
    }


def save_splits(splits, path):
    np.savez(path, **splits)
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump({k: v.tolist() for k, v in splits.items()}, f)


def labels(df, idx):
    return np.array([CLASS_TO_ID[l] for l in df.label.values[idx]])
