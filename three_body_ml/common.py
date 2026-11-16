"""Shared config, data loading and locked splits (run 2: Survived labels, off-grid test sets)."""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data", "run2"))
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "outputs", "run2"))

GRID_META = os.path.join(DATA, "pilot_3body_metadata.csv")
GRID_TRAJ = os.path.join(DATA, "pilot_3body_trajectories.npz")
TEST_STD = os.path.join(DATA, "test_standard_metadata.csv")
TEST_ENS = os.path.join(DATA, "test_ensemble_metadata.csv")
TEST_TRAJ = os.path.join(DATA, "test_trajectories.npz")

SEED = 42
CLASSES = ["Ejection", "Survived", "Collision"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Never used as model inputs: outcome-derived or computed after t=0.
LEAKY_COLS = {"rel_dE", "traj_complete", "ma_ratio_final", "ma_thresh_final", "exchange", "label",
              "standard_label", "agreement_fraction", "votes_Survived", "votes_Ejection",
              "votes_Collision", "votes_Unresolved"}
META_ONLY = {"init_overlapping", "r_peri_init", "ma_ratio_init", "ma_thresh_init"}  # analysis slices / baseline only

# Held-out contiguous (a_out, e_out) blocks of the GRID, removed before the train/val split (inclusive bounds).
# Chosen from the run-2 outcome map so each straddles one decision boundary.
HELDOUT_BLOCKS = {
    "block_ES": {"a_out": (1.60, 2.10), "e_out": (0.20, 0.40)},   # Ejection <-> Survived (496 runs: 58% E / 42% S)
    "block_EC": {"a_out": (0.95, 1.30), "e_out": (0.50, 0.75)},   # Ejection <-> Collision (450 runs: 48% C / 47% E / 5% S)
}
VAL_FRACTION = 0.15   # grid (minus blocks) -> 85% train / 15% val; the off-grid standard set is the test set


def _load_traj(path):
    # only the float 'trajectories' array is read; the pickled label arrays are never loaded
    with np.load(path, allow_pickle=False) as z:
        return z["trajectories"]


def load_grid():
    df = pd.read_csv(GRID_META).sort_values("run_id").reset_index(drop=True)
    assert (df.run_id.values == np.arange(len(df))).all(), "grid run_id must be 0..N-1"
    traj = _load_traj(GRID_TRAJ)
    assert len(traj) == len(df), f"grid trajectories {len(traj)} != metadata rows {len(df)}"
    return df, traj


def load_test():
    std = pd.read_csv(TEST_STD).sort_values("run_id").reset_index(drop=True)
    ens = pd.read_csv(TEST_ENS).sort_values("run_id").reset_index(drop=True)
    traj = _load_traj(TEST_TRAJ)
    assert len(std) == len(ens) == len(traj)
    for c in ["run_id", "a_out", "e_out", "phase_out"]:
        assert np.allclose(std[c].values, ens[c].values), f"standard/ensemble rows misaligned on {c}"
    assert (std.label.values == ens.standard_label.values).all(), "standard labels differ between files"
    return std, ens, traj


def heldout_mask(df):
    m = np.zeros(len(df), bool)
    for b in HELDOUT_BLOCKS.values():
        (a0, a1), (e0, e1) = b["a_out"], b["e_out"]
        m |= df.a_out.between(a0, a1).values & df.e_out.between(e0, e1).values
    return m


def block_of(df, idx):
    out = {}
    for name, b in HELDOUT_BLOCKS.items():
        (a0, a1), (e0, e1) = b["a_out"], b["e_out"]
        a, e = df.a_out.values[idx], df.e_out.values[idx]
        out[name] = (a >= a0) & (a <= a1) & (e >= e0) & (e <= e1)
    return out


def make_splits(df):
    """Locked grid splits as arrays of run_id. Deterministic given SEED."""
    resolved = df.label.isin(CLASSES).values
    block = heldout_mask(df)
    pool = np.where(resolved & ~block)[0]
    tr, va = train_test_split(pool, test_size=VAL_FRACTION, stratify=df.label.values[pool], random_state=SEED)
    return {"train": np.sort(tr), "val": np.sort(va),
            "heldout_block": np.where(resolved & block)[0],
            "grid_unresolved": np.where(df.label.values == "Unresolved")[0]}


def save_splits(splits, path):
    np.savez(path, **splits)
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump({k: v.tolist() for k, v in splits.items()}, f)


def encode(labels):
    return np.array([CLASS_TO_ID[l] for l in labels])
