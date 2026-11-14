"""Feature-set builders. Every builder returns (X, feature_names, valid_mask) for ALL runs;
valid_mask marks runs whose features are finite (used to drop NaN-padded trajectories).
rel_dE and traj_complete are never read here."""
import os
import numpy as np
from common import LEAKY_COLS, TRAJ_PATH

IC_COLS = ["a_out", "e_out", "sin_phase", "cos_phase"]


def _ic(df):
    X = np.column_stack([df.a_out, df.e_out, np.sin(df.phase_out), np.cos(df.phase_out)])
    return X, list(IC_COLS)


def fs1_ic(df):
    X, names = _ic(df)
    return X, names, np.isfinite(X).all(1)


def fs2_ic_conserved(df):
    X, names = _ic(df)
    X = np.column_stack([X, df.E0, df.Lz0])
    return X, names + ["E0", "Lz0"], np.isfinite(X).all(1)


# ---------- trajectory feature sets ----------
_TRAJ = None


def load_traj():
    """Returns (traj[N, T, F], times[T] or None, feat_names). Adapt keys here if the .npz differs."""
    global _TRAJ
    if _TRAJ is None:
        z = np.load(TRAJ_PATH, allow_pickle=False)
        _TRAJ = adapt_npz(z)
    return _TRAJ


def adapt_npz(z):
    """pilot_3body_trajectories.npz: 'trajectories' float64 [N=15625, 20 snapshots, 12 state values],
    NaN-padded after a collision. Snapshots are labelled t=1..20 (t=k -> index k-1); snapshot 1 is
    NOT the initial state (its energy != E0). The pickled 'labels' array is never loaded."""
    traj = z["trajectories"]
    # pad a dummy t=0 slot so traj[:, t] means snapshot t (1-based)
    traj = np.concatenate([np.full_like(traj[:, :1], np.nan), traj], axis=1)
    names = [f"c{i}" for i in range(traj.shape[2])]
    return traj, None, names


def _snapshot_cols(names, t):
    return [f"{n}@t{t}" for n in names]


def fs3_early_traj(df, t_lo=1, t_hi=10):
    traj, _, names = load_traj()
    W = traj[:, t_lo:t_hi + 1, :]                     # snapshot indices t_lo..t_hi
    X = W.reshape(len(W), -1)
    cols = [c for t in range(t_lo, t_hi + 1) for c in _snapshot_cols(names, t)]
    return X, cols, np.isfinite(X).all(1)


def fs4_ic_late_state(df, t=20):
    traj, _, names = load_traj()
    Xi, ic_names = _ic(df)
    X = np.column_stack([Xi, traj[:, t, :]])
    return X, ic_names + _snapshot_cols(names, t), np.isfinite(X).all(1)


FEATURE_SETS = {
    "fs1_ic": fs1_ic,
    "fs2_ic_conserved": fs2_ic_conserved,
    "fs3_early_traj": fs3_early_traj,
    "fs4_ic_late_state": fs4_ic_late_state,
}
NEEDS_TRAJ = {"fs3_early_traj", "fs4_ic_late_state"}
assert not (set(IC_COLS) & LEAKY_COLS)
