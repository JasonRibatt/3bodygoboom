"""Feature-set builders. Each takes (metadata DataFrame, trajectory array [N, 20, 12]) and returns
(X, feature_names, valid_mask). valid_mask marks rows whose features are all finite, so NaN-padded
(post-collision) trajectories are dropped rather than imputed. Outcome-derived columns are never read."""
import numpy as np
from common import LEAKY_COLS, META_ONLY

IC_COLS = ["a_out", "e_out", "sin_phase", "cos_phase"]
# trajectory columns (data_generator.py): x0,x1,x2, y0,y1,y2, vx0,vx1,vx2, vy0,vy1,vy2 ; snapshot k is t = k (k = 1..20)
STATE_NAMES = ["x0", "x1", "x2", "y0", "y1", "y2", "vx0", "vx1", "vx2", "vy0", "vy1", "vy2"]


def _ic(df):
    X = np.column_stack([df.a_out, df.e_out, np.sin(df.phase_out), np.cos(df.phase_out)])
    return X, list(IC_COLS)


def _snap(traj, t):
    return traj[:, t - 1, :]                      # t = 1..20 -> index 0..19


def fs1_ic(df, traj):
    X, names = _ic(df)
    return X, names, np.isfinite(X).all(1)


def fs2_ic_conserved(df, traj):
    X, names = _ic(df)
    X = np.column_stack([X, df.E0, df.Lz0])
    return X, names + ["E0", "Lz0"], np.isfinite(X).all(1)


def fs3_early_traj(df, traj, t_lo=1, t_hi=10):
    X = np.concatenate([_snap(traj, t) for t in range(t_lo, t_hi + 1)], axis=1)
    names = [f"{n}@t{t}" for t in range(t_lo, t_hi + 1) for n in STATE_NAMES]
    return X, names, np.isfinite(X).all(1)


def fs4_ic_late_state(df, traj, t=20):
    Xi, ic_names = _ic(df)
    X = np.column_stack([Xi, _snap(traj, t)])
    return X, ic_names + [f"{n}@t{t}" for n in STATE_NAMES], np.isfinite(X).all(1)


FEATURE_SETS = {
    "fs1_ic": fs1_ic,
    "fs2_ic_conserved": fs2_ic_conserved,
    "fs3_early_traj": fs3_early_traj,
    "fs4_ic_late_state": fs4_ic_late_state,
}
IC_ONLY = {"fs1_ic", "fs2_ic_conserved"}
assert not (set(IC_COLS) & (LEAKY_COLS | META_ONLY))
