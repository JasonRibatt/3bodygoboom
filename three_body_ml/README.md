# Three-body outcome classification

**Current: run 2** (Survived labels, off-grid standard + K=5 ensemble test sets). See `RESULTS_run2.md`.
Put Jonah's `data_run2` files in `data/run2/`, then: `python train.py` → `python evaluate.py` → `python plots_run2.py`.
Outputs go to `outputs/run2/`. The pilot results below are kept for reference (`outputs/` top level, `common_pilot.py`).

# Three-body outcome classification — pilot dataset

Run: `python train.py` (all four feature sets) → `python summary.py` → `python plots.py`.
Needs `data/pilot_3body_metadata.xlsx` (or set `META_PATH` to the CSV) and `data/pilot_3body_trajectories.npz`.

## Data handling
- 15,625 runs → drop 3,067 Unresolved → 12,558 labelled (53.2% Ejection / 24.3% Stable / 22.5% Collision).
- `rel_dE` and `traj_complete` are never read as features. Both are outcome-derived: `rel_dE` is NaN exactly for
  Collisions, and `traj_complete == 0` only occurs for Collisions.
- The `.npz` has `trajectories` [15625, 20, 12] (NaN-padded after collision) and a pickled `labels` array, which is
  never loaded (labels come from the metadata). Snapshots are indexed t = 1…20. Snapshot 1 is not the initial
  state. The 12 state columns are named `c0…c11` because their exact layout isn't documented.
- Trajectory sets keep only runs whose used snapshots are all finite ("survived to t"). Feature set 3 needs t ≤ 10 and drops
  1,205 early collisions. Feature set 4 needs t = 20, drops 1,617 collisions, and is equivalent to `traj_complete == 1`. This turns those two sets into conditional tasks
  ("given the system is intact at t…"), so `summary.py` also re-scores the IC models on those same subsets.

## Splits (locked, seed 42, `outputs/split_indices.npz/.json`, values = run_id)
- Two held-out contiguous (a_out, e_out) blocks, removed before splitting (722 labelled runs):
  - `block_ES`: a_out ∈ [1.45, 1.95], e_out ∈ [0, 0.20] (straddles the Ejection/Stable boundary)
  - `block_EC`: a_out ∈ [0.95, 1.30], e_out ∈ [0.50, 0.75] (straddles the Ejection/Collision boundary)
  - Reason: outcomes are almost deterministic in (a_out, e_out), so no single compact block contains all three classes.
    The suggested [1.8, 2.2] × [0.3, 0.5] block is ~99% Ejection.
- Remaining labelled runs: stratified 70 / 15 / 15 → 8,285 / 1,775 / 1,776.
- Scalers are fit on train only. Models are selected by validation macro-F1.

## Models
- Multinomial logistic regression: grid over C ∈ {0.01…100} × class_weight ∈ {None, balanced}.
- MLP (PyTorch): ReLU, AdamW, dropout, weight decay, early stopping on val macro-F1. Grid: hidden ∈ {64-64, 128-128, 256-128-64}
  × dropout ∈ {0.1, 0.3} × weight decay ∈ {1e-4, 1e-2}.

## Outputs per feature set (`outputs/<fs>/`)
`scaler.joblib`, `logreg.joblib`, `mlp.pt`, `split_indices_used.npz`, `{model}_confusion_{split}.npy`,
`{model}_softmax_unresolved.csv` (+ test / held-out), and `report.json` (metrics, per-class F1, tuning trials).
Overall: `summary.json`, `RESULTS.md`, `figures/`.
