# Run 2 results — Survived labels, off-grid standard + K=5 ensemble test sets

Data: Jonah's `data_run2` (grid 15,590 runs: 42.8% Ejection / 39.0% Survived / 18.2% Collision / 15 Unresolved;
off-grid test 2,000 runs with a K=5 ensemble at 1e-10 phase nudges).
Pipeline: `python train.py` → `python evaluate.py` → `python plots_run2.py` (data in `data/run2/`, outputs in `outputs/run2/`).

## Setup
- Train/val = grid minus held-out blocks, stratified 85/15 (12,434 / 2,195). Test = the 2,000 off-grid runs.
- Held-out grid blocks (946 runs), re-chosen for the run-2 labels:
  E/S a_out 1.60–2.10, e_out 0.20–0.40 (58% E / 42% S); E/C a_out 0.95–1.30, e_out 0.50–0.75 (48% C / 47% E).
- Never in X: rel_dE, traj_complete, ma_ratio_final, ma_thresh_final (blank for every non-Survived run), exchange,
  vote columns. init_overlapping, r_peri_init, ma_ratio_init, ma_thresh_init are used only for slicing / the MA baseline.
- Scalers fit on train only; model selection on val macro-F1. Same LogReg and MLP grids as the pilot.

## Accuracy (off-grid test)
| Feature set | Model | vs single-run label | macro-F1 | vs majority label | held-out E/S | held-out E/C |
|---|---|---|---|---|---|---|
| (1) IC | LogReg | 0.763 | 0.729 | 0.789 | 0.661 | 0.513 |
| (1) IC | MLP | 0.783 | 0.763 | 0.812 | 0.690 | 0.556 |
| (2) IC + E0, Lz0 | LogReg | 0.767 | 0.754 | 0.784 | 0.571 | 0.549 |
| (2) IC + E0, Lz0 | MLP | 0.782 | 0.761 | 0.816 | 0.679 | 0.556 |
| (3) traj t=1–10 | LogReg | 0.680 | 0.630 | 0.691 | 0.528 | 0.578 |
| (3) traj t=1–10 | MLP | 0.809 | 0.742 | 0.837 | 0.635 | 0.657 |
| (4) IC + state t=20 | LogReg | 0.747 | 0.694 | 0.772 | 0.649 | 0.475 |
| (4) IC + state t=20 | MLP | 0.838 | 0.768 | 0.863 | 0.659 | 0.649 |

Majority-class baseline 0.438. Sets 3–4 score fewer test rows (1,842 / 1,786) because runs that collide before
their snapshots are excluded. On those same rows the IC MLP scores 0.771 / 0.769 (macro-F1 0.696 / 0.651), so the
trajectory gain is +3.8 pts (t=1–10) and +6.9 pts (t=20).

## Chaos ceiling (ensemble set)
- Plug-in mean agreement 0.910 (Jonah's number; counts each run's own vote, so optimistic).
- Leave-one-out (predict each vote from the other four) 0.866 (pessimistic). 71% of runs are unanimous.
- IC MLP: expected accuracy on a random realization 0.773 — about 9–14 pts under the ceiling.
- The gap is not chaos at the 1e-10 scale: the IC MLP misses 16% of unanimous runs, 70% of those at e_out > 0.7
  and almost all Ejection↔Collision. Outcomes there are locally deterministic but change between neighbouring grid
  points (phase spacing 0.25 rad), so a model trained on the 25-per-cell grid cannot resolve them.
- Accuracy vs majority by agreement (IC MLP): 5/5 0.84, 4/5 0.87, 3/5 0.65.

## Calibration vs the votes
- Reliability is excellent: ECE vs vote share 0.007 (IC MLP), 0.016 (t=20 MLP) — when the model says 70%, ~70% of votes agree.
- But the model is less sharp than the ensemble: Brier vs votes 0.179 against 0.060 expected if its probabilities
  were the true outcome probabilities. It hedges on runs whose outcome is fixed at 1e-10 but varies below grid spacing.

## Mardling–Aarseth baseline (Survived vs not, single-run labels)
- MA at t=0 never calls a disrupted run stable (0 cases), but misses every metastable survivor:
  49% of grid survivors and 376 / 757 test survivors fail MA.
- Overall: MA 0.812 vs IC MLP 0.913 (LogReg 0.895, t=20 MLP 0.904).
- Near the MA boundary (ratio/threshold 0.8–1.25, n=300): MA 0.49 vs models 0.99–1.00.
- Metastable survivors (n=376): MA 0 by definition; IC LogReg 0.73, IC MLP 0.60, t=20 MLP 0.65.
- Metastable survivors are more chaotic (ensemble agreement 0.84 vs 0.93 for other runs).

## Compared with the pilot
Accuracy is lower than the pilot (IC MLP 0.856 → 0.783) because the pilot dropped the ~20% Unresolved runs, which
are exactly the hard metastable cases now labelled Survived. It is not a regression in the model.

## Figures (outputs/run2/figures)
outcome_map_run2.png, accuracy_vs_ceiling.png, accuracy_by_agreement.png, calibration_vs_votes.png,
ma_baseline.png, confusion_test_standard.png
