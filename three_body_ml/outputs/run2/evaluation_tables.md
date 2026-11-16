Chaos ceiling (K=5): plug-in 0.910, leave-one-out 0.866; unanimous 71.0%; agreement counts {'1/5': 0, '2/5': 15, '3/5': 294, '4/5': 270, '5/5': 1421}

| Model | acc vs standard | acc vs majority | expected acc on a realization | ceiling (plug-in / LOO, same rows) | Brier vs votes (floor) | ECE |
|---|---|---|---|---|---|---|
| fs1_ic/logreg | 0.763 | 0.789 | 0.757 | 0.910 / 0.866 | 0.203 (0.067) | 0.020 |
| fs1_ic/mlp | 0.783 | 0.812 | 0.773 | 0.910 / 0.866 | 0.179 (0.060) | 0.007 |
| fs2_ic_conserved/logreg | 0.767 | 0.784 | 0.758 | 0.910 / 0.866 | 0.211 (0.067) | 0.020 |
| fs2_ic_conserved/mlp | 0.782 | 0.816 | 0.775 | 0.910 / 0.866 | 0.178 (0.061) | 0.014 |
| fs3_early_traj/logreg * | 0.680 | 0.691 | 0.674 | 0.902 / 0.855 | 0.291 (0.080) | 0.041 |
| fs3_early_traj/mlp * | 0.809 | 0.837 | 0.797 | 0.902 / 0.855 | 0.155 (0.051) | 0.022 |
| fs4_ic_late_state/logreg * | 0.747 | 0.772 | 0.743 | 0.899 / 0.850 | 0.216 (0.068) | 0.038 |
| fs4_ic_late_state/mlp * | 0.838 | 0.863 | 0.822 | 0.899 / 0.850 | 0.117 (0.051) | 0.016 |

* trajectory feature sets: ensemble columns shown for reference only (perturbed siblings' trajectories are not saved).

| Model | all resolved | metastable (Survived, MA fails) | MA passes but disrupted | near MA boundary (ratio/thresh in 0.8-1.25) |
|---|---|---|---|---|
| fs1_ic/logreg | MA 0.812 vs model 0.895 (n=1996) | MA 0.000 vs model 0.726 (n=376) | n=0 | MA 0.487 vs model 0.990 (n=300) |
| fs1_ic/mlp | MA 0.812 vs model 0.913 (n=1996) | MA 0.000 vs model 0.598 (n=376) | n=0 | MA 0.487 vs model 0.997 (n=300) |
| fs2_ic_conserved/logreg | MA 0.812 vs model 0.898 (n=1996) | MA 0.000 vs model 0.769 (n=376) | n=0 | MA 0.487 vs model 0.990 (n=300) |
| fs2_ic_conserved/mlp | MA 0.812 vs model 0.914 (n=1996) | MA 0.000 vs model 0.628 (n=376) | n=0 | MA 0.487 vs model 0.990 (n=300) |
| fs3_early_traj/logreg | MA 0.796 vs model 0.848 (n=1842) | MA 0.000 vs model 0.670 (n=376) | n=0 | MA 0.487 vs model 0.960 (n=300) |
| fs3_early_traj/mlp | MA 0.796 vs model 0.900 (n=1842) | MA 0.000 vs model 0.633 (n=376) | n=0 | MA 0.487 vs model 0.987 (n=300) |
| fs4_ic_late_state/logreg | MA 0.789 vs model 0.876 (n=1786) | MA 0.000 vs model 0.721 (n=376) | n=0 | MA 0.487 vs model 0.990 (n=300) |
| fs4_ic_late_state/mlp | MA 0.789 vs model 0.904 (n=1786) | MA 0.000 vs model 0.649 (n=376) | n=0 | MA 0.487 vs model 0.990 (n=300) |
