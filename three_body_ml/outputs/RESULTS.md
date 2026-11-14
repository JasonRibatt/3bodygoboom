| Feature set | Model | n test | Test acc | Test macro-F1 | Held-out E/S acc | Held-out E/C acc |
|---|---|---|---|---|---|---|
| fs1_ic | logreg | 1776 | 0.828 | 0.833 | 0.979 | 0.572 |
| fs1_ic | mlp | 1776 | 0.856 | 0.848 | 0.997 | 0.572 |
| fs2_ic_conserved | logreg | 1776 | 0.840 | 0.841 | 0.997 | 0.593 |
| fs2_ic_conserved | mlp | 1776 | 0.857 | 0.852 | 1.000 | 0.583 |
| fs3_early_traj | logreg | 1599 | 0.732 | 0.704 | 0.776 | 0.632 |
| fs3_early_traj | mlp | 1599 | 0.866 | 0.802 | 0.924 | 0.681 |
| fs4_ic_late_state | logreg | 1544 | 0.811 | 0.764 | 0.979 | 0.503 |
| fs4_ic_late_state | mlp | 1544 | 0.903 | 0.818 | 0.959 | 0.730 |

IC-only models re-scored on the trajectory sets' surviving subsets (test acc / macro-F1):

- fs3_early_traj test: fs1_ic logreg 0.809/0.768; fs1_ic mlp 0.844/0.772; fs2_ic_conserved logreg 0.823/0.774; fs2_ic_conserved mlp 0.845/0.782
- fs3_early_traj heldout_block: fs1_ic logreg 0.696/0.727; fs1_ic mlp 0.722/0.718; fs2_ic_conserved logreg 0.716/0.747; fs2_ic_conserved mlp 0.726/0.728
- fs4_ic_late_state test: fs1_ic logreg 0.804/0.731; fs1_ic mlp 0.843/0.726; fs2_ic_conserved logreg 0.818/0.734; fs2_ic_conserved mlp 0.842/0.737
- fs4_ic_late_state heldout_block: fs1_ic logreg 0.689/0.709; fs1_ic mlp 0.722/0.697; fs2_ic_conserved logreg 0.707/0.727; fs2_ic_conserved mlp 0.721/0.702
