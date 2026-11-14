"""Collect results into outputs/summary.json + outputs/RESULTS.md, including a fair comparison:
IC-only models re-scored on the same 'survived to t' subsets used by the trajectory feature sets."""
import json
import os
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from common import OUT, CLASSES, load_meta, labels
from features import FEATURE_SETS
from train import MLP

df = load_meta()
FS = [f for f in FEATURE_SETS if os.path.exists(os.path.join(OUT, f, "report.json"))]
R = {f: json.load(open(os.path.join(OUT, f, "report.json"))) for f in FS}


def load_models(fs):
    d = os.path.join(OUT, fs)
    sc = joblib.load(os.path.join(d, "scaler.joblib"))["scaler"]
    lr = joblib.load(os.path.join(d, "logreg.joblib"))
    ck = torch.load(os.path.join(d, "mlp.pt"))
    m = MLP(ck["d_in"], ck["hidden"], ck["dropout"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return sc, lr, m


def score(fs, idx):
    X, _, _ = FEATURE_SETS[fs](df)
    sc, lr, m = load_models(fs)
    Xs = sc.transform(X[idx]); y = labels(df, idx)
    with torch.no_grad():
        pm = m(torch.tensor(Xs, dtype=torch.float32)).argmax(1).numpy()
    pl = lr.predict(Xs)
    return {k: {"acc": round(accuracy_score(y, p), 4), "f1": round(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0), 4)}
            for k, p in [("logreg", pl), ("mlp", pm)]}


rows, fair = [], {}
for fs in FS:
    for mn, mr in R[fs]["models"].items():
        rows.append({"feature_set": fs, "model": mn, "n_test": mr["test"]["n"],
                     "test_acc": round(mr["test"]["accuracy"], 4), "test_macro_f1": round(mr["test"]["macro_f1"], 4),
                     "block_ES_acc": round(mr["block_ES"]["accuracy"], 4), "block_EC_acc": round(mr["block_EC"]["accuracy"], 4),
                     "heldout_acc": round(mr["heldout_block"]["accuracy"], 4), "heldout_macro_f1": round(mr["heldout_block"]["macro_f1"], 4),
                     "hyperparams": mr["hyperparams"]})
    if fs in ("fs3_early_traj", "fs4_ic_late_state"):
        sp = np.load(os.path.join(OUT, fs, "split_indices_used.npz"))
        fair[fs] = {split: {ic: score(ic, sp[split]) for ic in ("fs1_ic", "fs2_ic_conserved") if ic in R}
                    for split in ("test", "heldout_block")}
json.dump({"rows": rows, "ic_models_on_traj_subsets": fair,
           "split_sizes": {f: R[f]["split_sizes"] for f in FS},
           "rows_dropped_nonfinite": {f: R[f]["rows_dropped_nonfinite"] for f in FS}},
          open(os.path.join(OUT, "summary.json"), "w"), indent=1)

L = ["| Feature set | Model | n test | Test acc | Test macro-F1 | Held-out E/S acc | Held-out E/C acc |", "|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['feature_set']} | {r['model']} | {r['n_test']} | {r['test_acc']:.3f} | {r['test_macro_f1']:.3f} | {r['block_ES_acc']:.3f} | {r['block_EC_acc']:.3f} |")
L += ["", "IC-only models re-scored on the trajectory sets' surviving subsets (test acc / macro-F1):", ""]
for fs, d in fair.items():
    for split, v in d.items():
        L.append(f"- {fs} {split}: " + "; ".join(f"{ic} {m} {s['acc']:.3f}/{s['f1']:.3f}" for ic, mm in v.items() for m, s in mm.items()))
open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
