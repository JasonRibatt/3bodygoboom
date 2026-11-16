"""Run 2: train multinomial logistic regression + tuned MLP on each feature set (grid data),
evaluate on val, held-out grid blocks, and the off-grid standard + ensemble test sets.
Usage: python train.py [feature_set ...]   (default: all four)"""
import itertools
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from common import (CLASSES, CLASS_TO_ID, OUT, SEED, load_grid, load_test, make_splits, save_splits,
                    encode, block_of)
from features import FEATURE_SETS

torch.set_num_threads(max(1, os.cpu_count() or 1))


class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout, n_out=3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xva, yva, hidden, dropout, wd, lr=1e-3, epochs=(5 if os.environ.get('QUICK') else 400), patience=40, bs=256, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva)
    best, best_state, bad, best_ep = -1, None, 0, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); lossf(m(Xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lv = m(Xv); vloss = lossf(lv, yv).item()
            vf1 = f1_score(yva, lv.argmax(1).numpy(), average="macro")
        sched.step(vloss)
        if vf1 > best + 1e-4:
            best, best_state, bad, best_ep = vf1, {k: v.clone() for k, v in m.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state); m.eval()
    return m, best, best_ep


def mlp_proba(m, X):
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(X, dtype=torch.float32)), 1).numpy()


MLP_GRID = {"hidden": [(64, 64), (128, 128), (256, 128, 64)], "dropout": [0.1, 0.3], "wd": [1e-4, 1e-2]}
if os.environ.get("QUICK"):   # smoke test only
    MLP_GRID = {"hidden": [(32,)], "dropout": [0.1], "wd": [1e-4]}
LR_GRID = {"C": [0.01, 0.1, 1, 10, 100], "class_weight": [None, "balanced"]}


def metrics(y, p):
    yhat = p.argmax(1)
    return {"n": int(len(y)),
            "accuracy": float(accuracy_score(y, yhat)),
            "macro_f1": float(f1_score(y, yhat, average="macro", labels=[0, 1, 2], zero_division=0)),
            "confusion": confusion_matrix(y, yhat, labels=[0, 1, 2]).tolist(),
            "per_class_f1": dict(zip(CLASSES, f1_score(y, yhat, average=None, labels=[0, 1, 2],
                                                       zero_division=0).round(4).tolist()))}


def run_feature_set(name, grid, gtraj, std, ens, ttraj, splits, out):
    d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    Xg, cols, vg = FEATURE_SETS[name](grid, gtraj)
    Xt, _, vt = FEATURE_SETS[name](std, ttraj)

    sp = {k: v[vg[v]] for k, v in splits.items()}
    test_resolved = std.label.isin(CLASSES).values
    t_idx = np.where(vt & test_resolved)[0]                    # standard test rows usable by this feature set
    e_idx = np.where(vt & ens.label.isin(CLASSES).values)[0]   # ensemble rows (majority label resolved)
    np.savez(os.path.join(d, "split_indices_used.npz"), **sp, test_standard=t_idx, test_ensemble=e_idx)

    scaler = StandardScaler().fit(Xg[sp["train"]])            # train rows only
    joblib.dump({"scaler": scaler, "feature_names": cols}, os.path.join(d, "scaler.joblib"))
    X = {k: scaler.transform(Xg[v]) for k, v in sp.items()}
    X["test_standard"] = scaler.transform(Xt[t_idx]); X["test_ensemble"] = scaler.transform(Xt[e_idx])
    X["test_all"] = scaler.transform(Xt[np.where(vt)[0]])
    y = {k: encode(grid.label.values[v]) for k, v in sp.items() if k != "grid_unresolved"}
    y["test_standard"] = encode(std.label.values[t_idx]); y["test_ensemble"] = encode(ens.label.values[e_idx])

    dropped = {"test_standard_nonfinite_by_label": std.label[~vt].value_counts().to_dict(),
               "grid_nonfinite_by_label": grid.label[~vg].value_counts().to_dict()}
    report = {"feature_set": name, "n_features": len(cols), "features": cols if len(cols) <= 16 else cols[:16] + ["..."],
              "sizes": {k: int(len(v)) for k, v in X.items()}, "rows_dropped_nonfinite": dropped, "models": {}}

    def finalize(mname, hp, vf1, P):
        res = {"hyperparams": hp, "val_macro_f1_at_selection": round(float(vf1), 4)}
        for k in ["train", "val", "heldout_block", "test_standard", "test_ensemble"]:
            res[k] = metrics(y[k], P[k])
            np.save(os.path.join(d, f"{mname}_confusion_{k}.npy"), np.array(res[k]["confusion"]))
        for bn, msk in block_of(grid, sp["heldout_block"]).items():
            res[bn] = metrics(y["heldout_block"][msk], P["heldout_block"][msk])
        # softmax on every test row this feature set can score (ensemble/MA analysis joins on run_id)
        all_idx = np.where(vt)[0]
        t = pd.DataFrame(P["test_all"], columns=[f"p_{c}" for c in CLASSES])
        t.insert(0, "run_id", std.run_id.values[all_idx])
        t["pred"] = [CLASSES[i] for i in P["test_all"].argmax(1)]
        t["standard_label"] = std.label.values[all_idx]; t["majority_label"] = ens.label.values[all_idx]
        t.to_csv(os.path.join(d, f"{mname}_softmax_test.csv"), index=False)
        # held-out block + grid Unresolved softmax
        for k in ["heldout_block", "grid_unresolved"]:
            idx = sp[k]
            if len(idx) == 0:
                continue
            t = pd.DataFrame(P[k], columns=[f"p_{c}" for c in CLASSES])
            t.insert(0, "run_id", grid.run_id.values[idx])
            t.insert(1, "a_out", grid.a_out.values[idx]); t.insert(2, "e_out", grid.e_out.values[idx])
            t["pred"] = [CLASSES[i] for i in P[k].argmax(1)]; t["label"] = grid.label.values[idx]
            t.to_csv(os.path.join(d, f"{mname}_softmax_{k}.csv"), index=False)
        return res

    # ---- logistic regression ----
    best = None
    for C, cw in itertools.product(LR_GRID["C"], LR_GRID["class_weight"]):
        lr = LogisticRegression(C=C, class_weight=cw, max_iter=5000, random_state=SEED).fit(X["train"], y["train"])
        f = f1_score(y["val"], lr.predict(X["val"]), average="macro")
        if best is None or f > best[0]:
            best = (f, lr, {"C": C, "class_weight": cw})
    joblib.dump(best[1], os.path.join(d, "logreg.joblib"))
    report["models"]["logreg"] = finalize("logreg", best[2], best[0], {k: best[1].predict_proba(v) for k, v in X.items()})

    # ---- MLP ----
    best, trials = None, []
    for hidden, dr, wd in itertools.product(MLP_GRID["hidden"], MLP_GRID["dropout"], MLP_GRID["wd"]):
        t0 = time.time()
        m, vf1, ep = train_mlp(X["train"], y["train"], X["val"], y["val"], hidden, dr, wd)
        trials.append({"hidden": hidden, "dropout": dr, "weight_decay": wd, "val_macro_f1": round(vf1, 4),
                       "best_epoch": ep, "sec": round(time.time() - t0, 1)})
        if best is None or vf1 > best[0]:
            best = (vf1, m, {"hidden": list(hidden), "dropout": dr, "weight_decay": wd, "lr": 1e-3, "best_epoch": ep})
    m = best[1]
    torch.save({"state_dict": m.state_dict(), "d_in": X["train"].shape[1], **best[2]}, os.path.join(d, "mlp.pt"))
    report["models"]["mlp"] = finalize("mlp", best[2], best[0], {k: mlp_proba(m, v) for k, v in X.items()})
    report["models"]["mlp"]["tuning_trials"] = trials

    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    grid, gtraj = load_grid()
    std, ens, ttraj = load_test()
    splits = make_splits(grid)
    save_splits(splits, os.path.join(OUT, "split_indices.npz"))
    for name in (sys.argv[1:] or list(FEATURE_SETS)):
        t0 = time.time()
        r = run_feature_set(name, grid, gtraj, std, ens, ttraj, splits, OUT)
        for mn, mr in r["models"].items():
            print(f"{name:18s} {mn:6s} test_std acc={mr['test_standard']['accuracy']:.3f} F1={mr['test_standard']['macro_f1']:.3f} | "
                  f"test_ens(maj) acc={mr['test_ensemble']['accuracy']:.3f} | block acc={mr['heldout_block']['accuracy']:.3f} | {mr['hyperparams']}")
        print(f"  ({time.time()-t0:.0f}s) sizes={r['sizes']}", flush=True)
