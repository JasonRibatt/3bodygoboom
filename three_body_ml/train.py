"""Train multinomial logistic regression + tuned MLP on each feature set.
Usage: python train.py [feature_set ...]   (default: all available)"""
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

from common import CLASSES, OUT, SEED, HELDOUT_BLOCKS, load_meta, make_splits, save_splits, labels
from features import FEATURE_SETS, NEEDS_TRAJ
from common import TRAJ_PATH

torch.set_num_threads(max(1, os.cpu_count() or 1))
EVAL_SPLITS = ["train", "val", "test", "heldout_block"]


# ---------------- models ----------------
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


def train_mlp(Xtr, ytr, Xva, yva, hidden, dropout, wd, lr=1e-3, epochs=400, patience=40, bs=256, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = MLP(Xtr.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva)
    best, best_state, bad = -1, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lossf(m(Xt[b]), yt[b]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            lv = m(Xv)
            vloss = lossf(lv, yv).item()
            vf1 = f1_score(yva, lv.argmax(1).numpy(), average="macro")
        sched.step(vloss)
        if vf1 > best + 1e-4:
            best, best_state, bad, best_ep = vf1, {k: v.clone() for k, v in m.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, best, best_ep


def mlp_proba(m, X):
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(X, dtype=torch.float32)), 1).numpy()


MLP_GRID = {
    "hidden": [(64, 64), (128, 128), (256, 128, 64)],
    "dropout": [0.1, 0.3],
    "wd": [1e-4, 1e-2],
}
LR_GRID = {"C": [0.01, 0.1, 1, 10, 100], "class_weight": [None, "balanced"]}


# ---------------- helpers ----------------
def metrics(y, p):
    yhat = p.argmax(1)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, yhat)),
        "macro_f1": float(f1_score(y, yhat, average="macro", labels=[0, 1, 2], zero_division=0)),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1, 2]).tolist(),
        "per_class_f1": dict(zip(CLASSES, f1_score(y, yhat, average=None, labels=[0, 1, 2], zero_division=0).round(4).tolist())),
    }


def run_feature_set(name, df, splits, out):
    X_all, cols, valid = FEATURE_SETS[name](df)
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    sp = {k: v[valid[v]] for k, v in splits.items()}
    dropped = {k: {c: int(((df.label.values[v] == c) & ~valid[v]).sum()) for c in CLASSES + ["Unresolved"]}
               for k, v in splits.items()}
    np.savez(os.path.join(d, "split_indices_used.npz"), **sp)

    scaler = StandardScaler().fit(X_all[sp["train"]])  # train only
    joblib.dump({"scaler": scaler, "feature_names": cols}, os.path.join(d, "scaler.joblib"))
    Xs = {k: scaler.transform(X_all[v]) for k, v in sp.items()}
    ys = {k: labels(df, v) for k, v in sp.items() if k != "unresolved"}
    block_of = {}
    for bn, b in HELDOUT_BLOCKS.items():
        idx = sp["heldout_block"]
        block_of[bn] = (df.a_out.values[idx] >= b["a_out"][0]) & (df.a_out.values[idx] <= b["a_out"][1]) & \
                       (df.e_out.values[idx] >= b["e_out"][0]) & (df.e_out.values[idx] <= b["e_out"][1])

    report = {"feature_set": name, "n_features": len(cols), "features": cols if len(cols) <= 12 else cols[:12] + ["..."],
              "split_sizes": {k: int(len(v)) for k, v in sp.items()}, "rows_dropped_nonfinite": dropped,
              "majority_baseline_test": float(np.mean(ys["test"] == np.bincount(ys["train"]).argmax())),
              "models": {}}

    # ---- logistic regression ----
    best = None
    for C, cw in itertools.product(LR_GRID["C"], LR_GRID["class_weight"]):
        lr = LogisticRegression(C=C, class_weight=cw, max_iter=5000, random_state=SEED).fit(Xs["train"], ys["train"])
        f = f1_score(ys["val"], lr.predict(Xs["val"]), average="macro")
        if best is None or f > best[0]:
            best = (f, lr, {"C": C, "class_weight": cw})
    lr_model = best[1]
    joblib.dump(lr_model, os.path.join(d, "logreg.joblib"))
    probs = {k: lr_model.predict_proba(Xs[k]) for k in Xs}
    report["models"]["logreg"] = finalize("logreg", best[2], best[0], probs, ys, sp, block_of, df, d)

    # ---- MLP ----
    best, trials = None, []
    for hidden, dr, wd in itertools.product(MLP_GRID["hidden"], MLP_GRID["dropout"], MLP_GRID["wd"]):
        t0 = time.time()
        m, vf1, ep = train_mlp(Xs["train"], ys["train"], Xs["val"], ys["val"], hidden, dr, wd)
        trials.append({"hidden": hidden, "dropout": dr, "weight_decay": wd, "val_macro_f1": round(vf1, 4),
                       "best_epoch": ep, "sec": round(time.time() - t0, 1)})
        if best is None or vf1 > best[0]:
            best = (vf1, m, {"hidden": list(hidden), "dropout": dr, "weight_decay": wd, "lr": 1e-3, "best_epoch": ep})
    m = best[1]
    torch.save({"state_dict": m.state_dict(), "d_in": Xs["train"].shape[1], **best[2]}, os.path.join(d, "mlp.pt"))
    probs = {k: mlp_proba(m, Xs[k]) for k in Xs}
    report["models"]["mlp"] = finalize("mlp", best[2], best[0], probs, ys, sp, block_of, df, d)
    report["models"]["mlp"]["tuning_trials"] = trials

    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def finalize(mname, hp, val_f1, probs, ys, sp, block_of, df, d):
    res = {"hyperparams": hp, "val_macro_f1_at_selection": round(float(val_f1), 4)}
    for k in EVAL_SPLITS:
        res[k] = metrics(ys[k], probs[k])
        np.save(os.path.join(d, f"{mname}_confusion_{k}.npy"), np.array(res[k]["confusion"]))
    for bn, msk in block_of.items():
        res[bn] = metrics(ys["heldout_block"][msk], probs["heldout_block"][msk])
    # softmax outputs: unresolved (the requested deliverable) + test/heldout for calibration checks
    for k in ["unresolved", "test", "heldout_block"]:
        idx = sp[k]
        t = pd.DataFrame(probs[k], columns=[f"p_{c.replace(' ', '_')}" for c in CLASSES])
        t.insert(0, "run_id", df.run_id.values[idx])
        t.insert(1, "a_out", df.a_out.values[idx]); t.insert(2, "e_out", df.e_out.values[idx])
        t.insert(3, "phase_out", df.phase_out.values[idx])
        t["pred"] = [CLASSES[i] for i in probs[k].argmax(1)]
        if k != "unresolved":
            t["true"] = df.label.values[idx]
        t.to_csv(os.path.join(d, f"{mname}_softmax_{k}.csv"), index=False)
    pu = probs["unresolved"]
    res["unresolved_summary"] = {"n": int(len(pu)),
                                 "pred_counts": dict(zip(CLASSES, np.bincount(pu.argmax(1), minlength=3).tolist())),
                                 "mean_probs": dict(zip(CLASSES, pu.mean(0).round(4).tolist())),
                                 "mean_max_prob": round(float(pu.max(1).mean()), 4)}
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load_meta()
    splits = make_splits(df)
    save_splits(splits, os.path.join(OUT, "split_indices.npz"))
    want = sys.argv[1:] or [k for k in FEATURE_SETS if k not in NEEDS_TRAJ or os.path.exists(TRAJ_PATH)]
    for name in want:
        t0 = time.time()
        r = run_feature_set(name, df, splits, OUT)
        for mn, mr in r["models"].items():
            print(f"{name:20s} {mn:7s} test acc={mr['test']['accuracy']:.3f} F1={mr['test']['macro_f1']:.3f} | "
                  f"block acc={mr['heldout_block']['accuracy']:.3f} F1={mr['heldout_block']['macro_f1']:.3f} | {mr['hyperparams']}")
        print(f"  ({time.time()-t0:.0f}s) sizes={r['split_sizes']}", flush=True)
