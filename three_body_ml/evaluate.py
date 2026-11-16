"""Run 2 evaluation beyond plain accuracy:
  1. Mardling-Aarseth (MA) analytical baseline vs the models, as Survived-vs-not on the standard test set.
  2. Ensemble test set: chaos ceiling (plug-in and leave-one-out), accuracy by agreement level, and
     calibration of softmax probabilities against the K=5 vote fractions.
Reads the softmax CSVs written by train.py. Writes outputs/run2/evaluation.json and evaluation_tables.md."""
import json
import os
import numpy as np
import pandas as pd
from common import OUT, CLASSES, load_test
from features import FEATURE_SETS, IC_ONLY

OUTCOMES = CLASSES + ["Unresolved"]
std, ens, _ = load_test()
K = int(ens[[f"votes_{c}" for c in OUTCOMES]].sum(1).iloc[0])
V = ens[[f"votes_{c}" for c in OUTCOMES]].to_numpy(float)            # [N, 4] vote counts
assert (V.sum(1) == K).all()


# ---------------- chaos ceilings ----------------
def loo_ceiling(V):
    """Accuracy of predicting one held-out vote from the majority of the other K-1 (ties split evenly)."""
    acc = np.zeros(len(V))
    for c in range(V.shape[1]):
        n_c = V[:, c]
        rest = V.copy(); rest[:, c] -= 1
        top = rest.max(1, keepdims=True)
        win = (rest == top)
        hit = win[:, c] / win.sum(1)
        acc += np.where(n_c > 0, n_c * hit, 0)
    return acc / V.sum(1)


ceil = {"K": K,
        "plug_in_mean_agreement": float((V.max(1) / K).mean()),          # Jonah's 91.0% (optimistic)
        "leave_one_out": float(loo_ceiling(V).mean()),                    # pessimistic
        "share_unanimous": float((V.max(1) == K).mean()),
        "agreement_distribution": {f"{a}/{K}": int(((V.max(1)) == a).sum()) for a in range(1, K + 1)}}
ceil_resolved_mask = std.label.isin(CLASSES).values


def model_eval(fs, model):
    path = os.path.join(OUT, fs, f"{model}_softmax_test.csv")
    if not os.path.exists(path):
        return None
    P = pd.read_csv(path).merge(std[["run_id", "ma_ratio_init", "ma_thresh_init", "label"]], on="run_id")
    idx = pd.Index(std.run_id).get_indexer(P.run_id)         # rows of std/ens for each scored run
    assert (idx >= 0).all()
    probs = P[[f"p_{c}" for c in CLASSES]].to_numpy()
    pred = probs.argmax(1)
    Vr = V[idx]; q = Vr / K                                  # vote fractions over 4 outcomes
    res = {"n_scored": int(len(P))}

    # --- standard test (single realization) & majority label
    s_ok = P.standard_label.isin(CLASSES).values
    m_ok = P.majority_label.isin(CLASSES).values
    std_y = np.array([CLASSES.index(l) if l in CLASSES else -1 for l in P.standard_label])
    maj_y = np.array([CLASSES.index(l) if l in CLASSES else -1 for l in P.majority_label])
    res["acc_vs_standard"] = float((pred[s_ok] == std_y[s_ok]).mean())
    res["acc_vs_majority"] = float((pred[m_ok] == maj_y[m_ok]).mean())
    # expected accuracy on a random realization = mean share of votes the prediction would have won
    res["expected_acc_vs_votes"] = float(q[np.arange(len(q)), pred].mean())
    res["ceiling_plug_in_same_rows"] = float(q.max(1).mean())
    res["ceiling_loo_same_rows"] = float(loo_ceiling(Vr).mean())

    # --- by agreement level
    agree = Vr.max(1).astype(int)
    by = {}
    for a in sorted(set(agree), reverse=True):
        mk = (agree == a) & m_ok
        by[f"{a}/{K}"] = {"n": int(mk.sum()),
                          "acc_vs_majority": float((pred[mk] == maj_y[mk]).mean()) if mk.any() else None,
                          "acc_vs_standard": float((pred[mk & s_ok] == std_y[mk & s_ok]).mean()) if (mk & s_ok).any() else None,
                          "mean_pmax": float(probs[mk].max(1).mean()) if mk.any() else None}
    res["by_agreement"] = by

    # --- calibration vs vote fractions (model puts 0 on Unresolved)
    p4 = np.column_stack([probs, np.zeros(len(probs))])
    brier = ((p4 - q) ** 2).sum(1)
    # if the model's probabilities were the true outcome probabilities, votes ~ Multinomial(K, p):
    # E[sum (p - q)^2] = sum p(1-p) / K   -> the noise floor a perfectly calibrated model would show
    floor = (probs * (1 - probs)).sum(1) / K
    res["calibration"] = {"brier_vs_votes": float(brier.mean()),
                          "brier_noise_floor_if_calibrated": float(floor.mean()),
                          "mean_total_variation": float(0.5 * np.abs(p4 - q).sum(1).mean())}
    # reliability: predicted confidence vs share of votes for the predicted class
    conf = probs.max(1); hit = q[np.arange(len(q)), pred]
    bins = np.linspace(1 / 3, 1, 8); rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (conf >= lo) & (conf < hi + (hi == 1))
        if mk.sum():
            rel.append({"bin": [round(lo, 3), round(hi, 3)], "n": int(mk.sum()),
                        "mean_conf": float(conf[mk].mean()), "mean_vote_share": float(hit[mk].mean())})
    res["reliability"] = rel
    res["ece_vs_votes"] = float(sum(r["n"] * abs(r["mean_conf"] - r["mean_vote_share"]) for r in rel) / len(conf))

    # --- MA baseline: Survived vs not, standard test labels
    surv_true = (P.standard_label.values == "Survived")
    ma_pred = (P.ma_ratio_init > P.ma_thresh_init).values
    mdl_pred = (pred == CLASSES.index("Survived"))
    slices = {"all resolved": s_ok,
              "metastable (Survived, MA fails)": s_ok & surv_true & ~ma_pred,
              "MA passes but disrupted": s_ok & ~surv_true & ma_pred,
              "near MA boundary (ratio/thresh in 0.8-1.25)": s_ok & ((P.ma_ratio_init / P.ma_thresh_init).between(0.8, 1.25)).values}
    res["survival_binary"] = {k: {"n": int(m.sum()),
                                  "MA_acc": float((ma_pred[m] == surv_true[m]).mean()) if m.any() else None,
                                  "model_acc": float((mdl_pred[m] == surv_true[m]).mean()) if m.any() else None}
                              for k, m in slices.items()}
    return res


out = {"ceiling": ceil, "models": {}}
for fs in FEATURE_SETS:
    for model in ["logreg", "mlp"]:
        r = model_eval(fs, model)
        if r is not None:
            r["ensemble_comparison_valid"] = fs in IC_ONLY      # trajectory sets: siblings' trajectories not saved
            out["models"][f"{fs}/{model}"] = r
with open(os.path.join(OUT, "evaluation.json"), "w") as f:
    json.dump(out, f, indent=1)

# ---- markdown tables ----
L = [f"Chaos ceiling (K={K}): plug-in {ceil['plug_in_mean_agreement']:.3f}, leave-one-out {ceil['leave_one_out']:.3f}; "
     f"unanimous {ceil['share_unanimous']:.1%}; agreement counts {ceil['agreement_distribution']}", "",
     "| Model | acc vs standard | acc vs majority | expected acc on a realization | ceiling (plug-in / LOO, same rows) | Brier vs votes (floor) | ECE |",
     "|---|---|---|---|---|---|---|"]
for k, r in out["models"].items():
    c = r["calibration"]
    L.append(f"| {k}{'' if r['ensemble_comparison_valid'] else ' *'} | {r['acc_vs_standard']:.3f} | {r['acc_vs_majority']:.3f} | "
             f"{r['expected_acc_vs_votes']:.3f} | {r['ceiling_plug_in_same_rows']:.3f} / {r['ceiling_loo_same_rows']:.3f} | "
             f"{c['brier_vs_votes']:.3f} ({c['brier_noise_floor_if_calibrated']:.3f}) | {r['ece_vs_votes']:.3f} |")
L += ["", "* trajectory feature sets: ensemble columns shown for reference only (perturbed siblings' trajectories are not saved).", "",
      "| Model | " + " | ".join(out["models"][next(iter(out["models"]))]["survival_binary"].keys()) + " |", "|---|---|---|---|---|"]
for k, r in out["models"].items():
    L.append(f"| {k} | " + " | ".join(f"MA {v['MA_acc']:.3f} vs model {v['model_acc']:.3f} (n={v['n']})" if v["n"] else "n=0"
                                     for v in r["survival_binary"].values()) + " |")
open(os.path.join(OUT, "evaluation_tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
