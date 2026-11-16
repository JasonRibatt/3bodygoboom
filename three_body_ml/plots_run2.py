"""Run 2 figures -> outputs/run2/figures/"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from common import OUT, CLASSES, HELDOUT_BLOCKS, load_grid

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
CLS_COL = {"Ejection": BLUE, "Survived": ORANGE, "Collision": AQUA, "Unresolved": GREY}
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#a9c9f0", "#2a78d6", "#0d3b73"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
FS = {"fs1_ic": "(1) Initial conditions", "fs2_ic_conserved": "(2) IC + E0, Lz0",
      "fs3_early_traj": "(3) Trajectory t=1–10", "fs4_ic_late_state": "(4) IC + state at t=20"}
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
EV = json.load(open(os.path.join(OUT, "evaluation.json")))
REP = {f: json.load(open(os.path.join(OUT, f, "report.json"))) for f in FS}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160); plt.close(fig)


# 1. outcome map with held-out blocks
def outcome_map():
    g, _ = load_grid()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    da, de = 0.1125, 0.95 / 24
    rng = np.random.default_rng(0)
    for lab in ["Unresolved", "Ejection", "Survived", "Collision"]:
        s = g[g.label == lab]
        ax.scatter(s.a_out + (s.phase_out / (2 * np.pi) - 0.5) * da * 0.8, s.e_out + (rng.random(len(s)) - 0.5) * de * 0.8,
                   s=3 if lab != "Unresolved" else 14, c=CLS_COL[lab], lw=0, label=f"{lab} ({len(s):,})")
    a = np.linspace(0.9, 3.6, 400); e = np.linspace(0, 0.95, 400); A, E = np.meshgrid(a, e)
    thr = 2.8 * 1.5 ** 0.4 * (1 + E) ** 0.4 / (1 - E) ** 1.2
    cs = ax.contour(A, E, A / 0.5 - thr, levels=[0], colors=INK, linewidths=1.2, linestyles="-")
    ax.text(3.3, 0.27, "Mardling–Aarseth\nboundary", fontsize=8, color=INK, ha="center", bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=1.5))
    for bn, b in HELDOUT_BLOCKS.items():
        (a0, a1), (e0, e1) = b["a_out"], b["e_out"]
        ax.add_patch(Rectangle((a0 - da / 2, e0 - de / 2), a1 - a0 + da, e1 - e0 + de, fill=False, ec=INK, lw=1.5, ls="--"))
        ax.text(a1 + da / 2 + 0.03, e1 + de / 2, "held-out\n" + {"block_ES": "E/S", "block_EC": "E/C"}[bn], va="top", fontsize=8.5)
    ax.set_xlabel("a_out"); ax.set_ylabel("e_out"); ax.set_xlim(0.85, 3.65); ax.set_ylim(-0.02, 0.97)
    ax.set_title("Run 2 outcomes by initial condition (25 phases per cell, jittered)", loc="left", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, markerscale=4, fontsize=9)
    fig.tight_layout(); save(fig, "outcome_map_run2.png")


# 2. headline: accuracy vs ceiling
def headline():
    rows = [("fs1_ic", "logreg"), ("fs1_ic", "mlp"), ("fs2_ic_conserved", "logreg"), ("fs2_ic_conserved", "mlp"),
            ("fs3_early_traj", "mlp"), ("fs4_ic_late_state", "mlp")]
    lab = [f"{FS[f]} · {'MLP' if m == 'mlp' else 'LogReg'}" for f, m in rows]
    std = [EV["models"][f"{f}/{m}"]["acc_vs_standard"] for f, m in rows]
    maj = [EV["models"][f"{f}/{m}"]["acc_vs_majority"] for f, m in rows]
    c = EV["ceiling"]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    y = np.arange(len(rows))
    ax.axvspan(c["leave_one_out"], c["plug_in_mean_agreement"], color=GREY, alpha=0.35, lw=0)
    ax.text((c["leave_one_out"] + c["plug_in_mean_agreement"]) / 2, -0.95, f"chaos ceiling\n{c['leave_one_out']:.3f}–{c['plug_in_mean_agreement']:.3f}",
            ha="center", va="bottom", fontsize=8, color=INK2)
    base = 875 / 1996
    ax.axvline(base, color=INK2, lw=1, ls=":"); ax.text(base + 0.004, -0.95, f"majority class\n{base:.3f}", fontsize=8, color=INK2, va="bottom")
    ax.barh(y - 0.19, std, 0.36, color=BLUE, label="vs single-run (standard) label")
    ax.barh(y + 0.19, maj, 0.36, color=ORANGE, label="vs majority-vote label (K=5)")
    for yy, a, b in zip(y, std, maj):
        ax.text(a + 0.004, yy - 0.19, f"{a:.3f}", va="center", fontsize=8, color=INK2)
        ax.text(b + 0.004, yy + 0.19, f"{b:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, lab, fontsize=9); ax.set_ylim(len(rows) - 0.4, -1.1); ax.set_xlim(0.4, 0.96)
    ax.set_xlabel("test accuracy (2,000 off-grid runs)"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    ax.set_title("Test accuracy against the chaos ceiling", loc="left", fontsize=11, pad=30)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(h, l, loc="lower left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 0.045))
    fig.text(0.015, 0.012, "Trajectory models (3, 4) score fewer rows: runs that collide before their snapshots are excluded.",
             fontsize=7.5, color=INK2, ha="left")
    save(fig, "accuracy_vs_ceiling.png")


# 3. accuracy by agreement level
def by_agreement():
    ms = [("fs1_ic/mlp", "(1) IC · MLP", BLUE), ("fs2_ic_conserved/mlp", "(2) IC+E0,Lz0 · MLP", ORANGE),
          ("fs4_ic_late_state/mlp", "(4) IC+t=20 · MLP*", AQUA)]
    levels = ["5/5", "4/5", "3/5"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(levels)); w = 0.26
    for i, (k, name, col) in enumerate(ms):
        v = [EV["models"][k]["by_agreement"][l]["acc_vs_majority"] for l in levels]
        n = [EV["models"][k]["by_agreement"][l]["n"] for l in levels]
        ax.bar(x + (i - 1) * w, v, w * 0.92, color=col, label=name)
        for xx, vv in zip(x + (i - 1) * w, v):
            ax.text(xx, vv + 0.012, f"{vv:.2f}", ha="center", fontsize=8, color=INK2)
    nlab = [f"{l} agree\n(n={EV['models']['fs1_ic/mlp']['by_agreement'][l]['n']})" for l in levels]
    ax.set_xticks(x, nlab); ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy vs majority label")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Accuracy by ensemble agreement (* trajectory model: fewer 5/5 rows)", loc="left", fontsize=10.5)
    fig.tight_layout(); save(fig, "accuracy_by_agreement.png")


# 4. reliability vs vote fractions
def reliability():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
    for ax, (k, name) in zip(axes, [("fs1_ic/mlp", "(1) Initial conditions · MLP"), ("fs4_ic_late_state/mlp", "(4) IC + t=20 · MLP")]):
        r = EV["models"][k]["reliability"]
        conf = [b["mean_conf"] for b in r]; share = [b["mean_vote_share"] for b in r]; n = [b["n"] for b in r]
        ax.plot([1 / 3, 1], [1 / 3, 1], color=GREY, lw=1, ls="--")
        ax.plot(conf, share, "-o", color=BLUE, ms=6, lw=1.6, mec=SURF, mew=1.5)
        for c_, s_, n_ in zip(conf, share, n):
            ax.text(c_ + 0.012, s_ - 0.035, f"n={n_}", fontsize=7, color=INK2)
        cal = EV["models"][k]["calibration"]
        ax.text(0.36, 0.95, f"ECE vs votes {EV['models'][k]['ece_vs_votes']:.3f}\nBrier {cal['brier_vs_votes']:.3f} (floor {cal['brier_noise_floor_if_calibrated']:.3f})",
                fontsize=8, color=INK2, va="top")
        ax.set_xlim(0.32, 1.02); ax.set_ylim(0.32, 1.02); ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("model confidence (max softmax)"); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("share of K=5 votes for the predicted class")
    fig.suptitle("Calibration against the ensemble votes", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); save(fig, "calibration_vs_votes.png")


# 5. MA baseline vs models (Survived vs not)
def ma_baseline():
    sl = ["all resolved", "near MA boundary (ratio/thresh in 0.8-1.25)", "metastable (Survived, MA fails)"]
    nice = ["All test runs", "Near the MA boundary\n(ratio/threshold 0.8–1.25)", "Metastable survivors\n(Survived, MA fails)"]
    ms = [("fs1_ic/logreg", "(1) IC · LogReg", ORANGE), ("fs1_ic/mlp", "(1) IC · MLP", BLUE), ("fs4_ic_late_state/mlp", "(4) IC+t=20 · MLP", AQUA)]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(sl)); w = 0.2
    ma = [EV["models"]["fs1_ic/mlp"]["survival_binary"][s]["MA_acc"] for s in sl]
    ax.bar(x - 1.5 * w, ma, w * 0.92, color=GREY, label="Mardling–Aarseth at t=0")
    for i, (k, name, col) in enumerate(ms):
        ax.bar(x + (i - 0.5) * w, [EV["models"][k]["survival_binary"][s]["model_acc"] for s in sl], w * 0.92, color=col, label=name)
    for i in range(len(sl)):
        vals = [ma[i]] + [EV["models"][k]["survival_binary"][sl[i]]["model_acc"] for k, _, _ in ms]
        for j, v in enumerate(vals):
            if not (j == 0 and v == 0):
                ax.text(x[i] + (j - 1.5) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ns = [EV["models"]["fs1_ic/mlp"]["survival_binary"][s]["n"] for s in sl]
    ax.set_xticks(x, [f"{a}\nn={n}" for a, n in zip(nice, ns)], fontsize=8.5)
    ax.set_ylim(0, 1.1); ax.set_ylabel("accuracy: Survived vs not")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.text(x[2] - 1.5 * w, 0.06, "0 by\ndefinition", ha="center", fontsize=7, color=INK2)
    fig.suptitle("Predicting survival: models vs the Mardling–Aarseth criterion", x=0.01, ha="left", fontsize=10.5)
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.legend(h, l, loc="upper left", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.01, 0.93))
    save(fig, "ma_baseline.png")


# 6. confusion matrices on the standard test set
def confusions():
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.0))
    short = ["Eject", "Surv", "Collide"]
    for j, fs in enumerate(FS):
        for i, m in enumerate(["logreg", "mlp"]):
            ax = axes[i, j]; r = REP[fs]["models"][m]["test_standard"]
            cm = np.array(r["confusion"], float); rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(rown, cmap=SEQ, vmin=0, vmax=1)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f"{int(cm[a, b])}", ha="center", va="center", fontsize=9, color="white" if rown[a, b] > 0.55 else INK)
            ax.set_xticks(range(3), short, fontsize=8); ax.set_yticks(range(3), short, fontsize=8)
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_title(f"{FS[fs]}\n{'LogReg' if m == 'logreg' else 'MLP'}  acc {r['accuracy']:.3f} · F1 {r['macro_f1']:.3f}", fontsize=8.5, loc="left")
    fig.suptitle("Confusion matrices — off-grid standard test set (rows = true; shading = recall)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); save(fig, "confusion_test_standard.png")


if __name__ == "__main__":
    outcome_map(); headline(); by_agreement(); reliability(); ma_baseline(); confusions()
    print("ok", sorted(os.listdir(FIG)))
