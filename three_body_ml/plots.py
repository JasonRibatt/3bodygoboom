"""Figures: outcome map with held-out blocks, confusion matrices, and model comparison."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from common import OUT, CLASSES, HELDOUT_BLOCKS, load_meta

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLS_COL = {"Ejection": "#2a78d6", "Stable Hierarchical": "#eb6834", "Collision": "#1baf7a", "Unresolved": "#b9b8b2"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#a9c9f0", "#2a78d6", "#0d3b73"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
FS_LABEL = {"fs1_ic": "(1) Initial conditions", "fs2_ic_conserved": "(2) IC + E0, Lz0",
            "fs3_early_traj": "(3) Early trajectory t=1–10", "fs4_ic_late_state": "(4) IC + state at t=20"}


def outcome_map(df, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    da, de = 0.1125, 0.95 / 24
    for lab in ["Unresolved", "Ejection", "Stable Hierarchical", "Collision"]:
        s = df[df.label == lab]
        # jitter by phase so the 25 phases per cell are visible
        jx = s.a_out + (s.phase_out / (2 * np.pi) - 0.5) * da * 0.8
        jy = s.e_out + (np.random.default_rng(0).random(len(s)) - 0.5) * de * 0.8
        ax.scatter(jx, jy, s=3, c=CLS_COL[lab], label=f"{lab} ({len(s):,})", lw=0, alpha=0.9)
    for bn, b in HELDOUT_BLOCKS.items():
        (a0, a1), (e0, e1) = b["a_out"], b["e_out"]
        ax.add_patch(Rectangle((a0 - da / 2, e0 - de / 2), a1 - a0 + da, e1 - e0 + de, fill=False, ec=INK, lw=1.6, ls="--"))
        ax.text(a1 + da / 2 + 0.03, e1 + de / 2, "held-out\n" + bn.split("_")[1][0] + "/" + bn.split("_")[1][1],
                va="top", fontsize=8.5, color=INK)
    ax.set_xlabel("a_out (outer semi-major axis)"); ax.set_ylabel("e_out (outer eccentricity)")
    ax.set_title("Outcome by initial condition (points jittered by phase)", loc="left", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, markerscale=4, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def confusion_grid(reports, split, path):
    fss = [f for f in FS_LABEL if f in reports]
    fig, axes = plt.subplots(2, len(fss), figsize=(3.1 * len(fss), 6.0), squeeze=False)
    short = ["Eject", "Stable", "Collide"]
    for j, fs in enumerate(fss):
        for i, m in enumerate(["logreg", "mlp"]):
            ax = axes[i, j]
            cm = np.array(reports[fs]["models"][m][split]["confusion"], float)
            rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(rown, cmap=SEQ, vmin=0, vmax=1)
            for r in range(3):
                for c in range(3):
                    ax.text(c, r, f"{int(cm[r, c])}", ha="center", va="center", fontsize=9,
                            color="white" if rown[r, c] > 0.55 else INK)
            ax.set_xticks(range(3), short, fontsize=8); ax.set_yticks(range(3), short, fontsize=8)
            for s in ax.spines.values(): s.set_visible(False)
            acc = reports[fs]["models"][m][split]["accuracy"]; f1 = reports[fs]["models"][m][split]["macro_f1"]
            ax.set_title(f"{FS_LABEL[fs]}\n{'LogReg' if m == 'logreg' else 'MLP'}  acc {acc:.3f} · F1 {f1:.3f}", fontsize=8.5, loc="left")
            if j == 0: ax.set_ylabel("true", fontsize=8)
            if i == 1: ax.set_xlabel("predicted", fontsize=8)
    fig.suptitle(f"Confusion matrices — {split} split (counts; shading = row-normalized recall)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def comparison(reports, path):
    fss = [f for f in FS_LABEL if f in reports]
    evals = [("test", "Test accuracy"), ("test", "Test macro-F1"), ("block_EC", "Held-out E/C block accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    col = {"logreg": "#2a78d6", "mlp": "#eb6834"}
    y = np.arange(len(fss))
    for ax, (sp, title) in zip(axes, evals):
        key = "macro_f1" if "F1" in title else "accuracy"
        for k, m in enumerate(["logreg", "mlp"]):
            v = [reports[f]["models"][m][sp][key] for f in fss]
            ax.barh(y + (k - 0.5) * 0.38, v, height=0.34, color=col[m], label="LogReg" if m == "logreg" else "MLP")
            for yy, vv in zip(y, v): ax.text(vv + 0.008, yy + (k - 0.5) * 0.38, f"{vv:.3f}", va="center", fontsize=8, color=INK2)
        if sp == "test" and key == "accuracy":
            ax.axvline(0.532, color=INK2, lw=1, ls=":"); ax.text(0.537, -0.62, "majority class 53.2%", fontsize=8, color=INK2, va="bottom")
        ax.set_xlim(0.4, 1.06); ax.set_ylim(len(fss) - 0.4, -0.75); ax.set_title(title, loc="left", fontsize=10); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_yticks(y, [FS_LABEL[f] for f in fss])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    df = load_meta()
    reports = {f: json.load(open(os.path.join(OUT, f, "report.json"))) for f in FS_LABEL if os.path.exists(os.path.join(OUT, f, "report.json"))}
    fig = os.path.join(OUT, "figures"); os.makedirs(fig, exist_ok=True)
    outcome_map(df, os.path.join(fig, "outcome_map_heldout_blocks.png"))
    for sp in ["test", "heldout_block"]:
        confusion_grid(reports, sp, os.path.join(fig, f"confusion_{sp}.png"))
    comparison(reports, os.path.join(fig, "model_comparison.png"))
    print("ok", list(reports))
