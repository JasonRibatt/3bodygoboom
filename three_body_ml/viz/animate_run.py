"""Animate one three-body run from the pilot grid: an exact replay of data_generator.simulate_encounter.
Usage (from three_body_ml/): python viz/animate_run.py RUN_ID [out.mp4]   (needs pip install rebound, and ffmpeg)
The label, event time and ejected star come from Jonah's own classification code (see viz/replay.py)."""
import sys
import numpy as np
import rebound
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
import shutil
if shutil.which("ffmpeg") is None:          # no system ffmpeg: use the pip one (pip install imageio-ffmpeg)
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load_grid
from replay import replay, check_against_npz

RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 3277
OUT = sys.argv[2] if len(sys.argv) > 2 else f"run_{RUN}.mp4"
SECONDS = int(sys.argv[3]) if len(sys.argv) > 3 else 30

SURF, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3936"
COL = ["#3987e5", "#d95926", "#199e70"]   # body 0, 1 (original binary), 2 (incoming)
NAMES = ["Star A (binary)", "Star B (binary)", "Star C (incoming)"]

grid, grid_traj = load_grid()
row = grid.loc[RUN]
info = replay(row.a_out, row.e_out, row.phase_out)
label, t_ev = info["label"], info["t_event"]
if label != row.label:
    print(f"WARNING: replay label {label!r} != dataset label {row.label!r}")
try:
    saved = grid_traj[RUN]
    print(f"replay vs saved trajectory (t=1..20): max |diff| = {check_against_npz(info, saved):.1e}")
except FileNotFoundError:
    pass

# resample the adaptive IAS15 steps onto an even time grid for the animation
t_raw, xy_raw = info["t"], info["xy"]
FPS = 30
ts = np.linspace(t_raw[0], t_raw[-1], max(FPS * SECONDS * 8, 2000))
pos = np.stack([np.stack([np.interp(ts, t_raw, xy_raw[:, i, d]) for d in range(2)], -1) for i in range(3)], 1)
dist = np.linalg.norm(pos, axis=2)
ejected = info["ejected"]
print(f"run {RUN}: dataset {row.label} | replay {label} at t={t_ev:.1f}"
      + (f", ejected star index {ejected}" if ejected is not None else ""))

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig = plt.figure(figsize=(7.2, 8.6), facecolor=SURF, dpi=150)
ax = fig.add_axes([0.08, 0.30, 0.88, 0.62], facecolor=SURF)
ax2 = fig.add_axes([0.10, 0.07, 0.86, 0.16], facecolor=SURF)
ax.set_aspect("equal"); ax.tick_params(labelsize=8)
for s in ["top", "right"]: ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
ax.set_xlabel("x", fontsize=9); ax.set_ylabel("y", fontsize=9)

fig.text(0.04, 0.965, f"Three-body run {RUN} — {label}", fontsize=14, weight="bold", color=INK)
fig.text(0.04, 0.938, f"a_out = {row.a_out:.3f}   e_out = {row.e_out:.3f}   phase = {row.phase_out:.3f}   "
         f"(equal masses, inner binary a_in = 0.5, REBOUND IAS15)", fontsize=8.5, color=INK2)
clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=10, color=INK, va="top", family="DejaVu Sans Mono")
note = ax.text(0.98, 0.03, "", transform=ax.transAxes, fontsize=9.5, color=INK, ha="right", va="bottom")

TRAIL = max(2, int(25 / (ts[1] - ts[0])))          # 25 time units of trail
trails, dots = [], []
for i in range(3):
    lc = LineCollection([], linewidths=1.4, capstyle="round"); ax.add_collection(lc); trails.append(lc)
    d, = ax.plot([], [], "o", ms=7, color=COL[i], mec=SURF, mew=1.5, zorder=5, label=NAMES[i]); dots.append(d)
ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)
ax2.axvline(t_ev, color=INK2, lw=0.8, ls="--")

for i in range(3):
    ax2.plot(ts, dist[:, i], color=COL[i], lw=1.0, alpha=0.35)
lines2 = [ax2.plot([], [], color=COL[i], lw=1.4)[0] for i in range(3)]
cursor = ax2.axvline(0, color=INK2, lw=0.8, ls=":")
ax2.set_yscale("log"); ax2.set_xlim(0, ts[-1]); ax2.set_ylim(0.05, max(dist.max() * 1.3, 20))
ax2.set_xlabel("time (N-body units)", fontsize=9); ax2.set_ylabel("distance\nfrom COM", fontsize=8.5)
ax2.tick_params(labelsize=8); ax2.grid(axis="y", color=GRID, lw=0.5)

def rgba(hexc, alphas):
    c = np.array(matplotlib.colors.to_rgba(hexc)); out = np.tile(c, (len(alphas), 1)); out[:, 3] = alphas; return out

frames = np.linspace(0, len(ts) - 1, FPS * SECONDS).astype(int)
half = 2.4
writer = FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast"])
with writer.saving(fig, OUT, dpi=150):
    for k in frames:
        lo = max(0, k - TRAIL)
        for i in range(3):
            seg = pos[lo:k + 1, i]
            if len(seg) > 1:
                segs = np.stack([seg[:-1], seg[1:]], axis=1)
                trails[i].set_segments(segs)
                trails[i].set_color(rgba(COL[i], np.linspace(0.0, 0.9, len(segs)) ** 1.5))
            dots[i].set_data([pos[k, i, 0]], [pos[k, i, 1]])
            lines2[i].set_data(ts[:k + 1], dist[:k + 1, i])
        # smooth auto-zoom: keep every body in view, never tighter than 2.4
        target = max(2.4, np.abs(pos[k]).max() * 1.15)
        half = target if target > half else half + 0.05 * (target - half)   # zoom out instantly, in gently
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        cursor.set_xdata([ts[k], ts[k]])
        clock.set_text(f"t = {ts[k]:6.1f}")
        if ts[k] >= t_ev - 1e-9:
            if label == "Ejection":
                note.set_text(f"t = {t_ev:.0f}: {NAMES[ejected].split(' (')[0]} meets the ejection criteria\n"
                              "(inner pair bound, outer E > 0.01, moving out, r > 50 a_in)"
                              + ("\nshowing 60 time units after classification" if ts[k] > t_ev + 1 else ""))
            elif label == "Collision":
                note.set_text(f"t = {t_ev:.2f}: collision (stars touch, r = 0.005)")
            else:
                note.set_text(f"t = {t_ev:.0f}: classified {label}")
        writer.grab_frame()
print("wrote", OUT)
