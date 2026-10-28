"""Draw the README figures (SVG) from the study's reported results (CPU, needs matplotlib).

python scripts/make_readme_figures.py      # writes docs/img/*.svg
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "svg.fonttype": "path",
})
DARK, MID, LIGHT, GREY = "#1f3b5c", "#6f8fb3", "#c9d6e6", "#8c8c8c"
W = 0.36

# ---- audit pipeline ---------------------------------------------------------------------------
steps = [
    ("Data", "prices:\nYahoo, ccxt\nnews: Benzinga"),
    ("Leakage", "training\ncutoffs, news\ntimestamps"),
    ("Forecast", "Kronos,\nChronos,\nLlama-3.1-8B"),
    ("Controls", "shuffled\nnews,\nanonymized"),
    ("Calibrate", "online\nEMOS + ACI"),
    ("Score", "CRPS vs\nGaussian,\npooled tests"),
    ("Replicate", "pre-\nregistered\nheld-out test"),
]
fig, ax = plt.subplots(figsize=(12, 2.6))
ax.set_xlim(0, len(steps))
ax.set_ylim(0, 1)
ax.axis("off")
for i, (head, body) in enumerate(steps):
    x = i + 0.06
    box = FancyBboxPatch((x, 0.12), 0.88, 0.76, boxstyle="round,pad=0.0,rounding_size=0.06",
                         facecolor=LIGHT if i % 2 else "#e8eef6", edgecolor=MID, linewidth=0.9)
    ax.add_patch(box)
    ax.text(x + 0.44, 0.76, head, ha="center", va="center", fontsize=10.5, fontweight="bold", color=DARK)
    ax.text(x + 0.44, 0.40, body, ha="center", va="center", fontsize=8.5, color="#222222", linespacing=1.3)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + 0.89, 0.5), (x + 1.05, 0.5), arrowstyle="-|>", mutation_scale=11,
                                     color=DARK, linewidth=1.0))
fig.savefig(OUT / "audit_pipeline.svg")
plt.close(fig)

# ---- reversion slopes -------------------------------------------------------------------------
models = ["Kronos\nmini", "Kronos\nsmall", "Kronos\nbase", "Chronos\nT5-small"]
real = {64: [-0.042, -0.188, -0.190, 0.0017], 256: [0.016, -0.546, -0.704, 0.0009]}
t_dk = {64: [-3.165, -10.423, -10.851, 0.206], 256: [2.262, -22.582, -33.096, 0.268]}
syn = {64: [-0.045, -0.079, -0.158, 0.03], 256: [0.037, -0.324, -0.418, 0.01]}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
x = np.arange(len(models))
for ax, L in zip(axes, (64, 256)):
    se = np.abs(np.array(real[L]) / np.array(t_dk[L]))
    ax.bar(x - W / 2, real[L], W, color=DARK, label="29 US assets (real)", zorder=3)
    ax.errorbar(x - W / 2, real[L], yerr=2.0 * se, fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, zorder=4)
    ax.bar(x + W / 2, syn[L], W, color=LIGHT, edgecolor=MID, linewidth=0.7, label="Synthetic random walks (truth = 0)",
           zorder=3)
    ax.axhline(0, color="black", linewidth=0.7, zorder=2)
    ax.set_xticks(x, models)
    ax.set_title(f"Lookback {L} bars")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6, zorder=0)
    ax.set_ylim(-0.8, 0.12)
axes[0].set_ylabel("Slope of mean forecast on gap")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "reversion_slopes.svg")
plt.close(fig)

# ---- false-positive rates ---------------------------------------------------------------------
modes = ["Volatility-scaled\nranges", "Volatility-scaled,\nσ partialled out", "Fixed ranges", "Direction labels"]
rate_real = np.array([95.0, 3.5, 4.5, 3.0])
rate_shuf = np.array([93.5, 10.5, 8.0, 4.0])


def ci(p, n=200):
    p = p / 100
    return 1.96 * np.sqrt(p * (1 - p) / n) * 100


fig, ax = plt.subplots(figsize=(9, 3.3))
x = np.arange(len(modes))
ax.bar(x - W / 2, rate_real, W, color=DARK, label="Real headlines", zorder=3)
ax.bar(x + W / 2, rate_shuf, W, color=LIGHT, edgecolor=MID, linewidth=0.7, label="Shuffled headlines", zorder=3)
for xs, r in ((x - W / 2, rate_real), (x + W / 2, rate_shuf)):
    ax.errorbar(xs, r, yerr=ci(r), fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, zorder=4)
    for xi, ri in zip(xs, r):
        ax.text(xi, ri + ci(ri) + 1.5, f"{ri:g}%", ha="center", va="bottom", fontsize=8.5, zorder=5)
ax.axhline(5, color=GREY, linestyle="--", linewidth=0.9, zorder=2, label="Nominal 5% level")
ax.set_xticks(x, modes)
ax.set_ylabel("Panels rejecting (%)")
ax.set_ylim(0, 110)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis="y", color="#e5e5e5", linewidth=0.6, zorder=0)
ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.13))
fig.tight_layout()
fig.savefig(OUT / "false_positive_rates.svg")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
