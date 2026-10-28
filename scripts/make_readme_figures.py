"""Draw the README figures (SVG) from the study's reported results (CPU, needs matplotlib).

python scripts/make_readme_figures.py      # writes docs/img/*.svg

Palette: eucalyptus / pistachio / peach / ivory / clay, deep-olive ink; serif display with mono captions.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

EUC, PIS, PEACH, IVORY, CLAY = "#CDD4B1", "#EBECCC", "#FEECD0", "#FFF9E2", "#DCA278"
INK, MUTED, CLAY_D, EUC_D = "#2E3529", "#5A6250", "#A5603A", "#76845A"
HAIR = "#A7A893"
GRID = "#E6E0C6"

_names = {f.name for f in fm.fontManager.ttflist}
SERIF = next((n for n in ("EB Garamond", "Garamond", "Georgia", "Cambria") if n in _names), "serif")
MONO = next((n for n in ("Cascadia Mono", "Consolas", "DejaVu Sans Mono") if n in _names), "monospace")

plt.rcParams.update({
    "font.family": SERIF, "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.edgecolor": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": IVORY, "axes.facecolor": IVORY, "savefig.facecolor": IVORY,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.18, "svg.fonttype": "path",
})


def mono(ax, x, y, s, size=9.5, color=MUTED, **kw):
    ax.text(x, y, s.upper(), fontfamily=MONO, fontsize=size, color=color, **kw)


def frame(fig):
    """Hairline border around the whole figure, like a printed card."""
    fig.add_artist(Rectangle((0.004, 0.004), 0.992, 0.992, transform=fig.transFigure, fill=False,
                             edgecolor=HAIR, linewidth=0.8))


def heading(fig, title, kicker, y):
    fig.text(0.03, y + 0.045, kicker.upper(), fontfamily=MONO, fontsize=9.5, color=MUTED, ha="left", va="bottom")
    fig.text(0.03, y - 0.01, title, fontsize=21, color=INK, ha="left", va="bottom")


def style(ax, grid="y"):
    ax.tick_params(length=0, labelsize=10.5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(MONO)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def legend(fig, ax, x=0.98, y=0.985):
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", bbox_to_anchor=(x, y), ncol=len(h), frameon=False,
               prop={"family": MONO, "size": 9.5})


# ---- hero ---------------------------------------------------------------------------------------
def hero():
    fig = plt.figure(figsize=(12.8, 4.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 420)
    ax.axis("off")
    ax.add_patch(Rectangle((14, 14), 1252, 392, fill=False, edgecolor=HAIR, linewidth=1))
    # colour-card mosaic on the right
    x0, y0, w, h = 760, 40, 480, 340
    tiles = [(0, 0.45, 0.5, 0.55, EUC, "Kronos", "price paths"), (0, 0, 0.5, 0.45, PIS, "Chronos", "price paths"),
             (0.5, 0.66, 0.5, 0.34, PEACH, "Llama-3.1", "reads the news"),
             (0.5, 0.33, 0.5, 0.33, IVORY, "29 assets", "US large caps"),
             (0.5, 0, 0.5, 0.33, CLAY, "2024–26", "out of sample")]
    for tx, ty, tw, th, c, nm, sub in tiles:
        X, Y, Wd, Ht = x0 + tx * w, y0 + ty * h, tw * w, th * h
        ax.add_patch(Rectangle((X, Y), Wd, Ht, facecolor=c, edgecolor=HAIR, linewidth=1))
        ax.text(X + 18, Y + 36, nm.upper(), fontsize=18, color=INK, fontfamily=SERIF)
        mono(ax, X + 18, Y + 16, sub, size=8.5)
    # wordmark and tagline on the left
    mono(ax, 70, 336, "Open-source audit harness  ·  MIT", size=10)
    ax.text(62, 222, "P-Quant", fontsize=80, color=INK, fontfamily=SERIF)
    ax.text(70, 166, "An open-source audit for", fontsize=24, color=CLAY_D, fontstyle="italic", fontfamily=SERIF)
    ax.text(70, 130, "market forecasting models", fontsize=24, color=CLAY_D, fontstyle="italic", fontfamily=SERIF)
    ax.plot([70, 680], [100, 100], color=CLAY_D, linewidth=1.2)
    mono(ax, 70, 64, "Kronos · Chronos · Llama-3.1-8B · 29 US assets · Jul 2024 – Oct 2026", size=9.5)
    fig.savefig(OUT / "hero.svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)


# ---- audit pipeline -----------------------------------------------------------------------------
def pipeline():
    steps = [("Data", "prices +\nheadlines"), ("Leakage", "training\ncutoffs"), ("Forecast", "Kronos ·\nChronos ·\nLlama"),
             ("Placebo", "shuffled\nnews"), ("Calibrate", "online\nEMOS + ACI"), ("Score", "CRPS vs\nGaussian"),
             ("Replicate", "pre-\nregistered")]
    colors = [EUC, PIS, PEACH, IVORY, EUC, PIS, CLAY]
    fig = plt.figure(figsize=(12.8, 3.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1.25)
    ax.axis("off")
    for i, ((h, b), c) in enumerate(zip(steps, colors)):
        x = i + 0.05
        ax.add_patch(Rectangle((x, 0.1), 0.88, 0.9, facecolor=c, edgecolor=HAIR, linewidth=1))
        mono(ax, x + 0.07, 0.88, f"{i + 1:02d}", size=9)
        ax.text(x + 0.07, 0.64, h, fontsize=19, color=INK)
        ax.text(x + 0.07, 0.17, b.upper(), fontsize=8.5, color=MUTED, fontfamily=MONO, linespacing=1.45)
        if i < 6:
            ax.add_patch(FancyArrowPatch((x + 0.88, 0.55), (x + 1.0, 0.55), arrowstyle="-|>", mutation_scale=10,
                                         color=CLAY_D, linewidth=1))
    mono(ax, 0.05, 1.1, "Every forecast passes the same seven controls", size=10)
    fig.savefig(OUT / "audit_pipeline.svg")
    plt.close(fig)


# ---- reversion slopes ---------------------------------------------------------------------------
def reversion():
    models = ["Kronos\nmini", "Kronos\nsmall", "Kronos\nbase", "Chronos\nT5-small"]
    real = {64: [-0.042, -0.188, -0.190, 0.0017], 256: [0.016, -0.546, -0.704, 0.0009]}
    tdk = {64: [-3.165, -10.423, -10.851, 0.206], 256: [2.262, -22.582, -33.096, 0.268]}
    syn = {64: [-0.045, -0.079, -0.158, 0.03], 256: [0.037, -0.324, -0.418, 0.01]}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    fig.subplots_adjust(top=0.74, wspace=0.08)
    x, w = np.arange(4), 0.36
    for ax, L in zip(axes, (64, 256)):
        se = np.abs(np.array(real[L]) / np.array(tdk[L]))
        ax.bar(x - w / 2, real[L], w, color=CLAY_D, label="29 real US assets", zorder=3)
        ax.errorbar(x - w / 2, real[L], yerr=2 * se, fmt="none", ecolor=INK, elinewidth=0.9, capsize=3, zorder=4)
        ax.bar(x + w / 2, syn[L], w, color=EUC_D, label="random walks (truth = 0)", zorder=3)
        ax.axhline(0, color=INK, linewidth=0.9)
        ax.set_xticks(x, models)
        ax.set_ylim(-0.8, 0.12)
        style(ax)
        ax.set_title(f"{L}-BAR LOOKBACK", fontfamily=MONO, fontsize=9.5, color=MUTED, loc="left")
    axes[0].set_ylabel("slope toward the window mean", fontfamily=MONO, fontsize=10)
    heading(fig, "Bigger Kronos models pull harder toward the recent average", "Result · reversion", y=0.86)
    legend(fig, axes[0])
    frame(fig)
    fig.savefig(OUT / "reversion_slopes.svg")
    plt.close(fig)


# ---- error vs gaussian --------------------------------------------------------------------------
def crps():
    rows = [("Kronos-base, 64 bars", 16, 2.8), ("Kronos-base, 256 bars", 229, 3.4), ("Chronos-T5-small, 64 bars", 4, 4.1),
            ("Chronos-T5-small, 256 bars", -0.3, 1.1)]
    cap = 30
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    fig.subplots_adjust(top=0.74, left=0.27)
    y = np.arange(len(rows))[::-1] * 1.0
    for yi, (nm, raw, cor) in zip(y, rows):
        rv = min(raw, cap)
        ax.barh(yi + 0.17, rv, 0.3, color=CLAY, zorder=3, label="raw" if yi == y[0] else None)
        ax.barh(yi - 0.17, cor, 0.3, color=INK, zorder=3, label="corrected" if yi == y[0] else None)
        lab = f"+{raw:g}%" if raw > 0 else f"{raw:g}%".replace("-", "−")
        if raw > cap:
            ax.text(cap + 0.4, yi + 0.17, f"//  {lab}", va="center", fontfamily=MONO, fontsize=10.5, color=CLAY_D)
        else:
            ax.text(max(rv, 0) + 0.5, yi + 0.17, lab, va="center", fontfamily=MONO, fontsize=10.5, color=MUTED)
        ax.text(cor + 0.5, yi - 0.17, f"+{cor:g}%", va="center", fontfamily=MONO, fontsize=10.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    for lab in ax.get_yticklabels():
        lab.set_fontsize(14)
        lab.set_color(INK)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_xlim(-2, 40)
    ax.set_xlabel("CRPS vs a Gaussian benchmark (%)  ·  positive = worse", fontfamily=MONO, fontsize=10)
    ax.tick_params(length=0)
    for lab in ax.get_xticklabels():
        lab.set_fontfamily(MONO)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    heading(fig, "Corrected, still no better than a bell curve", "Result · vs. a Gaussian · 29 US assets", y=0.86)
    legend(fig, ax)
    frame(fig)
    fig.savefig(OUT / "crps_vs_gaussian.svg")
    plt.close(fig)


# ---- calibration --------------------------------------------------------------------------------
def calibration():
    target, got = [50, 80, 90], [52, 82, 91]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.subplots_adjust(top=0.72)
    x, w = np.arange(3), 0.34
    ax.bar(x - w / 2, target, w, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=3, label="target")
    ax.bar(x + w / 2, got, w, color=CLAY_D, zorder=3, label="achieved")
    for xi, t, g in zip(x, target, got):
        ax.text(xi - w / 2, t + 2, f"{t}%", ha="center", fontfamily=MONO, fontsize=10.5, color=MUTED)
        ax.text(xi + w / 2, g + 2, f"{g}%", ha="center", fontfamily=MONO, fontsize=10.5, color=CLAY_D)
    ax.set_xticks(x, ["50% interval", "80% interval", "90% interval"])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    style(ax)
    heading(fig, "The intervals mean what they say", "Result · calibration · SPY", y=0.84)
    legend(fig, ax)
    frame(fig)
    fig.savefig(OUT / "calibration.svg")
    plt.close(fig)


# ---- news fusion --------------------------------------------------------------------------------
def fusion():
    labels = ["Kronos alone\n(uncorrected)", "+ real\nheadlines", "+ shuffled\nheadlines"]
    vals = [0.0468, 0.0415, 0.0414]
    cols = [IVORY, EUC_D, CLAY]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.subplots_adjust(top=0.72)
    x = np.arange(3)
    ax.bar(x, vals, 0.56, color=cols, edgecolor=INK, linewidth=1, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.001, f"{v:.4f}", ha="center", fontsize=17, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.053)
    ax.set_ylabel("CRPS  ·  lower is better", fontfamily=MONO, fontsize=10)
    style(ax)
    heading(fig, "The gain was never the news", "Result · news fusion · SPY", y=0.84)
    frame(fig)
    fig.savefig(OUT / "news_fusion.svg")
    plt.close(fig)


# ---- false positives ----------------------------------------------------------------------------
def false_positives():
    modes = ["Volatility-scaled\nranges", "Volatility-scaled,\nσ partialled out", "Fixed\nranges", "Direction\nlabels"]
    real = np.array([95.0, 3.5, 4.5, 3.0])
    shuf = np.array([93.5, 10.5, 8.0, 4.0])

    def ci(p):
        return 1.96 * np.sqrt(p / 100 * (1 - p / 100) / 200) * 100

    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    fig.subplots_adjust(top=0.74)
    x, w = np.arange(4), 0.36
    ax.bar(x - w / 2, real, w, color=CLAY_D, zorder=3, label="real headlines")
    ax.bar(x + w / 2, shuf, w, color=EUC, edgecolor=EUC_D, linewidth=0.8, zorder=3, label="shuffled headlines")
    for xs, r in ((x - w / 2, real), (x + w / 2, shuf)):
        ax.errorbar(xs, r, yerr=ci(r), fmt="none", ecolor=INK, elinewidth=0.9, capsize=3, zorder=4)
        for xi, ri in zip(xs, r):
            ax.text(xi, ri + ci(ri) + 2, f"{ri:g}%", ha="center", fontfamily=MONO, fontsize=10, color=INK)
    ax.axhline(5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
    ax.text(3.62, 14, "nominal 5%", fontfamily=MONO, fontsize=9.5, color=MUTED, ha="right")
    ax.set_xticks(x, modes)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("panels rejecting (%)", fontfamily=MONO, fontsize=10)
    style(ax)
    heading(fig, "Ask the wrong way and get a signal from nothing", "Result · 200 simulated panels, zero information", y=0.86)
    legend(fig, ax)
    frame(fig)
    fig.savefig(OUT / "false_positive_rates.svg")
    plt.close(fig)


for f in (hero, pipeline, reversion, crps, calibration, fusion, false_positives):
    f()
print("wrote", sorted(p.name for p in OUT.glob("*.svg")), "| serif:", SERIF, "| mono:", MONO)
