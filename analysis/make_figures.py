"""
Regenerates every figure in ../figures from numbers transcribed from the
QuantConnect run logs. Each constant is tagged with the run that printed it,
so any value can be checked against the raw log in ../results/.

    python analysis/make_figures.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# ---- chart chrome (light surface) -------------------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"   # validated: CVD dE >= 21
NET = "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 26,
})

def subtitle(ax, text):
    ax.text(0, 1.035, text, transform=ax.transAxes, color=INK2, fontsize=10.5, va="bottom")

def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)

R_PCT, ADV_ASSUMED = 0.20, 0.025      # R := 0.20 x ATR ; adverse charge = 0.025 x ATR

# =============================================================================
# FIG 1 -- net R under every adverse-selection assumption (exact re-pricing)
#   net(a) = net(0.025) + (0.025 - a) / 0.20       (same slope for every run)
# =============================================================================
NET_IS  = +0.1514    # Jumping Orange Frog, IS 2016-19, measured fills
NET_OOS = -0.0855    # Calculating Tan Alligator, OOS 2020-Jun 2026, v8, measured fills
be = lambda n: ADV_ASSUMED + R_PCT * n

fig, ax = plt.subplots(figsize=(9.5, 5.4))
x = np.linspace(0, 0.075, 200)
for n, col, lab in ((NET_IS, BLUE, "In-sample 2016–19"), (NET_OOS, ORANGE, "Out-of-sample 2020–26 (pre-registered)")):
    y = n + (ADV_ASSUMED - x) / R_PCT
    ax.plot(x, y, color=col, lw=2, solid_capstyle="round", label=lab)
    b = be(n)
    ax.plot([b], [0], "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=5)
    if n > 0:
        ax.annotate(f"breakeven {b:.4f} × ATR", (b, 0), xytext=(6, 10),
                    textcoords="offset points", color=INK, fontsize=9.5)
    else:
        ax.annotate(f"breakeven {b:.4f} × ATR", (b, 0), xytext=(0.004, 0.075),
                    textcoords="data", color=INK, fontsize=9.5,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=6))
    ax.plot([ADV_ASSUMED], [n], "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=5)
    ax.annotate(f"result as run: {n:+.4f}R", (ADV_ASSUMED, n), xytext=(8, 4),
                textcoords="offset points", color=INK2, fontsize=9.5)
ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
for v, txt in ((0.025, "assumed in every run"), (0.050, "'conservative' preset")):
    ax.axvline(v, color=AXIS, lw=1, zorder=1)
    ax.text(v + 0.0006, 0.288, txt, va="top", ha="left", color=MUTED, fontsize=9)
ax.set_xlim(0, 0.075); ax.set_ylim(-0.40, 0.30)
ax.set_xlabel("Assumed adverse selection on the breakout fill (× daily ATR)")
ax.set_ylabel("Net R per trade")
ax.set_title("The edge lives inside an assumption nobody had measured")
subtitle(ax, "Net R per trade, re-priced exactly at every adverse-selection level. "
             "Breakeven fell from 0.055×ATR in-sample to 0.008×ATR out-of-sample.")
ax.legend(loc="lower right", frameon=False, labelcolor=INK)
save(fig, "fig1_breakeven.png")

# =============================================================================
# FIG 2 -- where the gross edge went (per-trade decomposition, in R)
# =============================================================================
DECOMP = {
    # Upgraded Fluorescent Orange Pelican, IS 2016-19, u500, N=5, measured.
    # Reconstructed from its friction audit; round trip $0.1635 vs logged $0.1634.
    "In-sample 2016–19 (3,370 trades)": [0.206, -0.058, -0.125, -0.020],
    # Calculating Tan Alligator, OOS, v8 -- printed directly; SELF-CHECK AGREE.
    "Out-of-sample 2020–26 (5,294 trades)": [0.0783, -0.0273, -0.1250, -0.0114],
}
steps = ["True gross\nedge", "Spread", "Adverse\nselection", "Commission\n+ borrow", "Net"]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
for ax, (title, (g, s, a, c)) in zip(axes, DECOMP.items()):
    net = g + s + a + c
    level, bottoms, heights, cols = g, [0], [g], [BLUE]
    for d in (s, a, c):
        bottoms.append(level + d); heights.append(-d); cols.append(RED); level += d
    bottoms.append(min(0, net)); heights.append(abs(net)); cols.append(NET)
    xs = np.arange(5)
    ax.bar(xs, heights, bottom=bottoms, width=0.55, color=cols, edgecolor=SURFACE, linewidth=2)
    vals = [g, s, a, c, net]
    for i, v in enumerate(vals):
        top = bottoms[i] + heights[i]
        yv = top if (v >= 0 and i != 4) or (i == 4 and net >= 0) else bottoms[i]
        off = 0.008 if yv == top else -0.008
        ax.text(i, yv + off, f"{v:+.3f}", ha="center", va="bottom" if off > 0 else "top",
                color=INK, fontsize=9.5)
    ax.axhline(0, color=AXIS, lw=1.2)
    ax.set_xticks(xs, steps, color=INK2, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, fontsize=11.5, pad=8)
axes[0].set_ylabel("R per trade")
axes[0].set_ylim(-0.12, 0.25)
fig.suptitle("Where the gross edge went", x=0.065, y=1.115, ha="left", fontsize=14,
             fontweight="bold", color=INK)
fig.text(0.065, 1.005, "Per-trade decomposition under the measured fill model. Adverse selection is "
         "−0.125R in both periods, by construction: it is 0.025×ATR on an R of 0.20×ATR.",
         ha="left", color=INK2, fontsize=10.5)
save(fig, "fig2_decomposition.png")

# =============================================================================
# FIG 3 -- the one filter variable that 'survived' in-sample, out-of-sample
# =============================================================================
BUCKETS = ["0.8–1.2%", "1.2–2.0%", "2.0%+"]
IS_R,  IS_N  = [0.1135, 0.1940, 0.8787], [253, 75, 20]          # Pelican, IS, measured
OOS_R, OOS_N = [-0.0881, -0.0501, -0.2411], [3419, 1496, 354]   # Alligator, OOS, measured
fig, ax = plt.subplots(figsize=(11.5, 5.0))
xs, w = np.arange(3), 0.30
b1 = ax.bar(xs - w/2 - 0.01, IS_R, w, color=BLUE, label="In-sample 2016–19", edgecolor=SURFACE, lw=2)
b2 = ax.bar(xs + w/2 + 0.01, OOS_R, w, color=ORANGE, label="Out-of-sample 2020–26", edgecolor=SURFACE, lw=2)
for bars, vals, ns in ((b1, IS_R, IS_N), (b2, OOS_R, OOS_N)):
    for b, v, n in zip(bars, vals, ns):
        y = b.get_height() if v >= 0 else min(b.get_y(), b.get_y() + b.get_height())
        ax.text(b.get_x() + b.get_width()/2, y + (0.02 if v >= 0 else -0.02),
                f"{v:+.3f}\nn={n:,}", ha="center", va="bottom" if v >= 0 else "top",
                color=INK, fontsize=9)
ax.axhline(0, color=AXIS, lw=1.2)
ax.set_xticks(xs, BUCKETS, color=INK2)
ax.set_xlabel("Stop distance relative to price (R / price)")
ax.set_ylabel("Net R per trade")
ax.set_ylim(-0.42, 1.0)
ax.grid(axis="x", visible=False)
ax.set_title("The filter that sorted in-sample flipped sign out-of-sample")
subtitle(ax, "Net R by R/price bucket, measured fills. The 0.008 threshold was chosen on in-sample data; "
             "out-of-sample, all three buckets above it went negative.")
ax.legend(loc="upper left", frameon=False, labelcolor=INK)
save(fig, "fig3_filter_inversion.png")

# =============================================================================
# FIG 4 -- annual returns with FREE (2c/side) fills, 2016 - Jun 2026
#   Creative Apricot Chicken: v5, paper fills, REL_R_MIN 0.008, u500, N=5
# =============================================================================
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
RET   = [54.8, 23.6, 56.2, 119.7, 52.1, 385.0, 30.0, 17.1, -89.5, -51.1, 114.1]
fig, ax = plt.subplots(figsize=(10, 5.0))
ax.axvspan(-0.5, 3.5, color="#f0efec", zorder=0, lw=0)
ax.text(1.5, 405, "in-sample window", ha="center", color=MUTED, fontsize=9.5)
cols = [BLUE if r >= 0 else RED for r in RET]
ax.bar(range(len(YEARS)), RET, width=0.55, color=cols, linewidth=0, zorder=2)
for i in (5, 8, 9):
    r = RET[i]
    ax.text(i, r + (8 if r >= 0 else -8), f"{r:+.1f}%", ha="center",
            va="bottom" if r >= 0 else "top", color=INK, fontsize=10, fontweight="bold")
ax.axhline(0, color=AXIS, lw=1.2, zorder=3)
ax.set_xticks(range(len(YEARS)), [str(y) for y in YEARS[:-1]] + ["2026*"], color=INK2)
ax.set_ylabel("Annual return")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylim(-130, 440)
ax.grid(axis="x", visible=False)
ax.set_title("Even with cheap fills, 2024–25 lost 95% of the account")
subtitle(ax, "Same parameters as the pre-registered test (earlier v5 build), 2¢/side, zero adverse selection. "
             "Peak $9.1M (Feb 2023), max DD 96.7%.")
ax.text(0, -0.13, "* 2026 = January–June.", transform=ax.transAxes, color=MUTED, fontsize=9)
save(fig, "fig4_free_fills_by_year.png")
