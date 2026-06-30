"""
Figure 1 — General Methodological Architecture (v2).
Three-layer diagram with no internal title/legend. Caption goes in LaTeX only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_FILE = BASE_DIR / "FIGURES" / "methodological_architecture.png"

# ── colour palette ─────────────────────────────────────────────────────────
C_BLUE   = "#2C6E9B"   # layer 1 header
C_BLUE_L = "#D6E8F5"   # layer 1 body
C_GREEN  = "#2A7A4B"   # layer 2 header
C_GREEN_L= "#D5EDE0"   # layer 2 body
C_ORANGE = "#B85C1A"   # layer 3 header
C_ORANGE_L="#FAE3CE"   # layer 3 body
C_BOX    = "white"
C_BORDER = "#555555"

FONT_TITLE = dict(fontsize=10, fontweight="bold", color="white")
FONT_ITEM  = dict(fontsize=8.5, color="#1a1a1a", va="center", ha="center")
FONT_ARROW = dict(fontsize=7.5, color="#444444", style="italic")

# ── helpers ────────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, facecolor, edgecolor=C_BORDER, lw=0.8, radius=0.03):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(box)
    return box

def header(ax, x, y, w, h, color, text):
    rounded_box(ax, x, y, w, h, facecolor=color, edgecolor=color, lw=0)
    ax.text(x + w / 2, y + h / 2, text, **FONT_TITLE, ha="center", va="center")

def item_box(ax, cx, cy, w, h, text):
    rounded_box(ax, cx - w/2, cy - h/2, w, h, facecolor=C_BOX, edgecolor=C_BORDER, lw=0.7)
    ax.text(cx, cy, text, **FONT_ITEM, wrap=True)

def layer_background(ax, x, y, w, h, color):
    bg = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        facecolor=color,
        edgecolor=C_BORDER,
        linewidth=0.9,
        zorder=0,
    )
    ax.add_patch(bg)

def arrow_down(ax, x, y1, y2):
    ax.annotate(
        "", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=C_BORDER, lw=1.0),
        annotation_clip=False,
    )

# ── layout constants ───────────────────────────────────────────────────────
XMIN, XMAX = 0.0, 10.0
FIG_W, FIG_H = 13, 9

LAYER1_Y  = 6.30
LAYER2_Y  = 3.65
LAYER3_Y  = 0.90
LAYER_H   = 2.35
HEAD_H    = 0.38
PAD       = 0.22

ITEM_H    = 0.62
ITEM_W    = 1.72
ITEM_GAP  = 0.22

LAYER_W   = XMAX - XMIN - 0.30
LX        = XMIN + 0.15   # left edge of layers

def layer_items_y(layer_y):
    return layer_y + HEAD_H + PAD + ITEM_H / 2

# ── build figure ───────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(0.5, 9.3)
ax.axis("off")

# ── LAYER 1: Historical and Data Construction ──────────────────────────────
layer_background(ax, LX, LAYER1_Y, LAYER_W, LAYER_H, C_BLUE_L)
header(ax, LX, LAYER1_Y + LAYER_H - HEAD_H, LAYER_W, HEAD_H, C_BLUE,
       "Layer 1 — Historical and Data Construction")

items1 = [
    "Event Chronology\nConstruction",
    "Regime\nSeparation\n(Core / Pandemic / Post)",
    "Financial Data\nIntegration\n(S&P 500, Nasdaq,\nShanghai, VIX)",
    "Feature\nEngineering\n(returns, log-returns)",
    "Event Window\nConstruction\n(−1/+1, −3/+3, −5/+5)",
]
n1 = len(items1)
total_w1 = n1 * ITEM_W + (n1 - 1) * ITEM_GAP
x_start1 = LX + (LAYER_W - total_w1) / 2
iy1 = layer_items_y(LAYER1_Y)
for i, txt in enumerate(items1):
    cx = x_start1 + i * (ITEM_W + ITEM_GAP) + ITEM_W / 2
    item_box(ax, cx, iy1, ITEM_W, ITEM_H * 1.35, txt)

# ── LAYER 2: Event Study and Statistical Inference ─────────────────────────
layer_background(ax, LX, LAYER2_Y, LAYER_W, LAYER_H, C_GREEN_L)
header(ax, LX, LAYER2_Y + LAYER_H - HEAD_H, LAYER_W, HEAD_H, C_GREEN,
       "Layer 2 — Event Study and Statistical Inference")

items2 = [
    "Expected Return\nEstimation\n(mean-adjusted\n[−30, −6])",
    "Abnormal Returns\n(AR = R − E[R])",
    "Cumulative\nAbnormal Returns\n(CAR)",
    "Volatility &\nSignal Classification\n(local σ; ±0.50% zones)",
    "Statistical Testing\n(t-test, Wilcoxon,\nWelch, MWU,\nPearson, Spearman)",
]
n2 = len(items2)
total_w2 = n2 * ITEM_W + (n2 - 1) * ITEM_GAP
x_start2 = LX + (LAYER_W - total_w2) / 2
iy2 = layer_items_y(LAYER2_Y)
for i, txt in enumerate(items2):
    cx = x_start2 + i * (ITEM_W + ITEM_GAP) + ITEM_W / 2
    item_box(ax, cx, iy2, ITEM_W, ITEM_H * 1.35, txt)

# ── LAYER 3: Robustness, ML, and Interpretation ────────────────────────────
layer_background(ax, LX, LAYER3_Y, LAYER_W, LAYER_H, C_ORANGE_L)
header(ax, LX, LAYER3_Y + LAYER_H - HEAD_H, LAYER_W, HEAD_H, C_ORANGE,
       "Layer 3 — Robustness, Machine Learning, and Interpretation")

items3 = [
    "Robustness &\nSensitivity Analysis\n(windows, thresholds,\nsample composition)",
    "Placebo\nValidation\n(non-event dates)",
    "Robust Estimators\n(median, trimmed\n& winsorized mean,\nleave-one-out)",
    "Exploratory\nMachine Learning\n(Logistic Reg.,\nRandom Forest)",
    "Validation &\nReproducibility\n(TDD-inspired;\nfixed seed = 42)",
]
n3 = len(items3)
total_w3 = n3 * ITEM_W + (n3 - 1) * ITEM_GAP
x_start3 = LX + (LAYER_W - total_w3) / 2
iy3 = layer_items_y(LAYER3_Y)
for i, txt in enumerate(items3):
    cx = x_start3 + i * (ITEM_W + ITEM_GAP) + ITEM_W / 2
    item_box(ax, cx, iy3, ITEM_W, ITEM_H * 1.35, txt)

# ── arrows between layers ──────────────────────────────────────────────────
mid_x = LX + LAYER_W / 2
arrow_down(ax, mid_x, LAYER1_Y, LAYER2_Y + LAYER_H + 0.01)
arrow_down(ax, mid_x, LAYER2_Y, LAYER3_Y + LAYER_H + 0.01)

ax.text(mid_x + 0.25, (LAYER1_Y + LAYER2_Y + LAYER_H) / 2, "feeds into",
        **FONT_ARROW, ha="left")
ax.text(mid_x + 0.25, (LAYER2_Y + LAYER3_Y + LAYER_H) / 2, "feeds into",
        **FONT_ARROW, ha="left")

plt.tight_layout(pad=0.3)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT_FILE}")
