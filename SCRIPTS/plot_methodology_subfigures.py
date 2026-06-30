"""
Methodology subfigures A through I for the methodology chapter.
All figures are clean, academic, no internal titles (captions go in LaTeX).
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "FIGURES"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── shared style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_BLUE   = "#2C6E9B"
C_GREEN  = "#2A7A4B"
C_ORANGE = "#B85C1A"
C_GREY   = "#666666"
C_LIGHT  = "#F5F5F5"
C_RED    = "#B22222"

def rbox(ax, x, y, w, h, fc, ec="#333333", lw=0.8, pad=0.02, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={pad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)

def arr(ax, x0, y0, x1, y1, color="#333333", lw=1.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                annotation_clip=False)

# ══════════════════════════════════════════════════════════════════════════
# A — Event Chronology Construction
# ══════════════════════════════════════════════════════════════════════════

def fig_A():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    boxes = [
        (0.2,  1.7, 1.8, 1.1, "#D6E8F5", C_BLUE,   "67 Documented\nTariff-War Events\n(Master Chronology)"),
        (2.4,  0.5, 1.8, 1.1, "#D5EDE0", C_GREEN,  "Core Phase\n2018 – early 2020\n(17 events)"),
        (2.4,  1.8, 1.8, 1.1, "#FFF3CD", "#B8860B", "Pandemic Period\n2020 – 2023\n(19 events)"),
        (2.4,  3.1, 1.8, 1.1, "#FAE3CE", C_ORANGE, "Post-Pandemic\n2024 – 2026\n(31 events)"),
        (4.6,  1.1, 2.0, 1.1, "#EAF4EA", C_GREEN,  "Market Coverage\nCheck\n(S&P 500 anchor;\nestimation window)"),
        (7.0,  1.1, 2.5, 1.1, "#D5EDE0", C_GREEN,  "24 Market-Covered\nEvents\n(Expanded Sample\nfor Inference)"),
        (4.6,  3.0, 2.0, 1.1, "#F2F2F2", C_GREY,   "No Market Coverage\n→ Historical Record\nOnly"),
    ]

    for (x, y, w, h, fc, ec, txt) in boxes:
        rbox(ax, x, y, w, h, fc, ec)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=7.5, color="#1a1a1a")

    # arrows
    arr(ax, 2.0, 2.25, 2.4, 3.65)
    arr(ax, 2.0, 2.25, 2.4, 2.35)
    arr(ax, 2.0, 2.25, 2.4, 1.05)
    arr(ax, 4.2, 1.65, 4.6, 1.65)
    arr(ax, 4.2, 3.55, 4.6, 3.55)
    arr(ax, 6.6, 1.65, 7.0, 1.65)

    # classification label
    ax.text(3.3, 4.25, "Regime Classification", ha="center",
            fontsize=8, color=C_GREY, style="italic")

    # event types legend
    ax.text(0.22, 0.25,
            "Event types: Escalation · Relief · Structural · Retaliation",
            fontsize=7, color=C_GREY)

    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "fig_chronology_construction.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_chronology_construction.png")


# ══════════════════════════════════════════════════════════════════════════
# B — Event Window Construction
# ══════════════════════════════════════════════════════════════════════════

def fig_B():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(-35, 10)
    ax.set_ylim(-1.2, 3.8)
    ax.axis("off")

    # horizontal axis
    ax.axhline(0, color="#333333", lw=1.2, xmin=0.01, xmax=0.99)

    # tick marks
    for x in range(-34, 9):
        ax.plot([x, x], [-0.12, 0.12], color="#888888", lw=0.6)

    # key positions
    for xp, lbl in [(-30, "−30"), (-6, "−6"), (0, "0\n(anchor)"),
                    (-1, "−1"), (1, "+1"), (-3, "−3"), (3, "+3"),
                    (-5, "−5"), (5, "+5")]:
        ax.plot([xp, xp], [-0.2, 0.2], color=C_BLUE if xp == 0 else "#555555", lw=1.0)
        ax.text(xp, -0.5, lbl, ha="center", va="top", fontsize=7.5,
                color=C_BLUE if xp == 0 else "#333333",
                fontweight="bold" if xp == 0 else "normal")

    # estimation window band
    rbox(ax, -30, 0.3, 24, 0.55, "#D6E8F5", C_BLUE, pad=0.01)
    ax.text(-18, 0.57, "Estimation Window  [−30, −6]  (trading days)",
            ha="center", va="center", fontsize=8, color=C_BLUE, fontweight="bold")

    # event windows
    window_specs = [
        (-1, 1, 1.2, "#D5EDE0", C_GREEN,  "[−1, +1]"),
        (-3, 3, 1.8, "#FFF3CD", "#B8860B", "[−3, +3]"),
        (-5, 5, 2.4, "#FAE3CE", C_ORANGE, "[−5, +5]"),
    ]
    for (xl, xr, yy, fc, ec, lbl) in window_specs:
        rbox(ax, xl, yy - 0.22, xr - xl, 0.44, fc, ec, pad=0.01)
        ax.text((xl + xr) / 2, yy, lbl, ha="center", va="center",
                fontsize=8, color=ec, fontweight="bold")

    # anchor marker
    ax.plot([0], [0], "v", color=C_BLUE, ms=9, zorder=5)

    ax.text(-17, -1.1, "← Trading days (not calendar days) →",
            ha="center", fontsize=7.5, color=C_GREY, style="italic")

    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "fig_event_window_construction.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_event_window_construction.png")


# ══════════════════════════════════════════════════════════════════════════
# C — Abnormal Returns
# ══════════════════════════════════════════════════════════════════════════

def fig_C():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    boxes = [
        (0.3, 1.35, 2.0, 1.1, "#D6E8F5", C_BLUE,   "Observed\nReturn\n$R_{it}$"),
        (3.5, 1.35, 2.0, 1.1, "#FAE3CE", C_ORANGE, "Expected\nReturn\n$E(R_i)$\n(pre-event mean)"),
        (6.7, 1.35, 2.0, 1.1, "#D5EDE0", C_GREEN,  "Abnormal\nReturn\n$AR_{it}$"),
    ]
    for (x, y, w, h, fc, ec, txt) in boxes:
        rbox(ax, x, y, w, h, fc, ec)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=9, color="#1a1a1a")

    # minus sign
    ax.text(2.55, 1.90, "−", ha="center", va="center", fontsize=22, color="#333333")
    # equals sign
    ax.text(5.80, 1.90, "=", ha="center", va="center", fontsize=22, color="#333333")

    ax.text(4.5, 0.5,
            "Deviation of observed return from the pre-event baseline within the event window.",
            ha="center", fontsize=7.5, color=C_GREY, style="italic")
    ax.text(4.5, 0.2,
            "Interpreted as market movement associated with the event window — not as causal effect.",
            ha="center", fontsize=7.5, color=C_GREY, style="italic")

    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "fig_abnormal_returns.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_abnormal_returns.png")


# ══════════════════════════════════════════════════════════════════════════
# D — Cumulative Abnormal Returns
# ══════════════════════════════════════════════════════════════════════════

def fig_D():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                              gridspec_kw={"width_ratios": [1.6, 1]})

    # --- left: bar chart of daily ARs ---
    ax = axes[0]
    days = [-1, 0, 1]
    ars = [-0.18, -0.42, 0.10]
    colors = [C_RED if v < 0 else C_GREEN for v in ars]
    bars = ax.bar(days, ars, color=colors, width=0.6, edgecolor="#333333", lw=0.7)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(days)
    ax.set_xticklabels(["Day −1", "Day 0\n(anchor)", "Day +1"], fontsize=8)
    ax.set_ylabel("Abnormal Return (%)", fontsize=8)
    ax.set_title("Daily Abnormal Returns within [−1, +1] window\n(illustrative example)",
                 fontsize=8, color=C_GREY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, ars):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (0.015 if val >= 0 else -0.025),
                f"{val:+.2f}%", ha="center", fontsize=7,
                color=C_GREEN if val >= 0 else C_RED)

    # --- right: CAR formula ---
    ax2 = axes[1]
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 4)
    ax2.axis("off")
    rbox(ax2, 0.3, 2.0, 4.2, 1.5, "#D5EDE0", C_GREEN)
    car_val = sum(ars)
    ax2.text(2.4, 2.75, "CAR[−1, +1]", ha="center", fontsize=10,
             fontweight="bold", color=C_GREEN)
    ax2.text(2.4, 2.35, f"= $AR_{{-1}}$ + $AR_0$ + $AR_{{+1}}$",
             ha="center", fontsize=9, color="#1a1a1a")
    ax2.text(2.4, 2.05,
             f"= ({ars[0]:+.2f}%) + ({ars[1]:+.2f}%) + ({ars[2]:+.2f}%)",
             ha="center", fontsize=8.5, color="#1a1a1a")
    ax2.text(2.4, 1.68, f"= {car_val:+.2f}%",
             ha="center", fontsize=11, fontweight="bold",
             color=C_RED if car_val < 0 else C_GREEN)
    ax2.text(2.4, 0.7,
             "Sums abnormal returns\nover the event window\n(negative = below baseline)",
             ha="center", fontsize=7.5, color=C_GREY, style="italic")

    plt.tight_layout(pad=0.5)
    plt.savefig(FIG_DIR / "fig_cumulative_abnormal_returns.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_cumulative_abnormal_returns.png")


# ══════════════════════════════════════════════════════════════════════════
# E — Volatility and Signal Classification
# ══════════════════════════════════════════════════════════════════════════

def fig_E():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    # axis line
    ax.axhline(1.75, color="#333333", lw=1.5, xmin=0.02, xmax=0.98)

    # zones  -- NEUTRAL alargada: de [-0.5, 0.5] para [-0.85, 0.85]
    zone_specs = [
        (-3.4, -0.85, "#FDDADA", C_RED,   "NEGATIVE\nCAR < −0.50%"),
        (-0.85, 0.85, "#F5F5F5", C_GREY,  "NEUTRAL\n−0.50% ≤ CAR ≤ +0.50%"),
        ( 0.85, 3.4,  "#D5EDE0", C_GREEN, "POSITIVE\nCAR > +0.50%"),
    ]
    for (xl, xr, fc, ec, lbl) in zone_specs:
        rbox(ax, xl, 0.7, xr - xl, 2.1, fc, ec, pad=0.01)
        ax.text((xl + xr) / 2, 1.75, lbl, ha="center", va="center",
                fontsize=9, color=ec, fontweight="bold")

    # threshold markers  -- linhas e rotulos acompanham as novas fronteiras (+-0.85)
    for xp, lbl in [(-0.85, "−0.50%"), (0.85, "+0.50%")]:
        ax.plot([xp, xp], [0.68, 2.82], color="#555555", lw=1.2, ls="--")
        ax.text(xp, 0.45, lbl, ha="center", fontsize=8, color="#555555")

    # alternative thresholds note
    ax.text(0, 0.15,
            "Sensitivity analysis also tested: ±0.25% and ±0.75% thresholds",
            ha="center", fontsize=7.5, color=C_GREY, style="italic")

    ax.text(0, 3.3, "CAR value →",
            ha="center", fontsize=8, color=C_GREY)

    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "fig_volatility_signal_classification.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_volatility_signal_classification.png")


# ══════════════════════════════════════════════════════════════════════════
# F — Escalation and Relief Event Groups
# ══════════════════════════════════════════════════════════════════════════

def fig_F():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # header box
    rbox(ax, 3.5, 3.4, 4.0, 0.85, "#D6E8F5", C_BLUE, pad=0.02)
    ax.text(5.5, 3.82, "Tariff-War Event", ha="center", fontsize=10,
            fontweight="bold", color=C_BLUE)

    groups = [
        (0.2,  1.4, 3.0, 1.7, "#FDDADA", C_RED,   "ESCALATION (N=13)",
         "· Tariff announcements\n· Tariff implementations\n· Tariff increases\n· Retaliatory actions\n· Actions intensifying trade tensions"),
        (3.9,  1.4, 3.0, 1.7, "#F5F5F5", C_GREY, "STRUCTURAL (N=1)",
         "· Framework changes\n· Policy modifications not\n  fitting escalation or relief\n· Reported for completeness\n· Not interpreted statistically"),
        (7.6,  1.4, 3.0, 1.7, "#D5EDE0", C_GREEN, "RELIEF (N=10)",
         "· Tariff suspensions\n· Tariff reductions\n· Trade agreements\n· Negotiation progress\n· Signals easing tensions"),
    ]

    for (x, y, w, h, fc, ec, header_txt, items_txt) in groups:
        rbox(ax, x, y, w, h, fc, ec, pad=0.02)
        ax.text(x + w/2, y + h - 0.25, header_txt, ha="center", va="top",
                fontsize=8.5, fontweight="bold", color=ec)
        ax.text(x + 0.15, y + h - 0.55, items_txt, ha="left", va="top",
                fontsize=7.5, color="#1a1a1a")

    # arrows from header to groups
    for xc in [1.7, 5.4, 9.1]:
        arr(ax, 5.5, 3.4, xc, 3.1)

    ax.text(5.5, 0.25,
            "Classification is analytical, not a welfare or moral judgment.\n"
            "Goal: compare short-window market reactions across policy signal directions.",
            ha="center", fontsize=7.5, color=C_GREY, style="italic")

    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "fig_escalation_relief_groups.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_escalation_relief_groups.png")


# ══════════════════════════════════════════════════════════════════════════
# G — Statistical Testing
# ══════════════════════════════════════════════════════════════════════════

def fig_G():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    rows = [
        ("Does CAR differ from zero?",
         "One-sample t-test; Wilcoxon signed-rank;\nBootstrap 95% CI (5,000 resamples, seed 42)"),
        ("Do escalation and relief groups differ?",
         "Welch t-test (unequal variances);\nMann–Whitney U (non-parametric, two-sided)"),
        ("Is CAR associated with local volatility?",
         "Pearson correlation (linear);\nSpearman correlation (rank-based monotonic)"),
        ("Are central tendency estimates robust to outliers?",
         "Median, 10% trimmed mean, 10% winsorized mean;\nSign test; Leave-one-out influence analysis"),
    ]

    y_start = 4.5
    row_h   = 0.82
    col_q   = 0.2   # left column x
    col_t   = 5.2   # right column x
    col_w_q = 4.8
    col_w_t = 5.6
    head_h  = 0.42

    # header
    rbox(ax, col_q, y_start, col_w_q, head_h, C_BLUE, C_BLUE, lw=0)
    ax.text(col_q + col_w_q/2, y_start + head_h/2,
            "Research Question", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    rbox(ax, col_t, y_start, col_w_t, head_h, C_BLUE, C_BLUE, lw=0)
    ax.text(col_t + col_w_t/2, y_start + head_h/2,
            "Statistical Test(s)", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

    for i, (q, t) in enumerate(rows):
        yy = y_start - head_h - (i + 1) * row_h
        fc = "#F5F5F5" if i % 2 == 0 else "white"
        rbox(ax, col_q, yy, col_w_q, row_h, fc, "#CCCCCC", lw=0.5)
        rbox(ax, col_t, yy, col_w_t, row_h, fc, "#CCCCCC", lw=0.5)
        ax.text(col_q + 0.12, yy + row_h/2, q, ha="left", va="center",
                fontsize=8.5, color="#1a1a1a")
        ax.text(col_t + 0.12, yy + row_h/2, t, ha="left", va="center",
                fontsize=8.0, color="#1a1a1a")

    ax.text(5.5, 0.15,
            "All tests interpreted with caution. Statistical significance alone does not imply causality.",
            ha="center", fontsize=7.5, color=C_GREY, style="italic")

    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "fig_statistical_testing.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_statistical_testing.png")


# ══════════════════════════════════════════════════════════════════════════
# H1 — ML Flow
# ══════════════════════════════════════════════════════════════════════════

def fig_H1():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    steps = [
        (0.2,  1.0, 2.0, 1.5, "#D6E8F5", C_BLUE,   "Event-Window\nFeatures\n(24 events × 3 windows\n= 72 observations)"),
        (2.8,  1.0, 2.0, 1.5, "#EAF4EA", C_GREEN,  "Feature\nPreprocessing\n(exclude formal_car;\nstandardize)"),
        (5.4,  1.0, 2.0, 1.5, "#FFF3CD", "#B8860B", "Classification\nModels\nLogistic Reg.\nRandom Forest"),
        (8.0,  1.0, 2.0, 1.5, "#FAE3CE", C_ORANGE, "Predicted\nDirection\n(positive /\nnegative)"),
        (10.0, 1.0, 1.7, 1.5, "#F2F2F2", C_GREY,   "vs.\nMajority\nBaseline\n(55.56%)"),
    ]

    for (x, y, w, h, fc, ec, txt) in steps:
        rbox(ax, x, y, w, h, fc, ec)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=8, color="#1a1a1a")

    for x0, x1 in [(2.2, 2.8), (4.8, 5.4), (7.4, 8.0), (10.0, 10.0)]:
        if x0 < x1:
            arr(ax, x0, 1.75, x1, 1.75)

    ax.text(6.0, 0.25,
            "Models evaluated by accuracy and F1-score against majority baseline. "
            "Results interpreted as exploratory — not as primary evidence.",
            ha="center", fontsize=7.5, color=C_GREY, style="italic")

    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "fig_ml_flow.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_ml_flow.png")


# ══════════════════════════════════════════════════════════════════════════
# H2 — ML Leakage Prevention
# ══════════════════════════════════════════════════════════════════════════

def fig_H2():
    fig, ax = plt.subplots(figsize=(9, 4.2))   # um pouco mais alto
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.6, 3.8)                      # espaço extra embaixo p/ o caption
    ax.axis("off")

    # full feature set
    rbox(ax, 0.2, 0.7, 3.6, 2.4, "#D6E8F5", C_BLUE, pad=0.02)
    ax.text(2.0, 2.75, "Full Event-Window\nFeature Set", ha="center",
            fontsize=9, fontweight="bold", color=C_BLUE)
    features_in = ["car_sign (target)", "sp500_return", "nasdaq_return",
                   "vix", "volatility", "window_size"]
    for i, f in enumerate(features_in):
        ax.text(0.4, 2.35 - i * 0.26, f"·  {f}", fontsize=7.5, color="#1a1a1a")

    # excluded box  (usando features_out, que estava sem uso)
    features_out = ["formal_car_sp500"]
    rbox(ax, 0.2, 0.15, 3.6, 0.52, "#FDDADA", C_RED, lw=1.2)
    ax.text(2.0, 0.41, f"[X]  {features_out[0]}  (EXCLUDED -- leakage risk)",
            ha="center", fontsize=7.5, color=C_RED, fontweight="bold")

    arr(ax, 3.8, 1.9, 5.0, 1.9)

    # model input
    rbox(ax, 5.0, 0.7, 3.5, 2.4, "#D5EDE0", C_GREEN, pad=0.02)
    ax.text(6.75, 2.75, "Allowed Feature Set\nfor Model Training", ha="center",
            fontsize=9, fontweight="bold", color=C_GREEN)
    for i, f in enumerate(features_in[1:]):
        ax.text(5.2, 2.35 - i * 0.26, f"[ok] {f}", fontsize=7.5, color=C_GREEN)

    # caption: agora abaixo da caixa vermelha, ancorado pelo topo
    ax.text(4.5, -0.05,
            "formal_car_sp500 excluded: it is directly derived from the target (CAR direction)\n"
            "and would constitute outcome leakage if used as a predictor.",
            ha="center", va="top", fontsize=7.5, color=C_GREY, style="italic")

    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "fig_ml_leakage.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_ml_leakage.png")


# ══════════════════════════════════════════════════════════════════════════
# I — Validation Pipeline
# ══════════════════════════════════════════════════════════════════════════

def fig_I():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    steps = [
        (0.2,  1.0, 2.0, 1.5, "#D6E8F5", C_BLUE,   "Build\nScript\nbuild_*.py"),
        (2.8,  1.0, 2.0, 1.5, "#EAF4EA", C_GREEN,  "Output\nDataset\nor Figure\n(DATA/ FIGURES/)"),
        (5.4,  1.0, 2.0, 1.5, "#FFF3CD", "#B8860B", "Validation\nScript\nvalidate_*.py"),
        (8.0,  1.0, 2.0, 1.5, "#D5EDE0", C_GREEN,  "Approved\nOutput\n(PASS)"),
        (10.0, 1.0, 1.7, 1.5, "#FAE3CE", C_ORANGE, "Thesis\nTable\nor Figure"),
    ]

    for (x, y, w, h, fc, ec, txt) in steps:
        rbox(ax, x, y, w, h, fc, ec)
        # caixa de validacao: rotulo principal sobe p/ liberar espaco ao FAIL loop
        ty = y + h*0.66 if fc == "#FFF3CD" else y + h/2
        ax.text(x + w/2, ty, txt, ha="center", va="center",
                fontsize=8.5, color="#1a1a1a")

    for x0, x1 in [(2.2, 2.8), (4.8, 5.4), (7.4, 8.0), (10.0, 10.0)]:
        if x0 < x1:
            arr(ax, x0, 1.75, x1, 1.75)

# FAIL path -- completamente dentro da caixa amarela
    ax.annotate("", xy=(5.75, 1.5), xytext=(7.05, 1.5),
                arrowprops=dict(arrowstyle="<-", color=C_RED, lw=1.1))
    ax.text(6.4, 1.24, "FAIL → debug & rebuild", ha="center",
            fontsize=6.8, color=C_RED, style="italic")

    ax.text(6.0, 0.25,
            "TDD-inspired incremental validation: each pipeline stage is checked before the next stage builds on it.\n"
            "Fixed seed = 42 ensures numerical reproducibility.",
            ha="center", fontsize=7.5, color=C_GREY, style="italic")

    plt.tight_layout(pad=0.4)
    plt.savefig(FIG_DIR / "fig_validation_pipeline.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved: fig_validation_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_A()
    fig_B()
    fig_C()
    fig_D()
    fig_E()
    fig_F()
    fig_G()
    fig_H1()
    fig_H2()
    fig_I()
    print("All subfigures generated.")
