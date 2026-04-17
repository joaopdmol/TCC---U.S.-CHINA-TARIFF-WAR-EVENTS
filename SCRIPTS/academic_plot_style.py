from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


TITLE_FONTSIZE = 16
SUBTITLE_FONTSIZE = 13
AXIS_FONTSIZE = 11
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10

ESCALATION_COLOR = "#d62728"
RELIEF_COLOR = "#2ca02c"
NEUTRAL_COLOR = "#b7950b"
GRID_ALPHA = 0.18


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": TICK_FONTSIZE,
            "axes.titlesize": SUBTITLE_FONTSIZE,
            "axes.labelsize": AXIS_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "figure.titlesize": TITLE_FONTSIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def apply_axis_style(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    percent_x: bool = False,
    percent_y: bool = False,
    grid_axis: str = "y",
) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=SUBTITLE_FONTSIZE, pad=10)

    if percent_x:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.grid(axis=grid_axis, linestyle="--", alpha=GRID_ALPHA, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def add_reference_lines(
    ax,
    *,
    horizontal_zero: bool = False,
    vertical_zero: bool = False,
) -> None:
    if horizontal_zero:
        ax.axhline(0, color="#4d4d4d", linewidth=1.2, alpha=0.9)
    if vertical_zero:
        ax.axvline(0, color="#4d4d4d", linewidth=1.2, linestyle="--", alpha=0.9)


def shared_limits(values, pad_ratio: float = 0.12) -> tuple[float, float]:
    flat_values = [value for value in values if value is not None]
    if not flat_values:
        return (-1.0, 1.0)

    min_value = min(flat_values)
    max_value = max(flat_values)
    if min_value == max_value:
        pad = abs(min_value) * pad_ratio if min_value != 0 else 0.01
        return (min_value - pad, max_value + pad)

    pad = (max_value - min_value) * pad_ratio
    return (min_value - pad, max_value + pad)


def place_figure_legend(fig, handles, labels, *, ncol: int = 2, anchor_y: float = 1.02) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        frameon=False,
        ncol=ncol,
        fontsize=LEGEND_FONTSIZE,
    )


def save_figure(fig, output_path: Path, *, top_rect: float = 0.88) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, top_rect))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
