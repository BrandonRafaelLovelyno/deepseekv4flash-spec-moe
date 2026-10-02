"""Matplotlib figures for the expert-predictor article stage.

Per resident ratio (50% / 75%) two views of the same data: a grouped-bar chart
(x = layer, one bar per look-ahead distance) and a line chart (one line per
distance across layer depth). Both plot ready recall at 8 experts fetched per
token. matplotlib is imported inside each function, so importing this module
never pulls the plotting stack; every function returns PNG bytes.
"""

from __future__ import annotations

from typing import Any

INK = "#1A1A1A"
GRID = "#D9D9D9"
CLAMP_FILL = "#000000"

DISTANCE_COLORS = {2: "#3E6E9E", 4: "#C77B30", 7: "#4E8A6B"}
FALLBACK_COLOR = "#666666"

KS = 8


def _style() -> dict[str, Any]:
    """A clean publication style: sans-serif, light grid, no decoration."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "text.color": INK,
        "axes.edgecolor": "#BBBBBB",
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 200,
    }


def _figure_bytes(fig: Any) -> bytes:
    import io

    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return buffer.getvalue()


def _color(distance: int) -> str:
    return DISTANCE_COLORS.get(distance, FALLBACK_COLOR)


def _by_layer(
    payload: dict[str, Any], metric: str
) -> dict[int, dict[int, float]]:
    """Index each distance's per-layer value: ``{distance: {layer: value}}``."""
    out: dict[int, dict[int, float]] = {}
    for row in payload["rows"]:
        out.setdefault(row["distance"], {})[row["layer"]] = float(row[metric])
    return out


def _ratio_label(ratio: float) -> int:
    return round(ratio * 100)


def _shade_clamped(ax: Any, layers: list[int], clamped_below: int) -> None:
    """Shade the early layers where the deepest look-ahead is not yet effective."""
    import matplotlib.transforms as mtransforms

    if clamped_below <= layers[0]:
        return
    stop = next(
        (i for i, layer in enumerate(layers) if layer >= clamped_below), len(layers)
    )
    if stop <= 0:
        return
    ax.axvspan(
        -0.5, stop - 0.5, color=CLAMP_FILL, alpha=0.06, linewidth=0, zorder=0
    )
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        (stop - 1) / 2.0,
        0.97,
        "clamped",
        transform=trans,
        ha="center",
        va="top",
        fontsize=8,
        color="#888888",
    )


def _ylim(ax: Any, values: list[float]) -> None:
    if not values:
        return
    low = max(0.0, min(values) - 0.02)
    ax.set_ylim(low, 1.005)


def bar_png(payload: dict[str, Any], ratio: float) -> bytes:
    """Grouped bars: layer along x, one bar per distance, ready recall on y."""
    import matplotlib.pyplot as plt
    import numpy as np

    metric = payload["metrics"][ratio]
    by_layer = _by_layer(payload, metric)
    distances = payload["distances"]
    layers = payload["layers"]
    x = np.arange(len(layers))
    width = 0.8 / max(len(distances), 1)

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(16, 5.2))
        _shade_clamped(ax, layers, payload["clamped_below"])
        plotted: list[float] = []
        for i, distance in enumerate(distances):
            series = by_layer.get(distance, {})
            heights = [series.get(layer, np.nan) for layer in layers]
            plotted.extend(v for v in heights if not np.isnan(v))
            offset = (i - (len(distances) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                heights,
                width,
                label=f"distance {distance}",
                color=_color(distance),
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(layer) for layer in layers], rotation=90, fontsize=8)
        ax.set_xlabel("MoE layer")
        ax.set_ylabel(f"ready recall @{KS}\n({_ratio_label(ratio)}% resident)")
        ax.set_title(
            f"Ready recall @{KS} by layer and look-ahead distance "
            f"({_ratio_label(ratio)}% resident)",
            fontsize=14,
            fontweight="bold",
            pad=24,
        )
        ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        _ylim(ax, plotted)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            ncol=len(distances),
        )
        fig.tight_layout()
        return _figure_bytes(fig)


def line_png(payload: dict[str, Any], ratio: float) -> bytes:
    """One line per distance: ready recall across layer depth."""
    import matplotlib.pyplot as plt

    metric = payload["metrics"][ratio]
    by_layer = _by_layer(payload, metric)
    layers = payload["layers"]

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(12, 5.2))
        _shade_clamped(ax, layers, payload["clamped_below"])
        plotted: list[float] = []
        for distance in payload["distances"]:
            series = by_layer.get(distance, {})
            ys = [series.get(layer) for layer in layers]
            plotted.extend(v for v in ys if v is not None)
            ax.plot(
                layers,
                ys,
                "-o",
                markersize=3.5,
                linewidth=1.8,
                color=_color(distance),
                label=f"distance {distance}",
                zorder=3,
            )
        ax.set_xlabel("MoE layer")
        ax.set_ylabel(f"ready recall @{KS}\n({_ratio_label(ratio)}% resident)")
        ax.set_title(
            f"Ready recall @{KS} across layer depth, by look-ahead distance "
            f"({_ratio_label(ratio)}% resident)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        _ylim(ax, plotted)
        ax.legend(loc="lower right", frameon=False, ncol=len(payload["distances"]))
        fig.tight_layout()
        return _figure_bytes(fig)


def build_figures(payload: dict[str, Any]) -> dict[str, bytes]:
    """Render both views for every configured ratio, keyed by file name."""
    figures: dict[str, bytes] = {}
    for ratio in payload["metrics"]:
        tag = f"r{_ratio_label(ratio)}"
        figures[f"ready_recall_{tag}_bar.png"] = bar_png(payload, ratio)
        figures[f"ready_recall_{tag}_line.png"] = line_png(payload, ratio)
    return figures
