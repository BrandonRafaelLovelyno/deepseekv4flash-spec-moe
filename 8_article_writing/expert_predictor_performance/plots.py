"""Matplotlib figures for the expert-predictor article stage.

Per resident ratio (50% / 75%) two views of the same data: a grouped-bar chart
(x = layer, one bar per look-ahead distance) and a line chart (one line per
distance across layer depth). Both plot ready recall at 8 experts fetched per
token. A third figure shows, for both ratios at once, which look-ahead distance
was chosen for each layer, shaded light-to-dark by distance. matplotlib is
imported inside each function, so importing this module never pulls the
plotting stack; every function returns PNG bytes.
"""

from __future__ import annotations

from typing import Any

INK = "#1A1A1A"
GRID = "#D9D9D9"
CLAMP_FILL = "#000000"

DISTANCE_COLORS = {2: "#3E6E9E", 4: "#C77B30", 7: "#4E8A6B"}
FALLBACK_COLOR = "#666666"

# The choice figure reads distance as intensity, so it needs its own sequential
# ramp (light = nearest look-ahead, dark = furthest) rather than the qualitative
# per-distance colours used by the bar/line views.
CHOICE_LIGHT = "#DCE7F2"
CHOICE_DARK = "#16324F"

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


def _choice_ramp(distances: list[int]) -> tuple[Any, Any]:
    """A light-to-dark discrete map plus norm over ascending distances."""
    import matplotlib.colors as mcolors
    import numpy as np

    ramp = mcolors.LinearSegmentedColormap.from_list(
        "choice", [CHOICE_LIGHT, CHOICE_DARK]
    )
    shades = [
        ramp(rank / (len(distances) - 1)) if len(distances) > 1 else ramp(1.0)
        for rank in range(len(distances))
    ]
    cmap = mcolors.ListedColormap(shades)
    norm = mcolors.BoundaryNorm(
        np.arange(len(distances) + 1) - 0.5, cmap.N
    )
    return cmap, norm


def layer_choice_png(payload: dict[str, Any]) -> bytes:
    """Chosen look-ahead distance per layer, one colour strip per resident ratio.

    Distances are ranked nearest-to-furthest and shaded light-to-dark, so a
    darker cell means that layer was assigned a longer look-ahead. The two rows
    (50% / 75% resident) sit on a shared layer axis for direct comparison.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    distances = sorted(payload["distances"])
    rank = {distance: i for i, distance in enumerate(distances)}
    layers = payload["layers"]
    cmap, norm = _choice_ramp(distances)
    shades = [cmap(i) for i in range(len(distances))]
    clamped = payload["clamped_below"]
    stop = next(
        (i for i, layer in enumerate(layers) if layer >= clamped), len(layers)
    )

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            len(payload["metrics"]), 1, figsize=(16, 3.4), sharex=True
        )
        if len(payload["metrics"]) == 1:
            axes = [axes]
        for ax, ratio in zip(axes, payload["metrics"]):
            chosen = payload["choices"][ratio]
            strip = [[rank[chosen[layer]] for layer in layers]]
            ax.imshow(strip, cmap=cmap, norm=norm, aspect="auto", zorder=1)
            if 0 < stop < len(layers):
                ax.axvspan(
                    -0.5,
                    stop - 0.5,
                    color=CLAMP_FILL,
                    alpha=0.08,
                    linewidth=0,
                    zorder=2,
                )
            for column, layer in enumerate(layers):
                shade = shades[rank[chosen[layer]]]
                dark = sum(mcolors.to_rgb(shade)) / 3.0 < 0.55
                ax.text(
                    column,
                    0,
                    str(chosen[layer]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if dark else INK,
                    zorder=4,
                )
            ax.set_yticks([])
            ax.set_ylabel(f"{_ratio_label(ratio)}% resident", fontsize=10)
        if 0 < stop < len(layers):
            trans = mtransforms.blended_transform_factory(
                axes[0].transData, axes[0].transAxes
            )
            axes[0].text(
                (stop - 1) / 2.0,
                1.12,
                "clamped: not comparable",
                transform=trans,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#888888",
            )
        axes[-1].set_xticks(range(len(layers)))
        axes[-1].set_xticklabels(
            [str(layer) for layer in layers], rotation=90, fontsize=8
        )
        axes[-1].set_xlabel("MoE layer")
        fig.suptitle(
            "Chosen look-ahead distance per layer "
            f"(furthest with ready recall @{KS} \u2265 "
            f"{payload['recall_target']:.2f})",
            fontsize=14,
            fontweight="bold",
        )
        fig.subplots_adjust(
            left=0.055, right=0.9, top=0.74, bottom=0.17, hspace=0.45
        )
        cax = fig.add_axes([0.92, 0.24, 0.012, 0.5])
        bar = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            ticks=range(len(distances)),
        )
        bar.ax.set_yticklabels([str(distance) for distance in distances])
        bar.set_label("look-ahead distance", fontsize=9)
        return _figure_bytes(fig)


def build_figures(payload: dict[str, Any]) -> dict[str, bytes]:
    """Render both views for every configured ratio, keyed by file name."""
    figures: dict[str, bytes] = {}
    for ratio in payload["metrics"]:
        tag = f"r{_ratio_label(ratio)}"
        figures[f"ready_recall_{tag}_bar.png"] = bar_png(payload, ratio)
        figures[f"ready_recall_{tag}_line.png"] = line_png(payload, ratio)
    figures["lookahead_distance_choice.png"] = layer_choice_png(payload)
    return figures
