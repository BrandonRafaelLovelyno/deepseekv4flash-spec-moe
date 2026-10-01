"""Matplotlib figures for the average-used-experts run.

A clean, professional figure set for the article: a sans-serif face, a light
grid, muted colorblind-friendly series colours and no decorative styling.
matplotlib is imported inside each function so importing this module never pulls
the plotting stack; every function returns PNG bytes and writing them is the
caller's concern.
"""

from __future__ import annotations

from typing import Any

INK = "#1A1A1A"
GRID = "#D9D9D9"

PORTION_COLORS = {0.75: "#3E6E9E", 0.5: "#C77B30", 0.125: "#4E8A6B"}


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
        "legend.fontsize": 9,
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


def _splits(summary: dict[str, Any]) -> list[str]:
    present = {key.split("|")[0] for key in summary["combos"]}
    ordered = [str(split) for split in summary["config"]["data"]["split"]]
    return [split for split in ordered if split in present]


def _by_layer(
    rows: list[dict[str, Any]]
) -> dict[tuple[str, float, int], dict[int, float]]:
    """Index per-layer mean distinct counts by ``(split, portion, chunk)``."""
    out: dict[tuple[str, float, int], dict[int, float]] = {}
    for row in rows:
        key = (row["split"], float(row["decode_portion"]), int(row["chunk_size"]))
        out.setdefault(key, {})[int(row["layer"])] = float(row["mean_distinct"])
    return out


def mean_distinct_vs_chunk_png(payload: dict[str, Any]) -> bytes:
    """Mean distinct experts vs chunk size: lines = decode portions, band = layers.

    Train and test are merged into one panel with equal weight per split (see
    ``usage.finalize``'s ``merged`` series).
    """
    import matplotlib.pyplot as plt

    summary = payload["summary"]
    merged = summary["merged"]
    portions = sorted({float(key.split("|")[0]) for key in merged})
    chunk_sizes = summary["chunk_sizes"]
    n_experts = int(summary["n_experts"])

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        for portion in portions:
            xs: list[int] = []
            mean: list[float] = []
            low: list[float] = []
            high: list[float] = []
            for size in chunk_sizes:
                stats = merged.get(f"{portion}|{size}")
                if stats is None:
                    continue
                xs.append(size)
                mean.append(float(stats["layer_mean_distinct"]))
                low.append(float(stats["layer_p10"]))
                high.append(float(stats["layer_p90"]))
            color = PORTION_COLORS.get(portion, "#666666")
            ax.plot(
                xs, mean, "-o", color=color, linewidth=2.0, markersize=4,
                label=f"{int(round(portion * 100))}% decode",
            )
            ax.fill_between(xs, low, high, color=color, alpha=0.12, linewidth=0)
        ax.axhline(n_experts, color="#888888", linestyle="--", linewidth=1.2)
        ax.text(
            chunk_sizes[-1], n_experts, f" all {n_experts} experts",
            va="bottom", ha="right", fontsize=9, color="#666666",
        )
        ax.set_title(
            "Distinct routed experts activated by one prefill chunk",
            fontsize=14, fontweight="bold",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(chunk_sizes)
        ax.set_xticklabels([str(size) for size in chunk_sizes])
        ax.set_xlabel("Prefill chunk size")
        ax.set_ylabel("Mean distinct experts per chunk")
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        return _figure_bytes(fig)


def layer_heatmap_png(payload: dict[str, Any], portion: float) -> bytes:
    """Per-layer heatmap: rows = layers, cols = chunk sizes, cell = mean distinct."""
    import matplotlib.pyplot as plt
    import numpy as np

    summary = payload["summary"]
    rows = payload["rows"]
    by_layer = _by_layer(rows)
    splits = _splits(summary)
    chunk_sizes = summary["chunk_sizes"]
    layers = [int(layer) for layer in summary["layers"]]

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            1, len(splits),
            figsize=(6.4 * len(splits), max(4.0, 0.2 * len(layers))),
            squeeze=False, layout="constrained",
        )
        image = None
        for ax, split in zip(axes.ravel(), splits):
            grid = np.full((len(layers), len(chunk_sizes)), np.nan)
            for i, layer in enumerate(layers):
                for j, size in enumerate(chunk_sizes):
                    value = by_layer.get((split, float(portion), size), {}).get(layer)
                    if value is not None:
                        grid[i, j] = value
            image = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
            ax.set_title(f"{split} split")
            ax.set_xticks(range(len(chunk_sizes)))
            ax.set_xticklabels([str(size) for size in chunk_sizes])
            ax.set_xlabel("Prefill chunk size")
            step = max(len(layers) // 10, 1)
            ax.set_yticks(range(0, len(layers), step))
            ax.set_yticklabels([str(layers[i]) for i in range(0, len(layers), step)])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes.ravel()[0].set_ylabel("MoE layer")
        if image is not None:
            colorbar = fig.colorbar(
                image, ax=list(axes.ravel()), fraction=0.025, pad=0.02
            )
            colorbar.set_label("Mean distinct experts", fontsize=10)
        fig.suptitle(
            f"Mean distinct experts per chunk, layer by layer "
            f"({int(round(portion * 100))}% decode)",
            fontsize=15, fontweight="bold",
        )
        return _figure_bytes(fig)


def build_figures(payload: dict[str, Any]) -> dict[str, bytes]:
    """Render every figure this run produces, keyed by file name."""
    import matplotlib

    matplotlib.use("Agg")

    from helper import primary_portion

    portion = primary_portion(payload["summary"]["config"])
    return {
        "mean_distinct_vs_chunk.png": mean_distinct_vs_chunk_png(payload),
        f"layer_heatmap_p{int(round(portion * 100))}.png": layer_heatmap_png(
            payload, portion
        ),
    }
