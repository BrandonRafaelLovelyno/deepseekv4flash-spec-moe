"""Matplotlib figures for the final simulation.

matplotlib is imported inside each function, so importing this module (from the
local entrypoint or a test) never pulls the plotting stack. Every function returns
PNG bytes and writing them is the caller's concern.
"""

from __future__ import annotations

from typing import Any

PORTION_COLORS = {0.75: "#4C72B0", 0.5: "#DD8452", 0.125: "#55A868"}
CHUNK_COLORS = {64: "#4C72B0", 128: "#DD8452", 256: "#55A868", 512: "#8172B3"}
RATIO_COLORS = {0.25: "#4C72B0", 0.5: "#DD8452", 0.75: "#55A868"}
LINESTYLES = ["-", "--", "-.", ":"]
STATIC_COLOR = "#4C72B0"
CACHED_COLOR = "#DD8452"


def _portion(counts: Any) -> Any:
    """Turn a missing-count histogram into a portion curve."""
    import numpy as np

    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    return counts / total if total else counts


def _mean_missing(counts: Any) -> float:
    import numpy as np

    portion = _portion(counts)
    return float((np.arange(portion.shape[0]) * portion).sum())


def _cdf(portion: Any) -> Any:
    import numpy as np

    return np.cumsum(portion)


def _figure_bytes(fig: Any) -> bytes:
    import io

    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _primary_fetch(sim: dict[str, Any]) -> int:
    """The fetch count whose CDF is drawn, snapped to the nearest configured one."""
    fetches = sorted(int(count) for count in sim["fetch_counts"])
    target = int(sim.get("primary_fetch", fetches[-1]))
    return min(fetches, key=lambda count: abs(count - target))


def _grid(panels: list, ncols: int, cell: tuple[float, float]) -> tuple:
    import matplotlib.pyplot as plt

    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False
    )
    return fig, axes.ravel()


def cdf_missing_png(
    histograms: dict, sim: dict[str, Any], splits: list[str], ratio: float, fetch: int
) -> bytes:
    """CDF of missing experts per chunk: rows = chunk sizes, cols = splits."""

    portions = sorted(float(f) for f in sim["decode_portions"])
    chunk_sizes = sorted(int(b) for b in sim["chunk_sizes"])
    panels = [
        (chunk_size, split)
        for chunk_size in chunk_sizes
        for split in splits
    ]
    fig, grid = _grid(panels, len(splits), (4.6, 3.2))
    for ax, (chunk_size, split) in zip(grid, panels):
        for portion in portions:
            counts = histograms.get(
                (split, portion, chunk_size, ratio, "static", fetch)
            )
            if counts is None:
                continue
            cdf = _cdf(_portion(counts))
            ax.plot(
                range(cdf.shape[0]),
                cdf,
                "-o",
                markersize=2.5,
                color=PORTION_COLORS.get(portion),
                label=f"{int(portion * 100)}% decode",
            )
        ax.set_title(f"{split} B={chunk_size}")
        ax.set_xlabel("missing experts per chunk (<= m)")
        ax.grid(alpha=0.3)
    for ax in grid[len(panels):]:
        ax.axis("off")
    grid[0].set_ylabel("P(missing <= m)")
    grid[len(splits) - 1].legend(fontsize=7)
    fig.suptitle(
        f"Chunked-prefill CDF of missing experts, fetch={fetch}, "
        f"resident={round(ratio * 100)}%"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _figure_bytes(fig)


def mean_missing_vs_fetch_png(
    histograms: dict, sim: dict[str, Any], splits: list[str], ratio: float
) -> bytes:
    """Mean missing experts vs fetch count: rows = decode portions, cols = splits."""

    portions = sorted(float(f) for f in sim["decode_portions"])
    chunk_sizes = sorted(int(b) for b in sim["chunk_sizes"])
    fetches = sorted(int(count) for count in sim["fetch_counts"])
    panels = [(portion, split) for portion in portions for split in splits]
    fig, grid = _grid(panels, len(splits), (4.6, 3.2))
    for ax, (portion, split) in zip(grid, panels):
        for chunk_size in chunk_sizes:
            ys = [
                _mean_missing(
                    histograms.get(
                        (split, portion, chunk_size, ratio, "static", fetch), [0]
                    )
                )
                for fetch in fetches
            ]
            ax.plot(
                fetches,
                ys,
                "-o",
                markersize=3,
                color=CHUNK_COLORS.get(chunk_size),
                label=f"B={chunk_size}",
            )
        ax.set_title(f"{split}, {int(portion * 100)}% decode")
        ax.set_xlabel("experts fetched per layer per chunk")
        ax.set_ylabel("mean missing experts")
        ax.grid(alpha=0.3)
    for ax in grid[len(panels):]:
        ax.axis("off")
    grid[len(splits) - 1].legend(fontsize=7)
    fig.suptitle(
        f"Mean missing experts vs fetch, resident={round(ratio * 100)}%"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _figure_bytes(fig)


def static_vs_cached_cdf_png(
    histograms: dict, sim: dict[str, Any], splits: list[str], ratio: float, fetch: int
) -> bytes:
    """Overlay static (dashed) vs cached (solid) CDFs: rows = chunks, cols = splits."""

    portions = sorted(float(f) for f in sim["decode_portions"])
    chunk_sizes = sorted(int(b) for b in sim["chunk_sizes"])
    panels = [(chunk_size, split) for chunk_size in chunk_sizes for split in splits]
    fig, grid = _grid(panels, len(splits), (4.6, 3.2))
    for ax, (chunk_size, split) in zip(grid, panels):
        for portion in portions:
            color = PORTION_COLORS.get(portion)
            static = histograms.get(
                (split, portion, chunk_size, ratio, "static", fetch)
            )
            cached = histograms.get((split, portion, chunk_size, ratio, "cached", 0))
            if static is not None:
                cdf = _cdf(_portion(static))
                ax.plot(
                    range(cdf.shape[0]),
                    cdf,
                    "--",
                    linewidth=1.0,
                    color=color,
                    label=f"{int(portion * 100)}% static",
                )
            if cached is not None:
                cdf = _cdf(_portion(cached))
                ax.plot(
                    range(cdf.shape[0]),
                    cdf,
                    "-o",
                    markersize=2.5,
                    linewidth=1.2,
                    color=color,
                    label=f"{int(portion * 100)}% cache",
                )
        ax.set_title(f"{split} B={chunk_size}")
        ax.set_xlabel("missing experts per chunk (<= m)")
        ax.grid(alpha=0.3)
    for ax in grid[len(panels):]:
        ax.axis("off")
    grid[0].set_ylabel("P(missing <= m)")
    grid[len(splits) - 1].legend(fontsize=6, ncol=2)
    fig.suptitle(
        f"Static vs adaptive cache, fetch={fetch}, resident={round(ratio * 100)}%"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _figure_bytes(fig)


def static_vs_cached_mean_png(
    histograms: dict, sim: dict[str, Any], splits: list[str], ratio: float, fetch: int
) -> bytes:
    """Mean missing vs chunk size, static vs cached: rows = portions, cols = splits."""

    portions = sorted(float(f) for f in sim["decode_portions"])
    chunk_sizes = sorted(int(b) for b in sim["chunk_sizes"])
    panels = [(portion, split) for portion in portions for split in splits]
    fig, grid = _grid(panels, len(splits), (4.6, 3.2))
    for ax, (portion, split) in zip(grid, panels):
        static = [
            _mean_missing(
                histograms.get(
                    (split, portion, chunk_size, ratio, "static", fetch), [0]
                )
            )
            for chunk_size in chunk_sizes
        ]
        cached = [
            _mean_missing(
                histograms.get(
                    (split, portion, chunk_size, ratio, "cached", 0), [0]
                )
            )
            for chunk_size in chunk_sizes
        ]
        ax.plot(
            chunk_sizes, static, "--o", markersize=3,
            color=STATIC_COLOR, label=f"static fetch={fetch}",
        )
        ax.plot(
            chunk_sizes, cached, "-o", markersize=3,
            color=CACHED_COLOR, label="cached (oracle)",
        )
        ax.set_title(f"{split}, {int(portion * 100)}% decode")
        ax.set_xlabel("chunk size B")
        ax.set_ylabel("mean missing experts")
        ax.set_xticks(chunk_sizes)
        ax.grid(alpha=0.3)
    for ax in grid[len(panels):]:
        ax.axis("off")
    grid[0].legend(fontsize=7)
    fig.suptitle(f"Mean missing: static vs adaptive cache, resident={round(ratio * 100)}%")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _figure_bytes(fig)


def cache_traffic_png(
    traffic: dict, sim: dict[str, Any], splits: list[str]
) -> bytes:
    """Cache loads (solid) / evictions (dashed) vs chunk size, per ratio and split."""

    portions = sorted(float(f) for f in sim["decode_portions"])
    chunk_sizes = sorted(int(b) for b in sim["chunk_sizes"])
    ratios = sorted(float(r) for r in sim["ready_ratios"])
    panels = [(portion, split) for portion in portions for split in splits]
    fig, grid = _grid(panels, len(splits), (4.6, 3.2))
    for ax, (portion, split) in zip(grid, panels):
        for ratio in ratios:
            cells = {
                chunk_size: traffic.get(
                    f"{split}|{portion}|{chunk_size}", {}
                ).get(str(ratio), {})
                for chunk_size in chunk_sizes
            }
            loads = [cells[chunk_size].get("loads", 0) for chunk_size in chunk_sizes]
            evictions = [
                cells[chunk_size].get("evictions", 0) for chunk_size in chunk_sizes
            ]
            color = RATIO_COLORS.get(ratio)
            ax.plot(
                chunk_sizes, loads, "-o", markersize=3, color=color,
                label=f"keep {int(ratio * 100)}% load",
            )
            ax.plot(
                chunk_sizes, evictions, "--s", markersize=3, color=color,
                label=f"keep {int(ratio * 100)}% evict",
            )
        ax.set_title(f"{split}, {int(portion * 100)}% decode")
        ax.set_xlabel("chunk size B")
        ax.set_ylabel("cumulative loads / evictions")
        ax.set_xticks(chunk_sizes)
        ax.grid(alpha=0.3)
    for ax in grid[len(panels):]:
        ax.axis("off")
    grid[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Adaptive cache traffic (chunked prefill)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _figure_bytes(fig)


def build_figures(payload: dict[str, Any]) -> dict[str, bytes]:
    """Render every figure this run produces, keyed by file name."""
    import matplotlib

    matplotlib.use("Agg")

    sim = payload["summary"]["config"]["simulation"]
    splits = [str(split) for split in payload["summary"]["config"]["data"]["split"]]
    splits = [split for split in splits if any(key[0] == split for key in payload["histograms"])]
    ratios = sorted(float(ratio) for ratio in sim["ready_ratios"])
    policies = [str(policy) for policy in payload["summary"].get("policies", [])]
    fetch = _primary_fetch(sim)

    figures: dict[str, bytes] = {}
    for ratio in ratios:
        tag = f"r{round(ratio * 100)}"
        figures[f"cdf_missing_{tag}.png"] = cdf_missing_png(
            payload["histograms"], sim, splits, ratio, fetch
        )
        figures[f"mean_missing_vs_fetch_{tag}.png"] = mean_missing_vs_fetch_png(
            payload["histograms"], sim, splits, ratio
        )
        if "cached" in policies:
            figures[f"static_vs_cached_cdf_{tag}.png"] = static_vs_cached_cdf_png(
                payload["histograms"], sim, splits, ratio, fetch
            )
            figures[f"static_vs_cached_mean_{tag}.png"] = static_vs_cached_mean_png(
                payload["histograms"], sim, splits, ratio, fetch
            )
    if "cached" in policies and payload["summary"].get("traffic"):
        figures["cache_traffic.png"] = cache_traffic_png(
            payload["summary"]["traffic"], sim, splits
        )
    return figures
