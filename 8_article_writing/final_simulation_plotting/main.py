"""CDF grids of missing experts for the article.

Reads a finished ``7_final_simulation`` run (``input/<run_id>/distributions.csv``)
and renders one publication-style 3x3 grid per resident ratio:

* rows  -- prefill chunk size B (64, 128, 256)
* cols  -- experts fetched per layer per chunk (20, 40, 60)
* lines -- decode portion of the mixed chunked-prefill load (12.5%, 50%, 75%)

Each panel is the CDF of missing experts per chunk on the test split; the x-axis
is truncated where the CDF reaches 1 so the panel stays compact.

Local, CPU-only plotting stage -- no Modal, no config. Figures land in
``output/cdf_grid_r<ratio>.png``.

Usage:
    python main.py [run_id]
"""

from __future__ import annotations

import csv
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")
OUT_DIR = os.path.join(THIS_DIR, "output")

SPLIT = "test"
POLICY = "static"
RATIOS = [0.25, 0.5, 0.75]
CHUNK_SIZES = [64, 128, 256]
FETCHES = [20, 40, 60]
PORTIONS = [0.125, 0.5, 0.75]

PORTION_COLORS = {0.75: "#3E6E9E", 0.5: "#C77B30", 0.125: "#4E8A6B"}
INK = "#1A1A1A"
GRID = "#D9D9D9"


def latest_run_dir(root: str) -> str:
    """The most recent run directory under ``root``."""
    runs = [
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    if not runs:
        raise FileNotFoundError(f"no run directories under {root}")
    return os.path.join(root, max(runs))


def load_series(csv_path: str) -> dict:
    """Index the wanted rows by combo as ``{(ratio, chunk, fetch, portion): (xs, ys)}``."""
    series: dict[tuple, tuple[list[int], list[float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != SPLIT or row["policy"] != POLICY:
                continue
            ratio = float(row["ratio"])
            chunk = int(row["chunk_size"])
            fetch = int(row["fetch"])
            portion = float(row["decode_portion"])
            if ratio not in RATIOS or chunk not in CHUNK_SIZES:
                continue
            if fetch not in FETCHES or portion not in PORTIONS:
                continue
            xs, ys = series.setdefault((ratio, chunk, fetch, portion), ([], []))
            xs.append(int(row["missing"]))
            ys.append(float(row["cdf"]))
    return series


def _style() -> dict:
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
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 200,
    }


def _figure_bytes(fig) -> bytes:
    import io

    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return buffer.getvalue()


def _x_cut(ys: list[float], xs: list[int]) -> int:
    """The smallest missing count at which this CDF reaches 1."""
    for x, y in zip(xs, ys):
        if y >= 1.0:
            return x
    return xs[-1] if xs else 0


def plot_grid(series: dict, ratio: float) -> bytes:
    """A 3x3 CDF grid (rows=B, cols=fetch) for one resident ratio."""
    import matplotlib.pyplot as plt

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            len(CHUNK_SIZES),
            len(FETCHES),
            figsize=(4.2 * len(FETCHES), 3.1 * len(CHUNK_SIZES)),
            squeeze=False,
            sharex=False,
            layout="constrained",
        )
        for i, chunk in enumerate(CHUNK_SIZES):
            for j, fetch in enumerate(FETCHES):
                ax = axes[i][j]
                x_max = 0
                for portion in PORTIONS:
                    xs, ys = series.get((ratio, chunk, fetch, portion), ([], []))
                    if not xs:
                        continue
                    ax.step(
                        xs,
                        ys,
                        where="post",
                        color=PORTION_COLORS[portion],
                        linewidth=1.6,
                        label=f"{portion * 100:g}% decode",
                    )
                    x_max = max(x_max, _x_cut(ys, xs))
                ax.set_xlim(0, x_max + 1 if x_max else 1)
                ax.set_ylim(0, 1.02)
                ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
                ax.set_axisbelow(True)
                if i == 0:
                    ax.set_title(f"fetch={fetch}")
                if j == 0:
                    ax.set_ylabel(f"B={chunk}\nP(missing ≤ m)")
                else:
                    ax.set_ylabel("")
                if i == len(CHUNK_SIZES) - 1:
                    ax.set_xlabel("missing experts per chunk")
        axes[0][0].legend(loc="lower right", frameon=False)
        fig.suptitle(
            f"CDF of missing experts per chunk — test split, "
            f"resident {ratio * 100:g}%",
            fontsize=14,
            fontweight="bold",
        )
        return _figure_bytes(fig)


def write_figures(figures: dict[str, bytes], out_dir: str) -> list[str]:
    """Write each named PNG into ``out_dir``; return the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in figures.items():
        path = os.path.join(out_dir, name)
        with open(path, "wb") as handle:
            handle.write(data)
        paths.append(path)
    return paths


def main() -> None:
    run_dir = (
        os.path.join(INPUT_DIR, sys.argv[1])
        if len(sys.argv) > 1
        else latest_run_dir(INPUT_DIR)
    )
    print(f"input: {run_dir}")

    series = load_series(os.path.join(run_dir, "distributions.csv"))
    figures = {
        f"cdf_grid_r{ratio * 100:g}.png": plot_grid(series, ratio)
        for ratio in RATIOS
    }
    for path in write_figures(figures, OUT_DIR):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
