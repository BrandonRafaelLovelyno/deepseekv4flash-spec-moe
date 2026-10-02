"""Per-layer expert-predictor recall, by look-ahead distance.

The article companion to ``6_train_all``: it reads the finished per-layer runs
placed under ``input/`` and plots ready recall at 8 experts fetched per token,
layer by layer, for each look-ahead distance. For the 50% and 75% resident
ratios it renders both a grouped-bar view (x = layer, one bar per distance) and
a line view (one line per distance across depth).

Inputs are the run directories themselves, so this stage is CPU only -- no
predictor, no GPU, no training run. It discovers every MLP run under ``input/``,
selects the depths to compare, and writes the figures plus a tidy CSV into
``output/``.

Usage:
    python 8_article_writing/expert_predictor_performance/main.py
"""

from __future__ import annotations

import os
from typing import Any

import plots
import runs

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "output")
CSV_COLUMNS = ["layer", "distance", "effective_distance"]


def load_data() -> dict[str, Any]:
    """Discover the MLP runs and assemble the plotting payload."""
    return runs.load()


def write_rows(path: str, rows: list[dict[str, Any]], metrics: dict[float, str]) -> None:
    """Write the plotted per-layer values as CSV, one row per (run, layer)."""
    columns = CSV_COLUMNS + list(metrics.values())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in sorted(rows, key=lambda r: (r["distance"], r["layer"])):
            cells = [str(row.get(column, "")) for column in columns]
            handle.write(",".join(cells) + "\n")


def write_artifacts(payload: dict[str, Any]) -> list[str]:
    """Render the figures, write them and the CSV; return the written names."""
    os.makedirs(OUT_DIR, exist_ok=True)
    figures = plots.build_figures(payload)
    written = ["recall_by_distance.csv"]
    write_rows(
        os.path.join(OUT_DIR, "recall_by_distance.csv"),
        payload["rows"],
        payload["metrics"],
    )
    for name, data in figures.items():
        with open(os.path.join(OUT_DIR, name), "wb") as handle:
            handle.write(data)
        written.append(name)
    return written


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def print_report(payload: dict[str, Any]) -> None:
    """Print the layer-mean ready recall per distance, per ratio."""
    print(
        f"\nlayers: {len(payload['layers'])} "
        f"(clamped below {payload['clamped_below']})  "
        f"distances: {payload['distances']}"
    )
    for metric in payload["metrics"].values():
        print(f"\nmean {metric} (layer mean):")
        for distance in payload["distances"]:
            values = [
                row[metric]
                for row in payload["rows"]
                if row["distance"] == distance
            ]
            effective = [
                row[metric]
                for row in payload["rows"]
                if row["distance"] == distance
                and row["layer"] >= payload["clamped_below"]
            ]
            print(
                f"  distance {distance}: {_mean(values):.4f}  "
                f"(layers>={payload['clamped_below']}: {_mean(effective):.4f})"
            )


def main() -> None:
    payload = load_data()
    written = write_artifacts(payload)
    print_report(payload)
    print(f"\nwrote {len(written)} files to {OUT_DIR}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
