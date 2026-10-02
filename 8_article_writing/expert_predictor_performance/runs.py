"""Load the trained per-layer predictor runs for the article figure.

Each ``output/<run_id>/`` directory is one full train-all run: a ``summary.json``
(whose ``config.task.distance`` and ``config.model.arch`` describe the run) and a
``layers.csv`` (one row per MoE layer, carrying every ``ready_recall_*@k``
metric). This module discovers those runs, keeps the MLP ones, reshapes the
tables into a tidy per-``(layer, distance, ratio)`` form the plotter consumes,
and derives the chosen per-layer look-ahead distance (the furthest distance
clearing ``RECALL_TARGET``, else the nearest).

Only the standard library is imported at module scope, so ``main.py`` stays
importable on a machine without the scientific stack.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")

ARCH = "mlp"
KS = 8
RATIOS = (0.5, 0.75)

# Per-layer look-ahead selection: prefer the furthest distance (most lead time)
# that still clears this ready-recall bar; fall back to the nearest distance when
# no distance reaches it.
RECALL_TARGET = 0.9


def _metric(ratio: float, k: int = KS) -> str:
    """The ``ready_recall`` column name for a resident ratio and expert budget."""
    return f"ready_recall_r{round(ratio * 100)}@{k}"


METRICS = {ratio: _metric(ratio) for ratio in RATIOS}


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def list_runs(input_dir: str = INPUT_DIR, arch: str = ARCH) -> list[dict[str, Any]]:
    """Every run directory whose summary matches ``arch``; one entry per run.

    Runs are returned ordered by their look-ahead ``distance`` so the legend and
    the bar groups read low-to-high.
    """
    runs: list[dict[str, Any]] = []
    for run_id in sorted(os.listdir(input_dir)):
        run_dir = os.path.join(input_dir, run_id)
        summary_path = os.path.join(run_dir, "summary.json")
        csv_path = os.path.join(run_dir, "layers.csv")
        if not (os.path.isdir(run_dir) and os.path.exists(summary_path)):
            continue
        summary = _read_json(summary_path)
        config = summary.get("config", {})
        if config.get("model", {}).get("arch") != arch:
            continue
        runs.append(
            {
                "run_id": run_id,
                "distance": int(config["task"]["distance"]),
                "csv_path": csv_path,
            }
        )
    runs.sort(key=lambda run: run["distance"])
    return runs


def load_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten each run's ``layers.csv`` into tidy per-layer rows.

    Each row carries the run's look-ahead ``distance`` alongside the layer's own
    ``distance`` column (its ``effective_distance`` -- clamped at layer 0 for the
    early layers) and the ``ready_recall`` values keyed by ratio.
    """
    rows: list[dict[str, Any]] = []
    for run in runs:
        with open(run["csv_path"], encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                row: dict[str, Any] = {
                    "layer": int(record["layer"]),
                    "distance": run["distance"],
                    "effective_distance": int(record["distance"]),
                    "run_id": run["run_id"],
                }
                for column in METRICS.values():
                    row[column] = float(record[column])
                rows.append(row)
    return rows


def clamped_below(rows: list[dict[str, Any]]) -> int:
    """The first layer at which every distance is fully effective.

    Layer ``L`` reads activation ``L - min(L, distance)``, so the deepest
    look-ahead only takes effect at ``L >= distance``; below ``max(distance)``
    the series are not comparable and the plot shades that region.
    """
    return max((row["distance"] for row in rows), default=0)


def layers(rows: list[dict[str, Any]]) -> list[int]:
    """Every measured layer, ascending."""
    return sorted({row["layer"] for row in rows})


def choose_distances(
    rows: list[dict[str, Any]],
    metric: str,
    distances: list[int],
    target: float = RECALL_TARGET,
) -> dict[int, int]:
    """The chosen look-ahead distance per layer for one metric.

    The furthest ``distance`` whose ready recall meets ``target`` wins, so the
    predictor gives the loader as much lead time as that layer can afford; when
    no distance reaches the bar, the nearest (smallest) distance is used.
    """
    by_layer: dict[int, dict[int, float]] = {}
    for row in rows:
        by_layer.setdefault(row["layer"], {})[row["distance"]] = float(row[metric])
    chosen: dict[int, int] = {}
    for layer, values in by_layer.items():
        qualified = [d for d in distances if values.get(d, float("-inf")) >= target]
        chosen[layer] = max(qualified) if qualified else min(distances)
    return chosen


def load(input_dir: str = INPUT_DIR, arch: str = ARCH) -> dict[str, Any]:
    """Discover the runs and assemble the payload the figure builder consumes."""
    runs = list_runs(input_dir, arch=arch)
    if not runs:
        raise FileNotFoundError(
            f"no {arch!r} runs with a summary.json under {input_dir}; "
            "run 6_train_all first"
        )
    rows = load_rows(runs)
    distances = [run["distance"] for run in runs]
    return {
        "rows": rows,
        "runs": runs,
        "distances": distances,
        "metrics": METRICS,
        "choices": {
            ratio: choose_distances(rows, metric, distances)
            for ratio, metric in METRICS.items()
        },
        "recall_target": RECALL_TARGET,
        "clamped_below": clamped_below(rows),
        "layers": layers(rows),
    }
