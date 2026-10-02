"""Artifact IO and end-of-run reporting for a final simulation.

Standard library at module scope, so the local entrypoint can mirror a finished
run without importing numpy / matplotlib. Remote scope writes the run into its
volume directory (config, summary, distributions table, histogram archive and
figures); local scope mirrors the JSON/CSV/PNG subset and prints a report.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_bytes(path: str, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)


DISTRIBUTION_COLUMNS = [
    "split",
    "decode_portion",
    "chunk_size",
    "ratio",
    "policy",
    "fetch",
    "missing",
    "count",
    "portion",
    "cdf",
]


def write_distributions_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Write the per-combo missing-expert distributions as CSV."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(DISTRIBUTION_COLUMNS) + "\n")
        handle.writelines(",".join(str(row[column]) for column in DISTRIBUTION_COLUMNS) + "\n" for row in rows)


def _write_histograms(path: str, histograms: dict) -> None:
    """Archive every per-combo histogram as a flat ``.npz`` keyed by combo."""
    import numpy as np

    payload = {
        f"{split}|{portion}|{chunk_size}|{ratio}|{policy}|{fetch}": counts
        for (split, portion, chunk_size, ratio, policy, fetch), counts in histograms.items()
    }
    np.savez_compressed(path, **payload)


def write_run(
    out_dir: str,
    config_text: str,
    payload: dict[str, Any],
    figures: dict[str, bytes],
) -> list[str]:
    """Write a finished simulation into its volume directory; return the names."""
    os.makedirs(out_dir, exist_ok=True)
    summary = payload["summary"]

    _write_text(os.path.join(out_dir, "config.yaml"), config_text)
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    write_distributions_csv(
        os.path.join(out_dir, "distributions.csv"), payload["distributions"]
    )
    _write_histograms(os.path.join(out_dir, "histograms.npz"), payload["histograms"])

    written = ["config.yaml", "summary.json", "distributions.csv", "histograms.npz"]
    for name, data in figures.items():
        _write_bytes(os.path.join(out_dir, name), data)
        written.append(name)
    return written


def mirror_run(
    root: str,
    run_id: str,
    summary: dict[str, Any],
    distributions: list[dict[str, Any]],
    figures: dict[str, bytes],
) -> str:
    """Mirror the JSON/CSV/PNG subset into ``<root>/<run_id>/`` locally."""
    out_dir = os.path.join(root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    write_distributions_csv(os.path.join(out_dir, "distributions.csv"), distributions)
    for name, data in figures.items():
        _write_bytes(os.path.join(out_dir, name), data)
    return out_dir


def _int_keys(payload: dict) -> dict:
    """Re-key a JSON round-tripped ``{str: ...}`` map back to ``{int: ...}``."""
    return {int(key): value for key, value in payload.items()}


def print_profile_report(profile: dict[str, Any] | None) -> None:
    """Log the per-layer checkpoint assignment and the candidate recalls.

    ``profile`` is the ``distance_profile`` payload. A mixed profile prints one
    line per layer: the assigned run/distance/source plus the ready recall at
    every available distance, so a hand-written assignment can be sanity-checked.
    A uniform profile prints a single line.
    """
    if not profile or profile.get("mode") != "mixed":
        run_id = (profile or {}).get("run_id", "?")
        print(f"\n[profile] uniform: every layer from run {run_id}", flush=True)
        return

    layers = _int_keys(profile.get("layers", {}))
    candidates = {
        layer: _int_keys(scores)
        for layer, scores in _int_keys(profile.get("candidates", {})).items()
    }
    metric = profile.get("recall_metric", "recall")

    print(
        f"\n[profile] {profile.get('label', 'mixed')}  {len(layers)} layers  "
        f"recall={metric}",
        flush=True,
    )
    by_run: dict[str, int] = {}
    by_distance: dict[int, int] = {}
    for layer in sorted(layers):
        entry = layers[layer]
        run_id = str(entry["run_id"])
        distance = int(entry["distance"])
        source = int(entry["source_layer"])
        scores = candidates.get(layer, {})
        cells = " ".join(f"d{d}={scores[d]:.3f}" for d in sorted(scores))
        print(
            f"[profile] L{layer:02d}  run={run_id}  d={distance}  "
            f"src=L{source:02d}  {cells}",
            flush=True,
        )
        by_run[run_id] = by_run.get(run_id, 0) + 1
        by_distance[distance] = by_distance.get(distance, 0) + 1

    runs = " | ".join(f"{run} x{count}" for run, count in sorted(by_run.items()))
    distances = " | ".join(
        f"d{d} x{count}" for d, count in sorted(by_distance.items())
    )
    print(f"[profile] runs: {runs}", flush=True)
    print(f"[profile] distances: {distances}", flush=True)


def print_report(run_id: str, summary: dict[str, Any], out_dir: str) -> None:
    """Print a compact run report: layers, dispersal and mean-missing highlights."""
    print(f"\nrun_id: {run_id}  training_run: {summary['training_run_id']}")
    print(
        f"layers: {summary['n_layers']} "
        f"(skipped: {summary['skipped_layers'] or 'none'})"
    )

    sim = summary["config"]["simulation"]
    fetch = min(
        sorted(int(c) for c in sim["fetch_counts"]),
        key=lambda count: abs(count - int(sim.get("primary_fetch", count))),
    )
    ratio = max(float(r) for r in sim["ready_ratios"])

    def mean_table(policy: str, label: str, fetch_value: int) -> None:
        print(
            f"\nmean missing experts per chunk ({label}, keep {int(ratio * 100)}%):"
        )
        header = "  split  decode   " + "".join(
            f"B={b:<6}" for b in sorted(sim["chunk_sizes"])
        )
        print(header)
        for split in summary["config"]["data"]["split"]:
            for portion in sorted(float(f) for f in sim["decode_portions"]):
                cells = []
                for size in sorted(int(b) for b in sim["chunk_sizes"]):
                    key = f"{split}|{portion}|{size}|{ratio}|{policy}|{fetch_value}"
                    stats = summary["combos"].get(key)
                    cells.append(
                        f"{stats['mean_missing']:<8.3f}" if stats else f"{'-':<8}"
                    )
                print(f"  {split:<5}  {int(portion * 100):>3}%    " + "".join(cells))

    mean_table("static", f"static, fetch={fetch}", fetch)
    if "cached" in summary.get("policies", []):
        mean_table("cached", "adaptive cache (oracle)", 0)

    print("\ndispersal (distinct sessions/chunk, min within-session gap):")
    for key, stats in summary["diagnostics"].items():
        print(
            f"  {key:<28} sessions={stats['distinct_sessions_per_chunk']:.2f} "
            f"gap={stats['min_within_session_gap']:.0f} "
            f"decode={stats['realized_decode_portion']:.2f}"
        )

    print(f"\nartifacts mirrored to {out_dir}")
    print(
        "volume artifacts: "
        f"modal volume get deepseek-v4-flash-simulation {run_id}/"
    )
