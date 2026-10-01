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
        f"{split}|{portion}|{chunk_size}|{ratio}|{fetch}": counts
        for (split, portion, chunk_size, ratio, fetch), counts in histograms.items()
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

    print(f"\nmean missing experts per chunk (fetch={fetch}, keep {int(ratio * 100)}%):")
    header = "  split  decode   " + "".join(f"B={b:<6}" for b in sorted(sim["chunk_sizes"]))
    print(header)
    for split in summary["config"]["data"]["split"]:
        for portion in sorted(float(f) for f in sim["decode_portions"]):
            cells = []
            for size in sorted(int(b) for b in sim["chunk_sizes"]):
                key = f"{split}|{portion}|{size}|{ratio}|{fetch}"
                stats = summary["combos"].get(key)
                cells.append(f"{stats['mean_missing']:<8.3f}" if stats else f"{'-':<8}")
            print(f"  {split:<5}  {int(portion * 100):>3}%    " + "".join(cells))

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
