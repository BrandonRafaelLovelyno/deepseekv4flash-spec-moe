"""Artifact IO and end-of-run reporting for the average-used-experts run.

Standard library at module scope, so the local entrypoint can mirror a finished
run without importing numpy / matplotlib. Remote scope writes the run into its
volume directory (config, summary, per-layer table, histogram archive and
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


CSV_COLUMNS = [
    "split",
    "decode_portion",
    "chunk_size",
    "layer",
    "n_chunks",
    "mean_tokens",
    "mean_distinct",
    "p50",
    "p90",
    "max",
    "distinct_fraction",
    "ceiling",
]


def write_per_layer_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Write the per-``(split, portion, chunk_size, layer)`` table as CSV."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(CSV_COLUMNS) + "\n")
        handle.writelines(
            ",".join(str(row[column]) for column in CSV_COLUMNS) + "\n"
            for row in rows
        )


def _write_histograms(path: str, histograms: dict) -> None:
    """Archive every per-layer distinct-count histogram as a flat ``.npz``."""
    import numpy as np

    np.savez_compressed(path, **histograms)


def write_run(
    out_dir: str,
    config_text: str,
    payload: dict[str, Any],
    figures: dict[str, bytes],
) -> list[str]:
    """Write a finished run into its volume directory; return the names."""
    os.makedirs(out_dir, exist_ok=True)

    _write_text(os.path.join(out_dir, "config.yaml"), config_text)
    _write_json(os.path.join(out_dir, "summary.json"), payload["summary"])
    write_per_layer_csv(os.path.join(out_dir, "per_layer.csv"), payload["rows"])
    _write_histograms(os.path.join(out_dir, "histograms.npz"), payload["histograms"])

    written = ["config.yaml", "summary.json", "per_layer.csv", "histograms.npz"]
    for name, data in figures.items():
        _write_bytes(os.path.join(out_dir, name), data)
        written.append(name)
    return written


def mirror_run(
    root: str,
    run_id: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    figures: dict[str, bytes],
) -> str:
    """Mirror the JSON/CSV/PNG subset into ``<root>/<run_id>/`` locally."""
    out_dir = os.path.join(root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    write_per_layer_csv(os.path.join(out_dir, "per_layer.csv"), rows)
    for name, data in figures.items():
        _write_bytes(os.path.join(out_dir, name), data)
    return out_dir


def print_report(_run_id: str, summary: dict[str, Any], out_dir: str) -> None:
    """Print mean distinct experts per chunk, aggregated across layers."""
    print(
        f"\nlayers: {summary['n_layers']} "
        f"(missing: {summary['missing_layers'] or 'none'})  "
        f"experts: {summary['n_experts']}  top_k: {summary['top_k']}"
    )

    portions = sorted({float(key.split("|")[1]) for key in summary["combos"]})
    chunk_sizes = summary["chunk_sizes"]
    header = "  split  decode   " + "".join(f"B={b:<6}" for b in chunk_sizes)
    for portion in portions:
        print(f"\nmean distinct experts per chunk (layer mean), decode={portion:.3g}:")
        print(header)
        for split in summary["config"]["data"]["split"]:
            cells = []
            for size in chunk_sizes:
                stats = summary["combos"].get(f"{split}|{portion}|{size}")
                cells.append(
                    f"{stats['layer_mean_distinct']:<8.2f}" if stats else f"{'-':<8}"
                )
            print(f"  {split:<5}  {int(portion * 100):>3}%    " + "".join(cells))

    print(f"\nartifacts mirrored to {out_dir}")
    print("volume artifacts: modal volume get deepseek-v4-flash-article <run_id>/")
