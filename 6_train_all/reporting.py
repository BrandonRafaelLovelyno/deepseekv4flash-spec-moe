"""Artifact IO and end-of-run reporting for a train-all run.

Everything here is standard library at module scope, so the local entrypoint can
mirror a finished run without importing torch / numpy / matplotlib. Two scopes:

* per layer -- the GPU trainer writes one checkpoint + summary + figures under
  ``<run_id>/L{dd>/`` with :func:`write_layer_artifacts`;
* per run -- :func:`write_aggregate` writes ``summary.json`` / ``layers.csv`` /
  ``recall_by_layer.png``, :func:`mirror_run` copies them locally, and
  :func:`print_run_report` reports.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

BASE_COLUMNS = [
    "layer",
    "source_layer",
    "distance",
    "best_epoch",
    "best_eval_kl",
    "final_kl",
]


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_bytes(path: str, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_history_csv(
    path: str, history: list[dict], ks: list[int], ratios: list[float]
) -> None:
    """Write one layer's per-epoch history, leaving skipped eval cells empty."""
    columns = (
        ["epoch", "train_kl", "eval_kl"]
        + [f"recall@{k}" for k in ks]
        + ["lr", "seconds"]
        + [f"ready_recall_r{int(round(r * 100))}@{k}" for r in ratios for k in ks]
        + [f"resident_recall_r{int(round(r * 100))}" for r in ratios]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for entry in history:
            cells = [_format_cell(entry.get(column, "")) for column in columns]
            handle.write(",".join(cells) + "\n")


def row_from_summary(summary: dict) -> dict:
    """Flatten one layer's summary into a single aggregate-table row."""
    row = {
        "layer": summary["layer"],
        "source_layer": summary["source_layer"],
        "distance": summary["distance"],
        "best_epoch": summary["best_epoch"],
        "best_eval_kl": summary["best_eval_kl"],
        "final_kl": summary["final_metrics"]["kl"],
    }
    for name, value in summary["final_metrics"].items():
        if name != "kl":
            row[name] = value
    return row


def write_layer_artifacts(
    layer_dir: str,
    config_text: str,
    summary: dict,
    hot_experts: dict,
    curves_png: bytes,
    ready_png: bytes,
) -> None:
    """Write one layer's checkpoint-adjacent artifacts and figures."""
    os.makedirs(layer_dir, exist_ok=True)
    _write_json(os.path.join(layer_dir, "metrics.json"), summary)
    _write_text(os.path.join(layer_dir, "config.yaml"), config_text)
    _write_json(os.path.join(layer_dir, "hot_experts.json"), hot_experts)
    _write_bytes(os.path.join(layer_dir, "curves.png"), curves_png)
    _write_bytes(os.path.join(layer_dir, "ready_recall.png"), ready_png)
    cfg = summary["config"]
    write_history_csv(
        os.path.join(layer_dir, "history.csv"),
        summary["history"],
        [int(k) for k in cfg["eval"]["ks"]],
        [float(r) for r in cfg["eval"]["ready_ratios"]],
    )


def write_layers_csv(path: str, rows: list[dict]) -> None:
    """Write the per-layer aggregate table, columns in a stable order."""
    extra = sorted(
        {key for row in rows for key in row} - set(BASE_COLUMNS)
    )
    columns = BASE_COLUMNS + extra
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in sorted(rows, key=lambda row: row["layer"]):
            cells = [_format_cell(row.get(column)) for column in columns]
            handle.write(",".join(cells) + "\n")


def build_layers_summary(run_id: str, cfg: dict, rows: list[dict]) -> dict:
    """Assemble the run-level ``summary.json`` payload."""
    return {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_layers": len(rows),
        "config": cfg,
        "layers": sorted(rows, key=lambda row: row["layer"]),
    }


def write_aggregate(
    out_dir: str, run_id: str, cfg: dict, rows: list[dict], png: bytes
) -> dict:
    """Write the run-level summary, table and figure; return the summary."""
    summary = build_layers_summary(run_id, cfg, rows)
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    write_layers_csv(os.path.join(out_dir, "layers.csv"), rows)
    _write_bytes(os.path.join(out_dir, "recall_by_layer.png"), png)
    return summary


def mirror_run(
    root: str, run_id: str, summary: dict, rows: list[dict], png: bytes
) -> str:
    """Mirror the run-level artifacts into ``<root>/<run_id>/`` locally."""
    out_dir = os.path.join(root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    write_layers_csv(os.path.join(out_dir, "layers.csv"), rows)
    _write_bytes(os.path.join(out_dir, "recall_by_layer.png"), png)
    return out_dir


def print_cache_summary(built: dict) -> None:
    """Report the cache build/reuse summary from ``build_cache``."""
    print(
        f"cache: reset={built['reset']} built={len(built['built'])} "
        f"entries={built['n_entries']}"
    )


def _ranked(rows: list[dict], key: str) -> list[dict]:
    return [row for row in rows if row.get(key) is not None]


def print_run_report(
    run_id: str,
    rows: list[dict],
    out_dir: str,
    ks: list[int],
    failures: list[str] = (),
) -> None:
    """Print the run-level summary: coverage, recall means, best/worst layer."""
    print(f"\nrun_id: {run_id}  layers: {len(rows)}")
    if failures:
        print(f"failed layers: {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
    for k in ks:
        values = [row[f"recall@{k}"] for row in rows if f"recall@{k}" in row]
        if values:
            print(f"mean recall@{k}: {sum(values) / len(values):.4f}")

    by_kl = sorted(_ranked(rows, "best_eval_kl"), key=lambda row: row["best_eval_kl"])
    if by_kl:
        best, worst = by_kl[0], by_kl[-1]
        print(
            f"best layer: L{best['layer']:02d} "
            f"(eval_kl={best['best_eval_kl']}) | "
            f"worst: L{worst['layer']:02d} (eval_kl={worst['best_eval_kl']})"
        )
    print(f"artifacts mirrored to {out_dir}")
    print(
        "checkpoints on volume: "
        f"modal volume get deepseek-v4-flash-training {run_id}/ "
        "--include L*/checkpoint.safetensors"
    )
