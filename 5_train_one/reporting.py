"""Artifact IO and end-of-run reporting for a single training variant.

Everything here is standard library at module scope, so the local entrypoint can
mirror a finished run without importing torch / numpy / matplotlib. Two callers:

* the GPU trainer writes its summary artifacts onto the training volume with
  :func:`write_run_artifacts`;
* the local entrypoint mirrors the returned bytes/logs into
  ``5_train_one/output/<run_id>/`` with :func:`mirror_result` and reports with
  :func:`print_run_report`.
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


def write_history_csv(
    path: str, history: list[dict], ks: list[int], ratios: list[float]
) -> None:
    """Write the per-epoch history to CSV, leaving skipped eval cells empty."""
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
            cells = []
            for column in columns:
                value = entry.get(column, "")
                cells.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            handle.write(",".join(cells) + "\n")


def write_run_artifacts(
    out_dir: str, config_text: str, summary: dict, hot_experts: dict
) -> None:
    """Write the on-volume artifact set: metrics, config, hot experts."""
    _write_json(os.path.join(out_dir, "metrics.json"), summary)
    _write_text(os.path.join(out_dir, "config.yaml"), config_text)
    _write_json(os.path.join(out_dir, "hot_experts.json"), hot_experts)


def mirror_result(root: str, run_id: str, config_text: str, result: dict) -> str:
    """Mirror a finished run's artifacts+figures under ``<root>/<run_id>/``."""
    out_dir = os.path.join(root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    summary = result["summary"]
    write_run_artifacts(out_dir, config_text, summary, result["hot_experts"])
    _write_bytes(os.path.join(out_dir, "curves.png"), result["png"])
    _write_bytes(os.path.join(out_dir, "ready_recall.png"), result["ready_png"])
    write_history_csv(
        os.path.join(out_dir, "history.csv"),
        summary["history"],
        [int(k) for k in summary["config"]["eval"]["ks"]],
        [float(r) for r in summary["config"]["eval"]["ready_ratios"]],
    )
    return out_dir


def print_cache_summary(built: dict) -> None:
    """Report the cache build/reuse summary from ``build_cache``."""
    print(
        f"cache: reset={built['reset']} built={len(built['built'])} "
        f"entries={built['n_entries']}"
    )


def print_run_report(result: dict, out_dir: str) -> None:
    """Print the human-readable end-of-run summary."""
    summary = result["summary"]
    print(f"\nrow_id: {result['row_id']}")
    print(
        f"train tokens: {summary['n_train_tokens']} | "
        f"eval tokens: {summary['n_eval_tokens']}"
    )
    print(
        f"best epoch: {summary['best_epoch']} (eval_kl={summary['best_eval_kl']}) | "
        f"final: {summary['final_metrics']}"
    )
    print(f"artifacts mirrored to {out_dir}")
    print(
        "checkpoint on volume: "
        f"modal volume get deepseek-v4-flash-training "
        f"{result['run_id']}/checkpoint.safetensors"
    )
