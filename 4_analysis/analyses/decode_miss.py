"""Study: concurrent-decode unique missing experts.

Pass C of the original analysis. For each batch size B in 1..20, picks B
training decode sessions round-robin across datasets (seeded), replays their
decode steps concurrently, and counts, per step and layer, the number of
**distinct** demanded experts absent from the resident set (fetch-once, so
duplicates collapse; in 0..top_k*B). A step still counts once some sessions
finish decoding.

Artifacts (in ``<run>/decode_miss/``):
    * ``batch_distributions.csv``            -- batch,ratio,missing,portion
    * ``04_batch_missing_distribution.png``  -- 4x5 grid (B=1..20)
    * ``summary.json``, ``run.log``
"""

from __future__ import annotations

import os
from typing import Iterator

from analyses.base import Analysis
from helper import (
    BATCH_SIZES,
    RATIOS,
    AnalysisContext,
    Curves,
    Event,
    _batch_grid_figure,
    _decode_ids,
    _dist_stats,
    _emit,
    _file_event,
    _image_event,
    _save_curve_csv,
    _utc_now,
    _write_json,
    _write_run_log,
)


def _decode_batches(
    datasets: list[str],
    by_dataset,
    masks,
    n_layers: int,
    n_experts: int,
    top_k: int,
    logs: list[str],
) -> Iterator[Event]:
    """Replay concurrent decode batches, one log event per batch size.

    Returns the ``{batch: curves}`` mapping.
    """
    import numpy as np

    def session_for(slot: int):
        dataset = datasets[slot % len(datasets)]
        pool = by_dataset[dataset]
        return pool[(slot // len(datasets)) % len(pool)]

    decode_cache: dict[str, "np.ndarray"] = {}

    def decode_ids(record):
        path = record["path"]
        if path not in decode_cache:
            decode_cache[path] = _decode_ids(path)
        return decode_cache[path]

    batches: dict[int, Curves] = {}
    for batch in BATCH_SIZES:
        sessions = [decode_ids(session_for(slot)) for slot in range(batch)]
        n_steps = max(s.shape[0] for s in sessions)
        max_missing = top_k * batch
        hist = {
            ratio: np.zeros((n_layers, max_missing + 1), dtype=np.int64)
            for ratio in RATIOS
        }
        for layer in range(n_layers):
            demand = np.zeros((n_steps, n_experts), dtype=bool)
            for ids in sessions:
                rows = np.arange(ids.shape[0])
                demand[rows[:, None], ids[:, layer, :]] = True
            demanded = demand.sum(axis=1)
            for ratio in RATIOS:
                resident = masks[ratio][layer]
                missing = demanded - (demand & resident).sum(axis=1)
                hist[ratio][layer] += np.bincount(missing, minlength=max_missing + 1)
        batches[batch] = {
            ratio: hist[ratio].sum(axis=0) / (n_steps * n_layers) for ratio in RATIOS
        }
        stats = _dist_stats(batches[batch][RATIOS[-1]])
        yield _emit(
            logs,
            f"batch {batch:>2}: {n_steps} decode steps, up to {max_missing} missing; "
            f"keep 75% mean missing {stats['mean_missing']:.3f}",
        )
    return batches


def _fig_batch_missing(batches: dict[int, Curves], top_k: int) -> bytes:
    """Figure 04: 4x5 grid of per-batch decode miss distributions."""
    import numpy as np

    panels = [
        (
            f"batch B={batch}",
            batches[batch],
            {ratio: np.arange(top_k * batch + 1) for ratio in RATIOS},
            (0, top_k * batch),
        )
        for batch in BATCH_SIZES
    ]
    return _batch_grid_figure(
        panels,
        "portion of decode steps",
        "Concurrent decode: unique missing experts per layer (averaged over 43 layers)",
    )


class DecodeMissAnalysis(Analysis):
    name = "decode_miss"

    def run(self, ctx: AnalysisContext) -> Iterator[Event]:
        logs: list[str] = []
        out_dir = ctx.out_dir(self.name)

        batches = yield from _decode_batches(
            ctx.datasets,
            ctx.by_dataset,
            ctx.masks,
            ctx.n_layers,
            ctx.n_experts,
            ctx.top_k,
            logs,
        )
        ctx.batches = batches

        _save_curve_csv(os.path.join(out_dir, "batch_distributions.csv"), batches)
        yield _image_event(
            f"{self.name}/04_batch_missing_distribution.png",
            _fig_batch_missing(batches, ctx.top_k),
        )

        summary = {
            "run_id": ctx.run_id,
            "created_at": _utc_now(),
            "quick": ctx.quick,
            "ratios": list(RATIOS),
            "batches": {
                str(batch): {str(r): _dist_stats(batches[batch][r]) for r in RATIOS}
                for batch in BATCH_SIZES
            },
        }
        _write_json(os.path.join(out_dir, "summary.json"), summary)

        for name in ("batch_distributions.csv", "summary.json"):
            yield _file_event(os.path.join(out_dir, name), f"{self.name}/{name}")
        yield _file_event(_write_run_log(out_dir, logs), f"{self.name}/run.log")
