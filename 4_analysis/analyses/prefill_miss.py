"""Study: concurrent-prefill unique missing experts.

Pass D of the original analysis. Assumes an incremental, KV-cached server: a
turn's prefill is only its new non-assistant token phase (system + tools
template, user turns, tool results), and with no chunking a phase is one
forward pass. Phase jobs are pulled round-robin by turn index across sessions
and packed B-wide into events; per event and layer the missing count is the
number of distinct demanded experts absent from the resident set (bounded by
``n_experts - k``). A per-B token budget caps the corpus replayed.

Artifacts (in ``<run>/prefill_miss/``):
    * ``prefill_distributions.csv``            -- batch,ratio,missing,portion
    * ``05_prefill_missing_distribution.png``  -- 4x5 grid (B=1..20)
    * ``summary.json``, ``run.log``
"""

from __future__ import annotations

import os
from typing import Iterator

from analyses.base import Analysis
from helper import (
    BATCH_SIZES,
    PREFILL_TOKEN_BUDGET,
    RATIOS,
    AnalysisContext,
    Curves,
    Event,
    PrefillMetaEntry,
    _batch_grid_figure,
    _dist_stats,
    _emit,
    _file_event,
    _image_event,
    _prefill_phases,
    _save_curve_csv,
    _utc_now,
    _write_json,
    _write_run_log,
)


def _prefill_batches(
    datasets: list[str],
    by_dataset,
    masks,
    k_by_ratio: dict[float, int],
    n_layers: int,
    n_experts: int,
    logs: list[str],
) -> Iterator[Event]:
    """Replay concurrent prefill events, one log event per batch size.

    Returns ``(prefills, prefill_meta)``.
    """
    import numpy as np

    prefill_cache: dict[str, list] = {}

    def prefill_phases(record):
        path = record["path"]
        if path not in prefill_cache:
            prefill_cache[path] = _prefill_phases(path)
        return prefill_cache[path]

    session_records = [
        record for dataset in datasets for record in by_dataset[dataset]
    ]
    phases_by_session = [prefill_phases(record) for record in session_records]
    max_turns = max((len(phases) for phases in phases_by_session), default=0)
    queue = [
        phases_by_session[s][turn]
        for turn in range(max_turns)
        for s in range(len(phases_by_session))
        if turn < len(phases_by_session[s])
    ]

    prefills: dict[int, Curves] = {}
    prefill_meta: dict[int, PrefillMetaEntry] = {}
    for batch in BATCH_SIZES:
        caps = {ratio: n_experts - k_by_ratio[ratio] for ratio in RATIOS}
        hist = {
            ratio: np.zeros((n_layers, caps[ratio] + 1), dtype=np.int64)
            for ratio in RATIOS
        }
        coverage_sum = {ratio: 0.0 for ratio in RATIOS}
        coverage_n = 0
        n_events = 0
        tokens_used = 0
        for start in range(0, len(queue), batch):
            jobs = queue[start:start + batch]
            event_tokens = sum(job.shape[0] for job in jobs)
            if n_events and tokens_used + event_tokens > PREFILL_TOKEN_BUDGET:
                break
            event_ids = np.concatenate(jobs, axis=0)
            for layer in range(n_layers):
                present = (
                    np.bincount(
                        event_ids[:, layer, :].astype(np.int64).ravel(),
                        minlength=n_experts,
                    )
                    > 0
                )
                for ratio in RATIOS:
                    missing = int((present & ~masks[ratio][layer]).sum())
                    hist[ratio][layer, missing] += 1
                    coverage_sum[ratio] += missing / max(caps[ratio], 1)
                    if ratio == RATIOS[0]:
                        coverage_n += 1
            n_events += 1
            tokens_used += event_tokens
            if tokens_used >= PREFILL_TOKEN_BUDGET:
                break
        prefills[batch] = {
            ratio: hist[ratio].sum(axis=0) / max(n_events * n_layers, 1)
            for ratio in RATIOS
        }
        prefill_meta[batch] = {
            "n_events": n_events,
            "tokens_used": tokens_used,
            "mean_coverage": {
                ratio: coverage_sum[ratio] / max(coverage_n, 1) for ratio in RATIOS
            },
        }
        stats = _dist_stats(prefills[batch][RATIOS[-1]])
        yield _emit(
            logs,
            f"prefill batch {batch:>2}: {n_events} events, {tokens_used} tokens, "
            f"keep 75% mean missing {stats['mean_missing']:.3f}, "
            f"coverage {prefill_meta[batch]['mean_coverage'][RATIOS[-1]]:.3f}",
        )
    return prefills, prefill_meta


def _fig_prefill_missing(
    prefills: dict[int, Curves], k_by_ratio: dict[float, int], n_experts: int
) -> bytes:
    """Figure 05: 4x5 grid of per-batch prefill miss distributions."""
    import numpy as np

    panels = [
        (
            f"batch B={batch}",
            prefills[batch],
            {
                ratio: np.arange(n_experts - k_by_ratio[ratio] + 1)
                for ratio in RATIOS
            },
            None,
        )
        for batch in BATCH_SIZES
    ]
    return _batch_grid_figure(
        panels,
        "portion of prefill events",
        "Concurrent prefill: unique missing experts per layer, x capped at 256-k",
        markersize=2.5,
    )


class PrefillMissAnalysis(Analysis):
    name = "prefill_miss"

    def run(self, ctx: AnalysisContext) -> Iterator[Event]:
        logs: list[str] = []
        out_dir = ctx.out_dir(self.name)

        prefills, prefill_meta = yield from _prefill_batches(
            ctx.datasets,
            ctx.by_dataset,
            ctx.masks,
            ctx.k_by_ratio,
            ctx.n_layers,
            ctx.n_experts,
            logs,
        )
        ctx.prefills = prefills
        ctx.prefill_meta = prefill_meta

        _save_curve_csv(os.path.join(out_dir, "prefill_distributions.csv"), prefills)
        yield _image_event(
            f"{self.name}/05_prefill_missing_distribution.png",
            _fig_prefill_missing(prefills, ctx.k_by_ratio, ctx.n_experts),
        )

        summary = {
            "run_id": ctx.run_id,
            "created_at": _utc_now(),
            "quick": ctx.quick,
            "ratios": list(RATIOS),
            "prefills": {
                str(batch): {
                    "n_events": prefill_meta[batch]["n_events"],
                    "tokens_used": prefill_meta[batch]["tokens_used"],
                    "mean_coverage": {
                        str(r): round(prefill_meta[batch]["mean_coverage"][r], 6)
                        for r in RATIOS
                    },
                    "distributions": {
                        str(r): _dist_stats(prefills[batch][r]) for r in RATIOS
                    },
                }
                for batch in BATCH_SIZES
            },
        }
        _write_json(os.path.join(out_dir, "summary.json"), summary)

        for name in ("prefill_distributions.csv", "summary.json"):
            yield _file_event(os.path.join(out_dir, name), f"{self.name}/{name}")
        yield _file_event(_write_run_log(out_dir, logs), f"{self.name}/run.log")
