"""Study: coverage of the non-resident expert set (decode vs prefill).

Reads the ``static`` policy results the ``decode_miss`` and ``prefill_miss``
studies wrote into ``ctx.cache_batches`` / ``ctx.cache_prefills`` (train split,
matching the non-adaptive resident set) and compares how much of the non-resident
per-layer expert set is demanded as the batch grows. For each ratio
``cap = n_experts - k``; coverage is the mean missing count divided by ``cap``.

Artifacts (in ``<run>/coverage/``):
    * ``06_coverage.png`` -- decode vs prefill coverage, one line pair per ratio
    * ``summary.json``, ``run.log``
"""

from __future__ import annotations

import os
from typing import Iterator

from analyses.base import Analysis
from helper import (
    BATCH_SIZES,
    COLORS,
    RATIOS,
    AnalysisContext,
    Event,
    _emit,
    _figure_bytes,
    _file_event,
    _image_event,
    _pyplot,
    _utc_now,
    _write_json,
    _write_run_log,
)


def _coverage(cache_batches, cache_prefills, k_by_ratio, n_experts, split="train"):
    """Compute the decode and prefill coverage curves for every ratio.

    Reads the ``static`` policy curves the decode/prefill studies wrote for
    ``split`` (``train`` by default, matching the non-adaptive resident set).

    Returns ``(batch_sizes, coverage)`` where ``coverage[ratio]`` is
    ``{"decode": [...], "prefill": [...]}``.
    """
    import numpy as np

    batch_sizes = [
        batch
        for batch in BATCH_SIZES
        if (split, batch) in cache_batches and (split, batch) in cache_prefills
    ]
    coverage: dict[float, dict[str, list[float]]] = {}
    for ratio in RATIOS:
        cap = max(n_experts - k_by_ratio[ratio], 1)
        decode_coverage = []
        prefill_coverage = []
        for batch in batch_sizes:
            decode_distribution = cache_batches[(split, batch)]["static"][ratio]
            decode_missing = np.arange(decode_distribution.shape[0])
            decode_coverage.append(
                float((decode_distribution * decode_missing).sum()) / cap
            )
            prefill_distribution = cache_prefills[(split, batch)]["static"][ratio]
            prefill_missing = np.arange(prefill_distribution.shape[0])
            prefill_coverage.append(
                float((prefill_distribution * prefill_missing).sum()) / cap
            )
        coverage[ratio] = {"decode": decode_coverage, "prefill": prefill_coverage}
    return batch_sizes, coverage


def _fig_coverage(batch_sizes, coverage) -> bytes:
    """Figure 06: decode vs prefill coverage of the non-resident set."""
    plt = _pyplot()
    fig6, ax6 = plt.subplots(figsize=(7.5, 4.5))
    for ratio in RATIOS:
        ax6.plot(
            batch_sizes,
            coverage[ratio]["decode"],
            "-o",
            color=COLORS[ratio],
            label=f"decode keep {int(ratio * 100)}%",
        )
        ax6.plot(
            batch_sizes,
            coverage[ratio]["prefill"],
            "--s",
            color=COLORS[ratio],
            label=f"prefill keep {int(ratio * 100)}%",
        )
    ax6.set_xlabel("batch size B")
    ax6.set_ylabel("mean fraction of non-resident experts demanded")
    ax6.set_title("Coverage of the non-resident expert set per layer")
    ax6.grid(alpha=0.3)
    ax6.legend(fontsize=7, ncol=2)
    return _figure_bytes(fig6)


class CoverageAnalysis(Analysis):
    name = "coverage"

    def run(self, ctx: AnalysisContext) -> Iterator[Event]:
        logs: list[str] = []
        out_dir = ctx.out_dir(self.name)

        batch_sizes, coverage = _coverage(
            ctx.cache_batches, ctx.cache_prefills, ctx.k_by_ratio, ctx.n_experts
        )
        if not batch_sizes:
            yield _emit(logs, "no decode/prefill results to compare")
            return
        yield _emit(logs, f"compared decode vs prefill coverage over B={batch_sizes[0]}..{batch_sizes[-1]}")

        yield _image_event(
            f"{self.name}/06_coverage.png",
            _fig_coverage(batch_sizes, coverage),
        )

        summary = {
            "run_id": ctx.run_id,
            "created_at": _utc_now(),
            "quick": ctx.quick,
            "ratios": list(RATIOS),
            "batch_sizes": batch_sizes,
            "coverage": {
                str(r): {
                    "batch_sizes": batch_sizes,
                    "decode": coverage[r]["decode"],
                    "prefill": coverage[r]["prefill"],
                }
                for r in RATIOS
            },
        }
        _write_json(os.path.join(out_dir, "summary.json"), summary)

        yield _file_event(
            os.path.join(out_dir, "summary.json"), f"{self.name}/summary.json"
        )
        yield _file_event(_write_run_log(out_dir, logs), f"{self.name}/run.log")
