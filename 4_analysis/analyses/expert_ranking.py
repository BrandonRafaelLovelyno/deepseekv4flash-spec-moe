"""Study: rank experts by training frequency and build the resident sets.

Pass A of the original analysis. Counts how often each expert is used across
every training token (independently per layer), then derives the hottest-k
resident set for each preservation ratio. Every later study reads the resulting
``ctx.counts`` / ``ctx.k_by_ratio`` / ``ctx.masks``.

Artifacts (in ``<run>/expert_ranking/``):
    * ``train_expert_counts.npy`` -- ``[n_layers, n_experts]`` training counts
    * ``hot_experts.csv``         -- layer,ratio,rank,expert_id,train_count,share
    * ``summary.json``, ``run.log``
"""

from __future__ import annotations

import os
from typing import Iterator

from analyses.base import Analysis
from helper import (
    RATIOS,
    AnalysisContext,
    Event,
    _emit,
    _expert_counts,
    _file_event,
    _resident_masks,
    _utc_now,
    _write_json,
    _write_run_log,
)


def _write_hot_experts_csv(
    path: str, counts, k_by_ratio: dict[float, int], n_layers: int
) -> None:
    """Write the per-layer hottest experts for every ratio."""
    import numpy as np

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("layer,ratio,rank,expert_id,train_count,share\n")
        total_per_layer = counts.sum(axis=1, keepdims=True)
        for ratio in RATIOS:
            k = k_by_ratio[ratio]
            for layer in range(n_layers):
                order = np.argsort(counts[layer])[::-1][:k]
                for rank, expert in enumerate(order):
                    share = counts[layer, expert] / max(total_per_layer[layer, 0], 1)
                    handle.write(
                        f"{layer},{ratio},{rank},{expert},"
                        f"{int(counts[layer, expert])},{share:.8f}\n"
                    )


class ExpertRankingAnalysis(Analysis):
    name = "expert_ranking"

    def run(self, ctx: AnalysisContext) -> Iterator[Event]:
        logs: list[str] = []
        out_dir = ctx.out_dir(self.name)

        counts = _expert_counts(ctx.train, ctx.n_layers, ctx.n_experts)
        ctx.counts = counts
        yield _emit(
            logs,
            f"ranked experts over {len(ctx.train)} train slices; "
            f"hottest expert used {int(counts.max())} times",
        )

        k_by_ratio, masks = _resident_masks(counts, ctx.n_experts)
        ctx.k_by_ratio = k_by_ratio
        ctx.masks = masks
        for ratio in RATIOS:
            yield _emit(
                logs,
                f"keep {int(ratio * 100)}%: top {k_by_ratio[ratio]} experts/layer resident",
            )

        import numpy as np

        np.save(os.path.join(out_dir, "train_expert_counts.npy"), counts)
        _write_hot_experts_csv(
            os.path.join(out_dir, "hot_experts.csv"), counts, k_by_ratio, ctx.n_layers
        )

        summary = {
            "run_id": ctx.run_id,
            "created_at": _utc_now(),
            "quick": ctx.quick,
            "ratios": list(RATIOS),
            "k_by_ratio": {str(r): k_by_ratio[r] for r in RATIOS},
            "n_layers": ctx.n_layers,
            "n_experts": ctx.n_experts,
            "hottest_count": int(counts.max()),
        }
        _write_json(os.path.join(out_dir, "summary.json"), summary)

        for name in ("train_expert_counts.npy", "hot_experts.csv", "summary.json"):
            yield _file_event(os.path.join(out_dir, name), f"{self.name}/{name}")
        yield _file_event(_write_run_log(out_dir, logs), f"{self.name}/run.log")
