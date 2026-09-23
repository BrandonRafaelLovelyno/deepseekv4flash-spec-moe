"""Study: concurrent-prefill unique missing experts with an adaptive cache.

For every batch size ``B`` and split, a fresh ``AdaptiveCache`` is seeded from
the training-hot set, then prefill phases of that split are pulled round-robin by
turn index and packed ``B``-wide into events. Each event measures the same
observation twice -- against the fixed ``masks`` (static) and the evolving
resident set (cached) -- and the cache then updates from the event's union of
demanded experts. Train and test are independent; replay is capped at
``PREFILL_TOKEN_BUDGET`` tokens per ``(split, B)``.

Artifacts (in ``<run>/prefill_miss/``):
    * ``distributions.csv``              -- split,batch,policy,ratio,missing,portion
    * ``05_static_vs_cached_prefill.png`` -- 4x5 grid per split, static vs cached
    * ``09_prefill_cdf.png``             -- same grids as CDFs (P(missing <= m))
    * ``10_cache_traffic.png``           -- loads/evictions vs B, per split
    * ``summary.json``, ``run.log``
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import numpy as np

from analyses.base import Analysis
from helper import (
    BATCH_SIZES,
    COLORS,
    PREFILL_TOKEN_BUDGET,
    RATIOS,
    SPLITS,
    AdaptiveCache,
    AnalysisContext,
    Curves,
    Event,
    _cdf,
    _compare_grid_figure,
    _dist_stats,
    _emit,
    _figure_bytes,
    _file_event,
    _image_event,
    _prefill_phases,
    _pyplot,
    _utc_now,
    _write_json,
    _write_rows,
    _write_run_log,
)

CacheCurves = dict[str, Curves]


def _queue_for(pools, datasets: list[str], phases) -> list["np.ndarray"]:
    """Flatten a split's sessions and order phase jobs round-robin by turn index."""
    session_records = [record for ds in datasets for record in pools[ds]]
    phases_by_session = [phases(record) for record in session_records]
    max_turns = max((len(p) for p in phases_by_session), default=0)
    return [
        phases_by_session[s][turn]
        for turn in range(max_turns)
        for s in range(len(phases_by_session))
        if turn < len(phases_by_session[s])
    ]


def _prefill_cache_batches(
    datasets: list[str],
    pools_by_split: dict[str, dict],
    masks,
    counts,
    k_by_ratio: dict[float, int],
    n_layers: int,
    n_experts: int,
    logs: list[str],
) -> Iterator[Event]:
    """Replay every ``(split, B)``; return ``(curves, traffic)``."""
    import numpy as np

    phase_cache: dict[str, list] = {}

    def phases(record):
        path = record["path"]
        if path not in phase_cache:
            phase_cache[path] = _prefill_phases(path)
        return phase_cache[path]

    layer_axis = np.arange(n_layers)
    layer_index = np.arange(n_layers)[None, :, None]
    masks_stack = np.stack([masks[ratio] for ratio in RATIOS])  # [R, L, E]
    out: dict[tuple[str, int], CacheCurves] = {}
    traffic: dict[str, dict[int, dict]] = {}
    for split in SPLITS:
        pools = pools_by_split.get(split, {})
        split_datasets = [d for d in datasets if pools.get(d)]
        if not split_datasets:
            continue
        queue = _queue_for(pools, split_datasets, phases)
        split_traffic: dict[int, dict] = {}
        for batch in BATCH_SIZES:
            cache = AdaptiveCache(masks, counts, k_by_ratio, n_layers, n_experts)
            caps = {ratio: n_experts - k_by_ratio[ratio] for ratio in RATIOS}
            hist_static = {
                ratio: np.zeros((n_layers, caps[ratio] + 1), dtype=np.int64)
                for ratio in RATIOS
            }
            hist_cached = {
                ratio: np.zeros((n_layers, caps[ratio] + 1), dtype=np.int64)
                for ratio in RATIOS
            }
            n_events = 0
            tokens_used = 0
            for start in range(0, len(queue), batch):
                jobs = queue[start:start + batch]
                event_tokens = sum(job.shape[0] for job in jobs)
                if n_events and tokens_used + event_tokens > PREFILL_TOKEN_BUDGET:
                    break
                event_ids = np.concatenate(jobs, axis=0)
                demand = np.zeros((n_layers, n_experts), dtype=bool)
                demand[layer_index, event_ids] = True
                static_miss = (demand[None, :, :] & ~masks_stack).sum(axis=2)  # [R, L]
                cached_miss = cache.observe_mask_all(demand)  # [R, L]
                for ri, ratio in enumerate(RATIOS):
                    hist_static[ratio][layer_axis, static_miss[ri]] += 1
                    hist_cached[ratio][layer_axis, cached_miss[ri]] += 1
                n_events += 1
                tokens_used += event_tokens
                if tokens_used >= PREFILL_TOKEN_BUDGET:
                    break
            denom = max(n_events * n_layers, 1)
            out[(split, batch)] = {
                "static": {r: hist_static[r].sum(axis=0) / denom for r in RATIOS},
                "cached": {r: hist_cached[r].sum(axis=0) / denom for r in RATIOS},
            }
            split_traffic[batch] = cache.traffic()
            stats = _dist_stats(out[(split, batch)]["cached"][RATIOS[-1]])
            yield _emit(
                logs,
                f"{split:5} batch {batch:>2}: {n_events} events, {tokens_used} tokens, "
                f"keep 75% mean missing {stats['mean_missing']:.3f}, "
                f"loads {cache.loads[RATIOS[-1]]}",
            )
        traffic[split] = split_traffic
    return out, traffic


def _fig_prefill_cache(out, k_by_ratio, n_experts: int, split: str) -> bytes:
    """Figure 05: 4x5 grid of per-batch static vs cached curves for one split."""
    import numpy as np

    panels = [
        (
            f"batch B={batch}",
            out[(split, batch)]["static"],
            out[(split, batch)]["cached"],
            {ratio: np.arange(n_experts - k_by_ratio[ratio] + 1) for ratio in RATIOS},
            None,
        )
        for batch in BATCH_SIZES
    ]
    return _compare_grid_figure(
        panels,
        "portion of prefill events",
        f"Concurrent prefill ({split}): static vs adaptive cache, "
        "unique missing experts per layer, x capped at 256-k",
    )


def _fig_prefill_cdf(out, k_by_ratio, n_experts: int, split: str) -> bytes:
    """Figure 09: CDF (``P(missing <= m)``) of the per-batch prefill curves."""
    import numpy as np

    def to_cdf(curves: Curves) -> Curves:
        return {ratio: _cdf(curves[ratio]) for ratio in RATIOS}

    panels = [
        (
            f"batch B={batch}",
            to_cdf(out[(split, batch)]["static"]),
            to_cdf(out[(split, batch)]["cached"]),
            {ratio: np.arange(n_experts - k_by_ratio[ratio] + 1) for ratio in RATIOS},
            None,
        )
        for batch in BATCH_SIZES
    ]
    return _compare_grid_figure(
        panels,
        "portion of prefill events (CDF)",
        f"Concurrent prefill ({split}) CDF: static vs adaptive cache, x capped at 256-k",
        xlabel="missing experts (<= m)",
    )


def _fig_traffic(traffic) -> bytes:
    """Figure 10: cache loads/evictions across batch sizes, per ratio and split."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, split in zip(axes, SPLITS):
        if split not in traffic:
            ax.axis("off")
            continue
        xs = list(traffic[split].keys())
        for ratio in RATIOS:
            loads = [traffic[split][b][str(ratio)]["loads"] for b in xs]
            evictions = [traffic[split][b][str(ratio)]["evictions"] for b in xs]
            ax.plot(
                xs,
                loads,
                "-o",
                markersize=3,
                color=COLORS[ratio],
                label=f"keep {int(ratio * 100)}% load",
            )
            ax.plot(
                xs,
                evictions,
                "--s",
                markersize=3,
                color=COLORS[ratio],
                label=f"keep {int(ratio * 100)}% evict",
            )
        ax.set_xlabel("batch size B")
        ax.set_title(split)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cumulative expert loads / evictions")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Adaptive cache traffic (prefill)")
    return _figure_bytes(fig)


class PrefillMissAnalysis(Analysis):
    name = "prefill_miss"

    def run(self, ctx: AnalysisContext) -> Iterator[Event]:
        logs: list[str] = []
        out_dir = ctx.out_dir(self.name)

        out, traffic = yield from _prefill_cache_batches(
            ctx.datasets,
            {"train": ctx.by_dataset, "test": ctx.test_by_dataset},
            ctx.masks,
            ctx.counts,
            ctx.k_by_ratio,
            ctx.n_layers,
            ctx.n_experts,
            logs,
        )
        ctx.cache_prefills = out
        ctx.cache_meta["prefill"] = {
            "policy": "lfu-displacement",
            "traffic": traffic,
        }

        rows: list[str] = []
        for (split, batch), curves in out.items():
            for policy in ("static", "cached"):
                for ratio in RATIOS:
                    for value, portion in enumerate(curves[policy][ratio]):
                        rows.append(
                            f"{split},{batch},{policy},{ratio},{value},{portion:.8f}"
                        )
        _write_rows(
            os.path.join(out_dir, "distributions.csv"),
            "split,batch,policy,ratio,missing,portion",
            rows,
        )

        for split in SPLITS:
            if split in traffic:
                yield _image_event(
                    f"{self.name}/05_static_vs_cached_prefill.png"
                    if split == SPLITS[0]
                    else f"{self.name}/05_static_vs_cached_prefill_{split}.png",
                    _fig_prefill_cache(out, ctx.k_by_ratio, ctx.n_experts, split),
                )
                yield _image_event(
                    f"{self.name}/09_prefill_cdf.png"
                    if split == SPLITS[0]
                    else f"{self.name}/09_prefill_cdf_{split}.png",
                    _fig_prefill_cdf(out, ctx.k_by_ratio, ctx.n_experts, split),
                )
        yield _image_event(
            f"{self.name}/10_cache_traffic.png", _fig_traffic(traffic)
        )

        summary = {
            "run_id": ctx.run_id,
            "created_at": _utc_now(),
            "quick": ctx.quick,
            "ratios": list(RATIOS),
            "policy": "lfu-displacement",
            "traffic": traffic,
            "batches": {
                split: {
                    str(batch): {
                        policy: {
                            str(r): _dist_stats(out[(split, batch)][policy][r])
                            for r in RATIOS
                        }
                        for policy in ("static", "cached")
                    }
                    for batch in BATCH_SIZES
                    if (split, batch) in out
                }
                for split in SPLITS
                if split in traffic
            },
        }
        _write_json(os.path.join(out_dir, "summary.json"), summary)

        for name in ("distributions.csv", "summary.json"):
            yield _file_event(os.path.join(out_dir, name), f"{self.name}/{name}")
        yield _file_event(_write_run_log(out_dir, logs), f"{self.name}/run.log")


__all__ = ["PrefillMissAnalysis"]
