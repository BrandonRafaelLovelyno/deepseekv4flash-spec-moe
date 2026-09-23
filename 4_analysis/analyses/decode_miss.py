"""Study: concurrent-decode unique missing experts with an adaptive cache.

For every batch size ``B`` and split, a fresh ``AdaptiveCache`` is seeded from
the training-hot set, then ``B`` decode sessions of that split are replayed
concurrently. Each step measures the same observations twice -- against the
fixed ``masks`` (static) and the evolving resident set (cached) -- and the cache
then updates from the step's union of demanded experts. Train and test are
independent (test is unseen but starts from the same train-derived seed).

Artifacts (in ``<run>/decode_miss/``):
    * ``distributions.csv``             -- split,batch,policy,ratio,missing,portion
    * ``04_static_vs_cached_decode.png`` -- 4x5 grid per split, static vs cached
    * ``08_decode_cdf.png``             -- same grids as CDFs (P(missing <= m))
    * ``10_cache_traffic.png``          -- loads/evictions vs B, per split
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
    SPLITS,
    AdaptiveCache,
    AnalysisContext,
    Curves,
    Event,
    _cdf,
    _compare_grid_figure,
    _decode_ids,
    _dist_stats,
    _emit,
    _figure_bytes,
    _file_event,
    _image_event,
    _pyplot,
    _utc_now,
    _write_json,
    _write_rows,
    _write_run_log,
)

CacheCurves = dict[str, Curves]


def _session_for(slot: int, datasets: list[str], pools) -> dict:
    """Round-robin pick: dataset by slot, then the round-th session in its pool."""
    dataset = datasets[slot % len(datasets)]
    pool = pools[dataset]
    return pool[(slot // len(datasets)) % len(pool)]


def _decode_cache_batches(
    datasets: list[str],
    pools_by_split: dict[str, dict],
    masks,
    counts,
    k_by_ratio: dict[float, int],
    n_layers: int,
    n_experts: int,
    top_k: int,
    logs: list[str],
) -> Iterator[Event]:
    """Replay every ``(split, B)``; return ``(curves, traffic)``."""
    import numpy as np

    decode_cache: dict[str, "np.ndarray"] = {}

    def decode_ids(record):
        path = record["path"]
        if path not in decode_cache:
            decode_cache[path] = _decode_ids(path)
        return decode_cache[path]

    layer_axis = np.arange(n_layers)
    masks_stack = np.stack([masks[ratio] for ratio in RATIOS])  # [R, L, E]
    out: dict[tuple[str, int], CacheCurves] = {}
    traffic: dict[str, dict[int, dict]] = {}
    for split in SPLITS:
        pools = pools_by_split.get(split, {})
        split_datasets = [d for d in datasets if pools.get(d)]
        if not split_datasets:
            continue
        split_traffic: dict[int, dict] = {}
        for batch in BATCH_SIZES:
            cache = AdaptiveCache(masks, counts, k_by_ratio, n_layers, n_experts)
            sessions = [
                decode_ids(_session_for(slot, split_datasets, pools))
                for slot in range(batch)
            ]
            n_steps = max(s.shape[0] for s in sessions)
            max_missing = top_k * batch
            hist_static = {
                ratio: np.zeros((n_layers, max_missing + 1), dtype=np.int64)
                for ratio in RATIOS
            }
            hist_cached = {
                ratio: np.zeros((n_layers, max_missing + 1), dtype=np.int64)
                for ratio in RATIOS
            }
            for step in range(n_steps):
                demand = np.zeros((n_layers, n_experts), dtype=bool)
                for ids in sessions:
                    if step < ids.shape[0]:
                        np.put_along_axis(demand, ids[step], True, axis=1)
                static_miss = (demand[None, :, :] & ~masks_stack).sum(axis=2)  # [R, L]
                cached_miss = cache.observe_mask_all(demand)  # [R, L]
                for ri, ratio in enumerate(RATIOS):
                    hist_static[ratio][layer_axis, static_miss[ri]] += 1
                    hist_cached[ratio][layer_axis, cached_miss[ri]] += 1
            denom = max(n_steps * n_layers, 1)
            out[(split, batch)] = {
                "static": {r: hist_static[r].sum(axis=0) / denom for r in RATIOS},
                "cached": {r: hist_cached[r].sum(axis=0) / denom for r in RATIOS},
            }
            split_traffic[batch] = cache.traffic()
            stats = _dist_stats(out[(split, batch)]["cached"][RATIOS[-1]])
            yield _emit(
                logs,
                f"{split:5} batch {batch:>2}: {n_steps} steps, "
                f"keep 75% mean missing {stats['mean_missing']:.3f}, "
                f"loads {cache.loads[RATIOS[-1]]}",
            )
        traffic[split] = split_traffic
    return out, traffic


def _fig_decode_cache(out, top_k: int, split: str) -> bytes:
    """Figure 04: 4x5 grid of per-batch static vs cached curves for one split."""
    import numpy as np

    panels = [
        (
            f"batch B={batch}",
            out[(split, batch)]["static"],
            out[(split, batch)]["cached"],
            {ratio: np.arange(top_k * batch + 1) for ratio in RATIOS},
            (0, top_k * batch),
        )
        for batch in BATCH_SIZES
    ]
    return _compare_grid_figure(
        panels,
        "portion of decode steps",
        f"Concurrent decode ({split}): static vs adaptive cache, "
        "unique missing experts per layer",
    )


def _fig_decode_cdf(out, top_k: int, split: str) -> bytes:
    """Figure 08: CDF (``P(missing <= m)``) of the per-batch decode curves."""
    import numpy as np

    def to_cdf(curves: Curves) -> Curves:
        return {ratio: _cdf(curves[ratio]) for ratio in RATIOS}

    panels = [
        (
            f"batch B={batch}",
            to_cdf(out[(split, batch)]["static"]),
            to_cdf(out[(split, batch)]["cached"]),
            {ratio: np.arange(top_k * batch + 1) for ratio in RATIOS},
            (0, top_k * batch),
        )
        for batch in BATCH_SIZES
    ]
    return _compare_grid_figure(
        panels,
        "portion of decode steps (CDF)",
        f"Concurrent decode ({split}) CDF: static vs adaptive cache",
        xlabel="missing experts (<= m)",
    )


def _fig_traffic(traffic, key: str, xlabel: str) -> bytes:
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
        ax.set_xlabel(xlabel)
        ax.set_title(split)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cumulative expert loads / evictions")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Adaptive cache traffic ({key})")
    return _figure_bytes(fig)


class DecodeMissAnalysis(Analysis):
    name = "decode_miss"

    def run(self, ctx: AnalysisContext) -> Iterator[Event]:
        logs: list[str] = []
        out_dir = ctx.out_dir(self.name)

        out, traffic = yield from _decode_cache_batches(
            ctx.datasets,
            {"train": ctx.by_dataset, "test": ctx.test_by_dataset},
            ctx.masks,
            ctx.counts,
            ctx.k_by_ratio,
            ctx.n_layers,
            ctx.n_experts,
            ctx.top_k,
            logs,
        )
        ctx.cache_batches = out
        ctx.cache_meta["decode"] = {
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
                    f"{self.name}/04_static_vs_cached_decode.png"
                    if split == SPLITS[0]
                    else f"{self.name}/04_static_vs_cached_decode_{split}.png",
                    _fig_decode_cache(out, ctx.top_k, split),
                )
                yield _image_event(
                    f"{self.name}/08_decode_cdf.png"
                    if split == SPLITS[0]
                    else f"{self.name}/08_decode_cdf_{split}.png",
                    _fig_decode_cdf(out, ctx.top_k, split),
                )
        yield _image_event(
            f"{self.name}/10_cache_traffic.png",
            _fig_traffic(traffic, "decode", "batch size B"),
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


__all__ = ["DecodeMissAnalysis"]
