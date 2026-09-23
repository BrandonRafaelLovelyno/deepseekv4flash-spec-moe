"""Study: per-token expert miss with an adaptive expert cache.

Each split replays independently through an ``AdaptiveCache`` seeded from the
training-hot set: every token measures its miss against the current resident
set (static = fixed ``masks``, cached = adaptive), then the cache updates --
demanded experts are credited and missing ones displace the coldest residents.

Train and test start from the same train-derived seed, so train keeps its
advantage (the hot set was chosen on it); test is unseen. Replay is capped at
``CACHE_TOKEN_BUDGET`` tokens per split.

Artifacts (in ``<run>/token_miss/``):
    * ``distributions.csv``           -- split,dataset,policy,ratio,missing,portion
    * ``01_static_vs_cached_token.png`` -- pooled train vs test, static vs cached
    * ``summary.json``, ``run.log``
"""

from __future__ import annotations

import os
from typing import Any, Iterator

from analyses.base import Analysis
from helper import (
    CACHE_TOKEN_BUDGET,
    COLORS,
    RATIOS,
    SPLITS,
    AdaptiveCache,
    AnalysisContext,
    Curves,
    Event,
    GroupAccumulator,
    _dist_stats,
    _emit,
    _figure_bytes,
    _file_event,
    _image_event,
    _pyplot,
    _read_topk_ids,
    _utc_now,
    _write_json,
    _write_rows,
    _write_run_log,
)


def _replay_split(
    split: str,
    records,
    cache: AdaptiveCache,
    masks,
    n_layers: int,
    top_k: int,
    budget: int,
    logs: list[str],
) -> Iterator[Event]:
    """Replay one split's tokens, returning static + cached group accumulators.

    Each token is one forward pass: the static miss uses the fixed ``masks``,
    the cached miss uses (and then advances) ``cache``. Yields one ``log`` event
    per replayed slice and returns ``(static_groups, cached_groups)`` keyed by
    dataset id (or ``None`` for the pooled group).
    """
    import numpy as np

    shape = (len(RATIOS), n_layers, top_k + 1)
    layer_index = np.arange(n_layers)[None, :, None]
    layer_axis = np.arange(n_layers)

    def accumulator() -> GroupAccumulator:
        return {
            "counts": np.zeros(shape, dtype=np.int64),
            "tokens": np.zeros((len(RATIOS), n_layers)),
        }

    static_groups: dict[str | None, GroupAccumulator] = {}
    cached_groups: dict[str | None, GroupAccumulator] = {}
    used = 0
    for record in records:
        if used >= budget:
            break
        ids = _read_topk_ids(record["path"])
        take = int(min(ids.shape[0], budget - used))
        if take <= 0:
            break
        used += take
        keys: list[str | None] = [None, record["dataset_id"]]
        for key in keys:
            static_groups.setdefault(key, accumulator())
            cached_groups.setdefault(key, accumulator())
        for t in range(take):
            ids_t = ids[t]
            for ri, ratio in enumerate(RATIOS):
                resident = masks[ratio][layer_index, ids_t]  # [L, K] bool
                static_miss = (top_k - resident.sum(axis=-1)).astype(np.int64)
                cached_miss = cache.observe_ids(ratio, ids_t)
                for key in keys:
                    static_groups[key]["counts"][ri, layer_axis, static_miss] += 1
                    cached_groups[key]["counts"][ri, layer_axis, cached_miss] += 1
        for key in keys:
            static_groups[key]["tokens"] += take
            cached_groups[key]["tokens"] += take
        yield _emit(
            logs,
            f"{split}/{record['dataset_id']}/{record['document_id']} ({take} tokens)",
        )
    return static_groups, cached_groups


def _curve(acc: GroupAccumulator) -> Curves:
    """Average the per-layer fraction curves into one portion curve per ratio."""
    import numpy as np

    out: Curves = {}
    for ri, ratio in enumerate(RATIOS):
        tokens = np.maximum(acc["tokens"][ri], 1)[:, None]
        per_layer = acc["counts"][ri] / tokens  # [L, K+1]
        out[ratio] = per_layer.mean(axis=0)
    return out


def _build_cache(ctx: AnalysisContext, logs: list[str]):
    """Replay every split through a fresh cache; return pooled/per-dataset curves."""
    pooled: dict[str, dict[str, Curves]] = {}
    per_dataset: dict[str, dict[str, dict[str, Curves]]] = {}
    traffic: dict[str, dict] = {}
    for split in SPLITS:
        records = [r for r in ctx.records if r["split"] == split]
        if not records:
            continue
        cache = AdaptiveCache(
            ctx.masks, ctx.counts, ctx.k_by_ratio, ctx.n_layers, ctx.n_experts
        )
        static_groups, cached_groups = yield from _replay_split(
            split,
            records,
            cache,
            ctx.masks,
            ctx.n_layers,
            ctx.top_k,
            CACHE_TOKEN_BUDGET,
            logs,
        )
        pooled[split] = {
            "static": _curve(static_groups[None]),
            "cached": _curve(cached_groups[None]),
        }
        per_dataset[split] = {
            ds: {
                "static": _curve(static_groups[ds]),
                "cached": _curve(cached_groups[ds]),
            }
            for ds in ctx.datasets
            if ds in static_groups
        }
        traffic[split] = cache.traffic()
    return pooled, per_dataset, traffic


def _fig_token_cache(
    pooled: dict[str, dict[str, Curves]], n_tokens_by_split: dict[str, int], top_k: int
) -> bytes:
    """Figure 07: pooled train vs test, static vs cached, one bar pair per ratio."""
    import numpy as np

    plt = _pyplot()
    x = np.arange(top_k + 1)
    width = 0.13
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split in zip(axes, SPLITS):
        if split not in pooled:
            ax.axis("off")
            continue
        for i, ratio in enumerate(RATIOS):
            base = (i - 1) * 0.3
            ax.bar(
                x + base - width / 2,
                pooled[split]["static"][ratio],
                width,
                color=COLORS[ratio],
                label=f"keep {int(ratio * 100)}% static",
            )
            ax.bar(
                x + base + width / 2,
                pooled[split]["cached"][ratio],
                width,
                color=COLORS[ratio],
                hatch="//",
                edgecolor="white",
                label=f"keep {int(ratio * 100)}% cache",
            )
        ax.set_xticks(x)
        ax.set_xlabel("missing experts (out of 6)")
        ax.set_title(f"{split} ({n_tokens_by_split[split]:,} tokens)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("portion of tokens")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Token expert miss: static vs adaptive cache")
    return _figure_bytes(fig)


class TokenMissAnalysis(Analysis):
    name = "token_miss"

    def run(self, ctx: AnalysisContext) -> Iterator[Event]:
        logs: list[str] = []
        out_dir = ctx.out_dir(self.name)

        pooled, per_dataset, traffic = yield from _build_cache(ctx, logs)
        ctx.cache_pooled = pooled
        ctx.cache_per_dataset = per_dataset
        ctx.cache_meta["token"] = {
            "policy": "lfu-displacement",
            "token_budget": CACHE_TOKEN_BUDGET,
            "traffic": traffic,
        }

        for split in SPLITS:
            if split not in pooled:
                continue
            for ratio in RATIOS:
                s = _dist_stats(pooled[split]["static"][ratio])
                c = _dist_stats(pooled[split]["cached"][ratio])
                yield _emit(
                    logs,
                    f"{split:5} keep {int(ratio * 100):>2}%: "
                    f"mean missing static {s['mean_missing']:.4f} -> "
                    f"cached {c['mean_missing']:.4f}",
                )

        rows: list[str] = []
        for split in SPLITS:
            if split not in pooled:
                continue
            for policy in ("static", "cached"):
                for ratio in RATIOS:
                    for value, portion in enumerate(pooled[split][policy][ratio]):
                        rows.append(f"{split},pooled,{policy},{ratio},{value},{portion:.8f}")
            for ds, curves in per_dataset[split].items():
                for policy in ("static", "cached"):
                    for ratio in RATIOS:
                        for value, portion in enumerate(curves[policy][ratio]):
                            rows.append(
                                f"{split},{ds},{policy},{ratio},{value},{portion:.8f}"
                            )
        _write_rows(
            os.path.join(out_dir, "distributions.csv"),
            "split,dataset,policy,ratio,missing,portion",
            rows,
        )

        yield _image_event(
            f"{self.name}/01_static_vs_cached_token.png",
            _fig_token_cache(pooled, ctx.n_tokens_by_split, ctx.top_k),
        )

        summary: dict[str, Any] = {
            "run_id": ctx.run_id,
            "created_at": _utc_now(),
            "quick": ctx.quick,
            "ratios": list(RATIOS),
            "policy": "lfu-displacement",
            "token_budget": CACHE_TOKEN_BUDGET,
            "traffic": traffic,
            "splits": {
                split: {
                    policy: {
                        str(r): _dist_stats(pooled[split][policy][r])
                        for r in RATIOS
                    }
                    for policy in ("static", "cached")
                }
                for split in SPLITS
                if split in pooled
            },
        }
        _write_json(os.path.join(out_dir, "summary.json"), summary)

        for name in ("distributions.csv", "summary.json"):
            yield _file_event(os.path.join(out_dir, name), f"{self.name}/{name}")
        yield _file_event(_write_run_log(out_dir, logs), f"{self.name}/run.log")
