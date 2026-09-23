"""Study: per-token expert miss distribution (no batching).

Pass B of the original analysis. Replays every train/test token against the
train-derived resident sets and tallies, per token and layer,
``missing = top_k - (# experts resident)`` in 0..top_k. Curves are normalized
per layer then averaged over the 43 layers, so each is a portion of tokens.

Artifacts (in ``<run>/token_miss/``):
    * ``distributions.csv``          -- split,dataset,ratio,missing,portion
    * ``01_missing_distribution.png`` -- pooled train vs test, one bar per ratio
    * ``02_per_dataset.png``         -- 2xN small multiples
    * ``03_cumulative_missing.png``  -- portion needing at least m fetches
    * ``summary.json``, ``run.log``
"""

from __future__ import annotations

import os
from typing import Iterator

from analyses.base import Analysis
from helper import (
    COLORS,
    RATIOS,
    SPLITS,
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
    _write_run_log,
)


def _replay_tokens(
    records,
    masks,
    n_layers: int,
    top_k: int,
    logs: list[str],
) -> Iterator[Event]:
    """Tally per-token missing counts for every replay group.

    Yields one ``log`` event per replayed slice and returns the group
    accumulator dict keyed by ``(split, dataset_or_None)``.
    """
    import numpy as np

    shape = (len(RATIOS), n_layers, top_k + 1)
    layer_index = np.arange(n_layers)[None, :, None]

    def accumulator() -> GroupAccumulator:
        return {
            "counts": np.zeros(shape, dtype=np.int64),
            "tokens": np.zeros((len(RATIOS), n_layers)),
        }

    groups: dict[tuple[str, str | None], GroupAccumulator] = {}
    for split in SPLITS:
        for record in [r for r in records if r["split"] == split]:
            ids = _read_topk_ids(record["path"])
            n_tokens = ids.shape[0]
            keys = [(split, None), (split, record["dataset_id"])]
            for key in keys:
                groups.setdefault(key, accumulator())
            for ri, ratio in enumerate(RATIOS):
                resident = masks[ratio][layer_index, ids]  # [T, L, K] bool
                missing = top_k - resident.sum(axis=-1).astype(np.int64)  # [T, L]
                for value in range(top_k + 1):
                    per_layer = (missing == value).sum(axis=0)  # [L]
                    for key in keys:
                        groups[key]["counts"][ri, :, value] += per_layer
                for key in keys:
                    groups[key]["tokens"][ri] += n_tokens
            yield _emit(
                logs,
                f"replayed {split}/{record['dataset_id']}/{record['document_id']} "
                f"({n_tokens} tokens)",
            )
    return groups


def _curve(acc: GroupAccumulator) -> Curves:
    """Average the per-layer fraction curves into one portion curve per ratio."""
    import numpy as np

    out: Curves = {}
    for ri, ratio in enumerate(RATIOS):
        tokens = np.maximum(acc["tokens"][ri], 1)[:, None]
        per_layer = acc["counts"][ri] / tokens  # [L, K+1], rows sum to 1
        out[ratio] = per_layer.mean(axis=0)  # [K+1], sums to 1
    return out


def _build_curves(
    groups: dict[tuple[str, str | None], GroupAccumulator], datasets: list[str]
) -> tuple[dict[str, Curves], dict[str, dict[str, Curves]]]:
    """Build the pooled and per-dataset curves for each split."""
    pooled = {split: _curve(groups[(split, None)]) for split in SPLITS}
    per_dataset = {
        split: {ds: _curve(groups[(split, ds)]) for ds in datasets if (split, ds) in groups}
        for split in SPLITS
    }
    return pooled, per_dataset


def _write_distributions_csv(
    path: str, pooled: dict[str, Curves], per_dataset: dict[str, dict[str, Curves]]
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("split,dataset,ratio,missing,portion\n")
        for split in SPLITS:
            for ratio in RATIOS:
                for value, portion in enumerate(pooled[split][ratio]):
                    handle.write(f"{split},pooled,{ratio},{value},{portion:.8f}\n")
            for ds, curves in per_dataset[split].items():
                for ratio in RATIOS:
                    for value, portion in enumerate(curves[ratio]):
                        handle.write(f"{split},{ds},{ratio},{value},{portion:.8f}\n")


def _fig_missing_distribution(
    pooled: dict[str, Curves], n_tokens_by_split: dict[str, int], top_k: int
) -> bytes:
    """Figure 01: pooled train vs test, one bar per ratio."""
    import numpy as np

    plt = _pyplot()
    width = 0.26
    x = np.arange(top_k + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split in zip(axes, SPLITS):
        for i, ratio in enumerate(RATIOS):
            ax.bar(
                x + (i - 1) * width,
                pooled[split][ratio],
                width,
                label=f"keep {int(ratio * 100)}%",
                color=COLORS[ratio],
            )
        ax.set_xticks(x)
        ax.set_xlabel("missing experts (out of 6)")
        ax.set_title(f"{split} ({n_tokens_by_split[split]:,} tokens)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("portion of tokens")
    axes[0].legend(title="resident set")
    fig.suptitle("Expert miss distribution when preserving hot experts")
    return _figure_bytes(fig)


def _fig_per_dataset(
    per_dataset: dict[str, dict[str, Curves]], datasets: list[str], top_k: int
) -> bytes:
    """Figure 02: 2xN per-dataset small multiples."""
    import numpy as np

    plt = _pyplot()
    width = 0.26
    x = np.arange(top_k + 1)
    fig2, axes2 = plt.subplots(
        2, len(datasets), figsize=(4 * len(datasets), 8), squeeze=False
    )
    for row, split in enumerate(SPLITS):
        for col, ds in enumerate(datasets):
            ax = axes2[row][col]
            curves = per_dataset[split].get(ds)
            if curves is None:
                ax.axis("off")
                continue
            for i, ratio in enumerate(RATIOS):
                ax.bar(
                    x + (i - 1) * width,
                    curves[ratio],
                    width,
                    color=COLORS[ratio],
                )
            ax.set_xticks(x)
            ax.set_ylim(0, 1)
            if row == len(SPLITS) - 1:
                ax.set_xlabel("missing experts")
            if col == 0:
                ax.set_ylabel(f"{split}\nportion of tokens")
            ax.set_title(ds)
            ax.grid(axis="y", alpha=0.3)
    fig2.suptitle("Per-dataset expert miss distribution")
    return _figure_bytes(fig2)


def _fig_cumulative_missing(pooled: dict[str, Curves], top_k: int) -> bytes:
    """Figure 03: portion of tokens needing at least m fetches."""
    import numpy as np

    plt = _pyplot()
    x = np.arange(top_k + 1)
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    for split, style in zip(SPLITS, ("-o", "--s")):
        for ratio in RATIOS:
            dist = pooled[split][ratio]
            cumulative = np.cumsum(dist[::-1])[::-1]
            ax3.plot(
                x,
                cumulative,
                style,
                label=f"{split} keep {int(ratio * 100)}%",
                color=COLORS[ratio],
            )
    ax3.set_xticks(x)
    ax3.set_xlabel("at least m missing experts")
    ax3.set_ylabel("portion of tokens")
    ax3.set_title("Fraction of tokens needing at least m expert fetches")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)
    return _figure_bytes(fig3)


class TokenMissAnalysis(Analysis):
    name = "token_miss"

    def run(self, ctx: AnalysisContext) -> Iterator[Event]:
        logs: list[str] = []
        out_dir = ctx.out_dir(self.name)

        groups = yield from _replay_tokens(
            ctx.records, ctx.masks, ctx.n_layers, ctx.top_k, logs
        )
        pooled, per_dataset = _build_curves(groups, ctx.datasets)
        ctx.pooled = pooled
        ctx.per_dataset = per_dataset

        for split in SPLITS:
            for ratio in RATIOS:
                stats = _dist_stats(pooled[split][ratio])
                yield _emit(
                    logs,
                    f"{split:5} keep {int(ratio * 100):>2}%: "
                    f"fully resident {stats['portion_fully_resident']:.4f}, "
                    f"needs fetch {stats['portion_needing_fetch']:.4f}, "
                    f"mean missing {stats['mean_missing']:.4f}",
                )

        _write_distributions_csv(
            os.path.join(out_dir, "distributions.csv"), pooled, per_dataset
        )

        yield _image_event(
            f"{self.name}/01_missing_distribution.png",
            _fig_missing_distribution(pooled, ctx.n_tokens_by_split, ctx.top_k),
        )
        if ctx.datasets:
            yield _image_event(
                f"{self.name}/02_per_dataset.png",
                _fig_per_dataset(per_dataset, ctx.datasets, ctx.top_k),
            )
        yield _image_event(
            f"{self.name}/03_cumulative_missing.png",
            _fig_cumulative_missing(pooled, ctx.top_k),
        )

        summary = {
            "run_id": ctx.run_id,
            "created_at": _utc_now(),
            "quick": ctx.quick,
            "ratios": list(RATIOS),
            "n_layers": ctx.n_layers,
            "top_k": ctx.top_k,
            "n_experts": ctx.n_experts,
            "datasets": ctx.datasets,
            "n_tokens_by_split": ctx.n_tokens_by_split,
            "splits": {
                split: {
                    "n_tokens": ctx.n_tokens_by_split[split],
                    "pooled": {str(r): _dist_stats(pooled[split][r]) for r in RATIOS},
                    "per_dataset": {
                        ds: {str(r): _dist_stats(curves[r]) for r in RATIOS}
                        for ds, curves in per_dataset[split].items()
                    },
                }
                for split in SPLITS
            },
        }
        _write_json(os.path.join(out_dir, "summary.json"), summary)

        for name in ("distributions.csv", "summary.json"):
            yield _file_event(os.path.join(out_dir, name), f"{self.name}/{name}")
        yield _file_event(_write_run_log(out_dir, logs), f"{self.name}/run.log")
