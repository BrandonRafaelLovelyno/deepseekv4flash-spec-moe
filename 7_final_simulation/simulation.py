"""Replay the trained per-layer predictors under a mixed chunked-prefill load.

This is the CPU half of the stage. It reads the per-token predictions the GPU
``predict`` stage cached, then sweeps the pre-built chunk plans for every
``(decode_portion, chunk_size, resident_ratio, policy, fetch_count)``
combination. Two resident-set policies share the chunk loop (see
``config.yaml``'s ``simulation.policies``):

* ``static`` -- the fixed training-hot mask, optionally topped up by the top-N
  predicted non-resident experts from that chunk's prediction frequency;
* ``cached`` -- the adaptive LFU-displacement ``AdaptiveCache`` (mirrors
  ``4_analysis``), seeded from the same hot mask and updated once per chunk from
  the chunk's ground-truth demand. It is an *oracle*: it observes the true routed
  experts, so it upper-bounds any prediction-driven loader.

No GPU and no torch: the whole stage is numpy plus the cache files. numpy is
imported inside functions so the local entrypoint can import this module without
the scientific stack.
"""

from __future__ import annotations

from typing import Any

from helper import (
    cache_dims,
    distribution_stats,
    load_cache_manifest,
    load_config,
    load_harvest_manifest,
    load_prediction_manifest,
    prediction_dir,
    prediction_fingerprint,
    read_cached_expert_counts,
    read_cached_prediction,
    read_cached_topk,
    resident_masks,
    resolve_profile,
    utc_now,
)
from reporting import print_profile_report

HistKey = tuple[str, float, int, float, str, int]
PlanKey = tuple[str, float, int]


def simulate(
    config_text: str, run_id: str, profile_id: str = ""
) -> dict[str, Any]:
    """Sweep every layer against the cached predictions (CPU only)."""
    cfg = load_config(config_text)
    cache_dir = cfg["cache"]["dir"]
    cache_manifest = load_cache_manifest(cache_dir)
    if cache_manifest is None:
        raise ValueError(f"no cache at {cache_dir}; run 6_train_all first")
    harvest_manifest = load_harvest_manifest(cfg["harvest"]["dir"])
    dims = cache_dims(cache_manifest)

    metas, skipped, resolved_id, profile_desc = resolve_profile(cfg, dims)
    print_profile_report(profile_desc)
    cache_id = profile_id or resolved_id

    splits = [str(split) for split in cfg["data"]["split"]]
    plans, diagnostics = build_plans(cfg, cache_manifest, harvest_manifest, splits)
    if not plans:
        raise ValueError("no chunk plans built; check the harvest manifest and splits")

    pred_dir = prediction_dir(cfg["output"]["volume_dir"], cache_id)
    validate_predictions(cfg, cache_manifest, cache_id, metas, pred_dir)

    total: dict[str, dict] = {"hist": {}, "union": {}, "tokens": {}, "traffic": {}}
    for index, meta in enumerate(metas, start=1):
        print(
            f"[sim] layer {index}/{len(metas)} L{meta['layer']:02d} "
            f"d={meta['distance']} src=L{meta['source_layer']:02d}",
            flush=True,
        )
        layer_splits = sorted({key[0] for key in plans})
        predictions = {
            split: read_cached_prediction(pred_dir, meta["layer"], split)
            for split in layer_splits
        }
        hist, union, tokens, traffic = replay_layer(
            cfg, dims, cache_dir, cache_manifest, plans, meta, predictions
        )
        _accumulate(total, hist, union, tokens, traffic)

    return finalize(
        cfg,
        run_id,
        resolved_id,
        profile_desc,
        metas,
        skipped,
        dims,
        total,
        diagnostics,
    )


# --------------------------------------------------------------------------- #
# Prediction-cache validation
# --------------------------------------------------------------------------- #
def validate_predictions(
    cfg: dict[str, Any],
    cache_manifest: dict[str, Any],
    profile_id: str,
    metas: list[dict[str, Any]],
    pred_dir: str,
) -> None:
    """Fail early when the prediction cache is missing or stale.

    The sweep is only meaningful if every layer's predictions match the current
    cache and profile; otherwise tell the caller to run the predict stage.
    """
    manifest = load_prediction_manifest(pred_dir)
    if manifest is None:
        raise ValueError(
            f"no prediction cache at {pred_dir}; run the predict stage first"
        )
    expected = prediction_fingerprint(
        profile_id,
        cache_manifest,
        int(cfg["simulation"]["prediction_k"]),
        bool(cfg["simulation"]["apply_bias"]),
    )
    if manifest.get("fingerprint") != expected:
        raise ValueError(
            f"prediction cache at {pred_dir} is stale "
            f"(fingerprint {manifest.get('fingerprint')} != {expected}); "
            "re-run the predict stage"
        )
    entries = manifest.get("entries", {})
    missing = [int(meta["layer"]) for meta in metas if str(meta["layer"]) not in entries]
    if missing:
        raise ValueError(
            f"prediction cache is missing layers {missing}; re-run the predict stage"
        )


# --------------------------------------------------------------------------- #
# Chunk plans (shared by every layer)
# --------------------------------------------------------------------------- #
def build_plans(
    cfg: dict[str, Any],
    cache_manifest: dict[str, Any],
    harvest_manifest: dict[str, Any],
    splits: list[str],
) -> tuple[dict[PlanKey, Any], dict[PlanKey, dict[str, float]]]:
    """Build every ``(split, decode_portion, chunk_size)`` chunk plan once.

    Plans depend only on the token streams, not on any layer, so they are built
    once and replayed for all checkpoints.
    """
    import numpy as np
    from helper import paired_slices, read_is_decode
    from streams import (
        ChunkPlan,
        build_session_tokens,
        chunk_diagnostics,
        dispersed_decode_stream,
        iter_chunks,
        sequential_prefill_stream,
    )

    sim = cfg["simulation"]
    seed = int(cfg["seed"])
    plans: dict[PlanKey, Any] = {}
    diagnostics: dict[PlanKey, dict[str, float]] = {}

    for offset, split in enumerate(splits):
        slices = paired_slices(cache_manifest, harvest_manifest, split)
        if not slices:
            print(f"[sim] {split}: no slices, skipping", flush=True)
            continue
        labels = {entry["path"]: read_is_decode(entry["path"]) for entry in slices}
        sessions = build_session_tokens(slices, labels)

        n_rows = sum(int(entry["n_tokens"]) for entry in slices)
        session_of_row = np.full(n_rows, -1, dtype=np.int64)
        is_decode_row = np.zeros(n_rows, dtype=bool)
        for index, session in enumerate(sessions):
            if session.decode.shape[0]:
                session_of_row[session.decode] = index
                is_decode_row[session.decode] = True
            if session.prefill.shape[0]:
                session_of_row[session.prefill] = index

        decode = dispersed_decode_stream(sessions, seed + offset)
        prefill = sequential_prefill_stream(sessions, seed + 100 + offset)

        for portion in sim["decode_portions"]:
            for chunk_size in sim["chunk_sizes"]:
                chunks: list[Any] = []
                n_decode: list[int] = []
                n_prefill: list[int] = []
                for chunk, take_decode, take_prefill in iter_chunks(
                    decode, prefill, int(chunk_size), float(portion)
                ):
                    chunks.append(chunk)
                    n_decode.append(take_decode)
                    n_prefill.append(take_prefill)
                plan = ChunkPlan(
                    chunks=chunks,
                    n_decode=np.asarray(n_decode, dtype=np.int64),
                    n_prefill=np.asarray(n_prefill, dtype=np.int64),
                )
                key: PlanKey = (split, float(portion), int(chunk_size))
                plans[key] = plan
                diagnostics[key] = chunk_diagnostics(
                    plan, session_of_row, is_decode_row
                )
    return plans, diagnostics


# --------------------------------------------------------------------------- #
# Per-layer sweep
# --------------------------------------------------------------------------- #
def replay_layer(
    cfg: dict[str, Any],
    dims: dict[str, int],
    cache_dir: str,
    cache_manifest: dict[str, Any],
    plans: dict[PlanKey, Any],
    meta: dict[str, Any],
    predictions: dict[str, Any],
) -> tuple[
    dict[HistKey, Any],
    dict[PlanKey, int],
    dict[PlanKey, int],
    dict[PlanKey, dict],
]:
    """Sweep one layer's chunk plans against its cached predictions.

    Two resident-set policies share the chunk loop:

    * ``static`` -- the fixed training-hot mask, optionally topped up by the
      top-N predicted non-resident experts *for this chunk only* (``fetch``);
    * ``cached`` -- the adaptive ``AdaptiveCache`` seeded from the same mask and
      updated once per chunk from the chunk's ground-truth demand (an oracle:
      it sees the true routed experts). ``fetch`` is not meaningful here and is
      recorded as ``0``.

    Returns ``(histograms, demand_union, token_counts, cache_traffic)`` keyed by
    chunk plan.
    """
    import numpy as np

    from helper import AdaptiveCache

    sim = cfg["simulation"]
    n_experts = int(dims["n_experts"])
    ratios = tuple(float(ratio) for ratio in sim["ready_ratios"])
    policies = tuple(str(policy) for policy in sim["policies"])
    fetches = sorted(int(count) for count in sim["fetch_counts"])
    cached = "cached" in policies
    static = "static" in policies

    counts = read_cached_expert_counts(cache_dir, cache_manifest, meta["layer"])
    masks, k_by_ratio = resident_masks(counts, ratios)
    splits = sorted({key[0] for key in plans})
    truths = {
        split: read_cached_topk(cache_dir, cache_manifest, split, meta["layer"])
        for split in splits
    }

    histograms: dict[HistKey, Any] = {}
    demand_union: dict[PlanKey, int] = {}
    token_counts: dict[PlanKey, int] = {}
    traffic: dict[PlanKey, dict] = {}

    def histogram(
        split: str,
        portion: float,
        chunk_size: int,
        ratio: float,
        policy: str,
        fetch: int,
    ) -> Any:
        key: HistKey = (split, portion, chunk_size, ratio, policy, fetch)
        if key not in histograms:
            histograms[key] = np.zeros(
                n_experts - int(k_by_ratio[ratio]) + 1, dtype=np.int64
            )
        return histograms[key]

    for plan_key, plan in plans.items():
        split, portion, chunk_size = plan_key
        truth = truths[split]
        predicted = predictions[split]
        cache = (
            AdaptiveCache(masks, counts, k_by_ratio, ratios, n_experts)
            if cached
            else None
        )
        for chunk in plan.chunks:
            demand = np.zeros(n_experts, dtype=bool)
            demand[truth[chunk].ravel().astype(np.int64)] = True
            demand_union[plan_key] = demand_union.get(plan_key, 0) + int(
                demand.sum()
            )
            token_counts[plan_key] = token_counts.get(plan_key, 0) + int(
                chunk.shape[0]
            )
            if static:
                frequency = np.bincount(
                    predicted[chunk].ravel().astype(np.int64), minlength=n_experts
                )
                for ratio in ratios:
                    mask = masks[ratio]
                    ranked = _rank_nonresident(frequency, mask)
                    for fetch in fetches:
                        available = mask
                        if fetch > 0:
                            available = mask.copy()
                            available[ranked[:fetch]] = True
                        missing = int(np.count_nonzero(demand & ~available))
                        histogram(
                            split, portion, chunk_size, ratio, "static", fetch
                        )[missing] += 1
            if cache is not None:
                cached_miss = cache.observe_mask(demand)
                for ri, ratio in enumerate(ratios):
                    key = histogram(
                        split, portion, chunk_size, ratio, "cached", 0
                    )
                    key[int(cached_miss[ri])] += 1
        if cache is not None:
            traffic[plan_key] = cache.traffic()
    return histograms, demand_union, token_counts, traffic


def _rank_nonresident(frequency: Any, mask: Any) -> Any:
    """Non-resident expert ids ordered by predicted frequency (descending)."""
    import numpy as np

    candidates = np.flatnonzero(~mask)
    if candidates.shape[0] == 0:
        return candidates
    order = np.argsort(frequency[candidates], kind="stable")[::-1]
    return candidates[order]


# --------------------------------------------------------------------------- #
# Aggregation and payload assembly
# --------------------------------------------------------------------------- #
def _accumulate(
    total: dict[str, dict],
    histograms: dict[HistKey, Any],
    demand_union: dict[PlanKey, int],
    token_counts: dict[PlanKey, int],
    traffic: dict[PlanKey, dict],
) -> None:
    """Fold one layer's histograms, counters and cache traffic into the run."""
    for key, value in histograms.items():
        current = total["hist"].get(key)
        total["hist"][key] = value if current is None else current + value
    for key, value in demand_union.items():
        total["union"][key] = total["union"].get(key, 0) + int(value)
    for key, value in token_counts.items():
        total["tokens"][key] = total["tokens"].get(key, 0) + int(value)
    for plan_key, by_ratio in traffic.items():
        aggregate = total["traffic"].setdefault(plan_key, {})
        for ratio, stats in by_ratio.items():
            current = aggregate.setdefault(
                ratio, {"loads": 0, "evictions": 0, "resident": 0}
            )
            current["loads"] += int(stats["loads"])
            current["evictions"] += int(stats["evictions"])
            current["resident"] += int(stats["resident"])


def finalize(
    cfg: dict[str, Any],
    run_id: str,
    profile_id: str,
    profile_desc: dict[str, Any],
    metas: list[dict[str, Any]],
    skipped: list[int],
    dims: dict[str, int],
    total: dict[str, dict],
    diagnostics: dict[PlanKey, dict[str, float]],
) -> dict[str, Any]:
    """Assemble the distributions table, per-combo stats and JSON summary."""
    histograms = total["hist"]
    ordering = sorted(
        histograms,
        key=lambda key: (key[0], key[1], key[2], key[3], key[4], key[5]),
    )
    rows: list[dict[str, Any]] = []
    combos: dict[str, Any] = {}
    for key in ordering:
        split, portion, chunk_size, ratio, policy, fetch = key
        counts = histograms[key]
        total_count = int(counts.sum())
        combos[f"{split}|{portion}|{chunk_size}|{ratio}|{policy}|{fetch}"] = (
            distribution_stats(counts)
        )
        cumulative = 0
        for missing, count in enumerate(counts):
            count = int(count)
            if count == 0:
                continue
            cumulative += count
            rows.append(
                {
                    "split": split,
                    "decode_portion": portion,
                    "chunk_size": chunk_size,
                    "ratio": ratio,
                    "policy": policy,
                    "fetch": fetch,
                    "missing": missing,
                    "count": count,
                    "portion": round(count / total_count, 8) if total_count else 0.0,
                    "cdf": round(cumulative / total_count, 8)
                    if total_count
                    else 0.0,
                }
            )

    uniqueness = {}
    for plan_key, union in total["union"].items():
        split, portion, chunk_size = plan_key
        tokens = total["tokens"].get(plan_key, 0)
        denom = max(tokens * int(dims["top_k"]), 1)
        uniqueness[f"{split}|{portion}|{chunk_size}"] = round(union / denom, 6)

    traffic = {
        f"{split}|{portion}|{chunk_size}": by_ratio
        for (split, portion, chunk_size), by_ratio in sorted(total["traffic"].items())
    }

    training_label = profile_desc.get("label") or profile_desc.get("run_id", profile_id)
    summary = {
        "run_id": run_id,
        "created_at": utc_now(),
        "training_run_id": training_label,
        "profile_id": profile_id,
        "distance_profile": profile_desc,
        "config": cfg,
        "n_layers": len(metas),
        "layers": [int(meta["layer"]) for meta in metas],
        "skipped_layers": skipped,
        "n_experts": int(dims["n_experts"]),
        "top_k": int(dims["top_k"]),
        "policies": [str(policy) for policy in cfg["simulation"]["policies"]],
        "combos": combos,
        "traffic": traffic,
        "diagnostics": {
            f"{split}|{portion}|{chunk_size}": stats
            for (split, portion, chunk_size), stats in sorted(
                diagnostics.items(), key=lambda item: item[0]
            )
        },
        "demand_uniqueness": uniqueness,
    }
    return {
        "summary": summary,
        "distributions": rows,
        "histograms": histograms,
    }
