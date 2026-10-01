"""Replay the ground-truth routing under a mixed chunked-prefill load.

This is the whole analysis. It reuses ``7_final_simulation``'s plan build and
replay *without* the predictor: for every ``(split, decode_portion, chunk_size)``
chunk plan it walks each layer's cached truth top-k and, per chunk, counts the
**distinct routed experts** the chunk's tokens activate together. The run summary
reports that count averaged over chunks, per chunk size and per layer --
the true per-batch expert working set that speculative loading must fit in VRAM.

numpy is imported inside functions so the local entrypoint can import this module
without the scientific stack.
"""

from __future__ import annotations

from typing import Any

from helper import (
    cache_dims,
    distinct_stats,
    load_cache_manifest,
    load_config,
    load_harvest_manifest,
    read_cached_topk,
    resolve_layers,
    utc_now,
)

PlanKey = tuple[str, float, int]
LayerKey = tuple[str, float, int, int]


def simulate(config_text: str, run_id: str) -> dict[str, Any]:
    """Sweep every layer's truth top-k against the chunk plans (CPU only)."""
    cfg = load_config(config_text)
    cache_dir = cfg["cache"]["dir"]
    cache_manifest = load_cache_manifest(cache_dir)
    if cache_manifest is None:
        raise ValueError(f"no cache at {cache_dir}; run 6_train_all first")
    harvest_manifest = load_harvest_manifest(cfg["harvest"]["dir"])
    dims = cache_dims(cache_manifest)

    splits = [str(split) for split in cfg["data"]["split"]]
    layers, missing = resolve_layers(cache_manifest, splits, cfg["data"]["layers"])

    plans, diagnostics = build_plans(cfg, cache_manifest, harvest_manifest, splits)
    if not plans:
        raise ValueError("no chunk plans built; check the harvest manifest and splits")

    total: dict[str, dict] = {"hist": {}, "tokens": {}, "chunks": {}}
    for index, layer in enumerate(layers, start=1):
        print(f"[usage] layer {index}/{len(layers)} L{layer:02d}", flush=True)
        histograms, tokens, chunks = replay_layer(
            dims, cache_dir, cache_manifest, plans, layer
        )
        _accumulate(total, layer, histograms, tokens, chunks)

    return finalize(cfg, run_id, layers, missing, dims, total, diagnostics)


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
    once and replayed for all layers.
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
            print(f"[usage] {split}: no slices, skipping", flush=True)
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
    dims: dict[str, int],
    cache_dir: str,
    cache_manifest: dict[str, Any],
    plans: dict[PlanKey, Any],
    layer: int,
) -> tuple[dict[PlanKey, Any], dict[PlanKey, int], dict[PlanKey, int]]:
    """Count the distinct experts each chunk activates for one layer.

    Returns ``(histograms, token_counts, chunk_counts)`` keyed by chunk plan;
    each histogram is indexed by the number of distinct experts in a chunk.
    """
    import numpy as np

    n_experts = int(dims["n_experts"])
    splits = sorted({key[0] for key in plans})
    truths = {
        split: read_cached_topk(cache_dir, cache_manifest, split, layer)
        for split in splits
    }

    histograms: dict[PlanKey, Any] = {}
    token_counts: dict[PlanKey, int] = {}
    chunk_counts: dict[PlanKey, int] = {}
    demand = np.zeros(n_experts, dtype=bool)

    for plan_key, plan in plans.items():
        split, _, _ = plan_key
        truth = truths[split]
        counts = histograms.get(plan_key)
        if counts is None:
            counts = np.zeros(n_experts + 1, dtype=np.int64)
            histograms[plan_key] = counts
        for chunk in plan.chunks:
            demand[:] = False
            demand[truth[chunk].ravel()] = True
            counts[int(np.count_nonzero(demand))] += 1
            token_counts[plan_key] = token_counts.get(plan_key, 0) + int(chunk.shape[0])
            chunk_counts[plan_key] = chunk_counts.get(plan_key, 0) + 1
    return histograms, token_counts, chunk_counts


# --------------------------------------------------------------------------- #
# Aggregation and payload assembly
# --------------------------------------------------------------------------- #
def _accumulate(
    total: dict[str, dict],
    layer: int,
    histograms: dict[PlanKey, Any],
    token_counts: dict[PlanKey, int],
    chunk_counts: dict[PlanKey, int],
) -> None:
    """Fold one layer's histograms and counters into the run accumulator."""
    for (split, portion, chunk_size), value in histograms.items():
        key: LayerKey = (split, portion, chunk_size, layer)
        total["hist"][key] = value
    for (split, portion, chunk_size), value in token_counts.items():
        total["tokens"][(split, portion, chunk_size, layer)] = int(value)
    for (split, portion, chunk_size), value in chunk_counts.items():
        total["chunks"][(split, portion, chunk_size, layer)] = int(value)


def finalize(
    cfg: dict[str, Any],
    run_id: str,
    layers: list[int],
    missing: list[int],
    dims: dict[str, int],
    total: dict[str, dict],
    diagnostics: dict[PlanKey, dict[str, float]],
) -> dict[str, Any]:
    """Assemble the per-layer table, per-combo stats and JSON summary."""
    import numpy as np

    n_experts = int(dims["n_experts"])
    top_k = int(dims["top_k"])

    ordering = sorted(total["hist"], key=lambda key: (key[0], key[1], key[2], key[3]))
    rows: list[dict[str, Any]] = []
    pooled_hist: dict[PlanKey, Any] = {}
    layer_means: dict[PlanKey, list[float]] = {}
    for split, portion, chunk_size, layer in ordering:
        hist = total["hist"][(split, portion, chunk_size, layer)]
        stats = distinct_stats(hist)
        chunks = int(total["chunks"].get((split, portion, chunk_size, layer), 0))
        tokens = int(total["tokens"].get((split, portion, chunk_size, layer), 0))
        rows.append(
            {
                "split": split,
                "decode_portion": portion,
                "chunk_size": int(chunk_size),
                "layer": int(layer),
                "n_chunks": chunks,
                "mean_tokens": round(tokens / chunks, 4) if chunks else 0.0,
                "mean_distinct": stats["mean_distinct"],
                "p50": stats["distinct_p50"],
                "p90": stats["distinct_p90"],
                "max": stats["distinct_max"],
                "distinct_fraction": stats["distinct_fraction"],
                "ceiling": min(int(chunk_size) * top_k, n_experts),
            }
        )
        combo: PlanKey = (split, portion, int(chunk_size))
        current = pooled_hist.get(combo)
        pooled_hist[combo] = hist if current is None else current + hist
        layer_means.setdefault(combo, []).append(stats["mean_distinct"])

    combos: dict[str, Any] = {}
    for combo in sorted(pooled_hist, key=lambda key: (key[0], key[1], key[2])):
        split, portion, chunk_size = combo
        pooled = dict(distinct_stats(pooled_hist[combo]))
        means = np.asarray(layer_means[combo], dtype=np.float64)
        pooled["n_layers"] = int(means.shape[0])
        pooled["layer_mean_distinct"] = round(float(means.mean()), 6)
        pooled["layer_p10"] = round(float(np.percentile(means, 10)), 6)
        pooled["layer_p90"] = round(float(np.percentile(means, 90)), 6)
        combos[f"{split}|{portion}|{chunk_size}"] = pooled

    # Equal-weight-per-split merge for the single-panel article line chart: each
    # split contributes 50%, so per-layer means are averaged across splits before
    # the across-layer band is taken. Per-split detail stays in ``rows``/``combos``.
    merged_values: dict[tuple[float, int], dict[int, list[float]]] = {}
    for row in rows:
        group = merged_values.setdefault(
            (float(row["decode_portion"]), int(row["chunk_size"])), {}
        )
        group.setdefault(int(row["layer"]), []).append(float(row["mean_distinct"]))
    merged: dict[str, Any] = {}
    for (portion, chunk_size), by_layer in sorted(merged_values.items()):
        per_layer = np.asarray(
            [float(np.mean(values)) for _, values in sorted(by_layer.items())],
            dtype=np.float64,
        )
        merged[f"{portion}|{chunk_size}"] = {
            "n_layers": int(per_layer.shape[0]),
            "layer_mean_distinct": round(float(per_layer.mean()), 6),
            "layer_p10": round(float(np.percentile(per_layer, 10)), 6),
            "layer_p90": round(float(np.percentile(per_layer, 90)), 6),
        }

    histograms = {
        f"{split}|{portion}|{chunk_size}|L{layer}": hist
        for (split, portion, chunk_size, layer), hist in total["hist"].items()
    }

    summary = {
        "run_id": run_id,
        "created_at": utc_now(),
        "config": cfg,
        "n_layers": len(layers),
        "layers": layers,
        "missing_layers": missing,
        "n_experts": n_experts,
        "top_k": top_k,
        "chunk_sizes": sorted(int(b) for b in cfg["simulation"]["chunk_sizes"]),
        "combos": combos,
        "merged": merged,
        "diagnostics": {
            f"{split}|{portion}|{chunk_size}": stats
            for (split, portion, chunk_size), stats in sorted(
                diagnostics.items(), key=lambda item: item[0]
            )
        },
    }
    return {
        "summary": summary,
        "rows": rows,
        "histograms": histograms,
    }
