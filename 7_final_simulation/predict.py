"""GPU prediction stage: run each layer's predictor and cache its top-k experts.

This is the **only** GPU user in ``7_final_simulation``. It reads the source
activation from the contiguous cache, forwards the layer's predictor, and writes
the per-token predicted top-k to a small prediction cache. The CPU sweep
(``simulation.simulate``) then reads that cache and never touches the GPU.

Predictions depend only on ``(training_run_id, cache, prediction_k, apply_bias)``,
so the cache is reusable across every simulation knob and is skipped when valid
unless ``force`` (or ``simulation.force_predict``) is set.

torch / numpy are imported inside functions so the local entrypoint can import
this module without the GPU stack.
"""

from __future__ import annotations

import json
import os
from typing import Any

from helper import (
    cache_dims,
    load_cache_manifest,
    load_config,
    load_prediction_manifest,
    load_run_summary,
    prediction_dir,
    prediction_fingerprint,
    read_cached_activation,
    read_cached_bias,
    resolve_layers,
    resolve_run_dir,
    utc_now,
    write_cached_prediction,
)


def predict_all(config_text: str, force: bool = False) -> dict[str, Any]:
    """Predict every resolved layer for both splits, caching what is missing."""
    cfg = load_config(config_text)
    force = bool(force or cfg["simulation"].get("force_predict", False))
    cache_dir = cfg["cache"]["dir"]
    cache_manifest = load_cache_manifest(cache_dir)
    if cache_manifest is None:
        raise ValueError(f"no cache at {cache_dir}; run 6_train_all first")
    dims = cache_dims(cache_manifest)

    run_dir, training_run_id = resolve_run_dir(
        cfg["training"]["dir"], cfg["training"]["run_id"]
    )
    run_summary = load_run_summary(run_dir)
    metas, skipped = resolve_layers(cfg, run_dir, run_summary, dims)

    sim = cfg["simulation"]
    prediction_k = int(sim["prediction_k"])
    apply_bias = bool(sim["apply_bias"])
    fingerprint = prediction_fingerprint(
        training_run_id, cache_manifest, prediction_k, apply_bias
    )
    pred_dir = prediction_dir(cfg["output"]["volume_dir"], training_run_id)

    manifest = load_prediction_manifest(pred_dir)
    fresh = manifest is not None and manifest.get("fingerprint") == fingerprint
    entries = dict(manifest.get("entries", {})) if fresh else {}

    splits = [str(split) for split in cfg["data"]["split"]]
    predicted: list[int] = []
    reused: list[int] = []
    for index, meta in enumerate(metas, start=1):
        layer = int(meta["layer"])
        if not force and entries.get(str(layer), {}).get("fingerprint") == fingerprint:
            reused.append(layer)
            print(f"[predict] L{layer:02d} cached", flush=True)
            continue
        print(
            f"[predict] {index}/{len(metas)} L{layer:02d} "
            f"src=L{meta['source_layer']:02d}",
            flush=True,
        )
        entries[str(layer)] = {
            "fingerprint": fingerprint,
            "splits": _predict_layer(
                cfg,
                cache_dir,
                cache_manifest,
                meta,
                splits,
                pred_dir,
                int(dims["n_experts"]),
                prediction_k,
                apply_bias,
            ),
        }
        predicted.append(layer)

    _write_manifest(
        pred_dir,
        training_run_id,
        fingerprint,
        prediction_k,
        apply_bias,
        cache_manifest,
        entries,
    )
    print(
        f"[predict] run={training_run_id} predicted={len(predicted)} "
        f"reused={len(reused)} dir={pred_dir}",
        flush=True,
    )
    return {
        "training_run_id": training_run_id,
        "predicted": predicted,
        "reused": reused,
        "skipped": skipped,
        "dir": pred_dir,
    }


def _predict_layer(
    cfg: dict[str, Any],
    cache_dir: str,
    cache_manifest: dict[str, Any],
    meta: dict[str, Any],
    splits: list[str],
    pred_dir: str,
    n_experts: int,
    prediction_k: int,
    apply_bias: bool,
) -> dict[str, Any]:
    """Forward one layer's predictor over every split and cache the results."""
    import torch
    from models import build_model
    from safetensors.torch import load_file

    model = (
        build_model(
            meta["arch"],
            int(meta["d_in"]),
            n_experts,
            rank=int(meta["rank"]),
            hidden=int(meta["hidden"]),
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(load_file(meta["checkpoint"]))

    bias = torch.as_tensor(
        read_cached_bias(cache_dir, cache_manifest, meta["layer"]),
        dtype=torch.float32,
    )
    bias = bias.cuda() if apply_bias else None

    result: dict[str, Any] = {}
    for split in splits:
        predicted = _predict_split(
            model, cache_dir, cache_manifest, split, meta, cfg, bias, prediction_k
        )
        write_cached_prediction(pred_dir, meta["layer"], split, predicted)
        result[split] = {"n_tokens": int(predicted.shape[0])}

    del model
    torch.cuda.empty_cache()
    return result


def _predict_split(
    model: Any,
    cache_dir: str,
    cache_manifest: dict[str, Any],
    split: str,
    meta: dict[str, Any],
    cfg: dict[str, Any],
    bias: Any,
    prediction_k: int,
) -> Any:
    """Predict every token of one split's top-k experts for this layer.

    The source activation is fp8 with a per-slice scale; rows are reconstructed
    lazily per forward chunk so the full fp32 activation is never materialised.
    """
    import numpy as np
    import torch

    forward_chunk = int(cfg["simulation"]["forward_chunk"])
    activation, scale = read_cached_activation(
        cache_dir,
        cache_manifest,
        split,
        meta["input_kind"],
        meta["source_layer"],
    )
    slice_counts = torch.tensor(
        [int(entry["n_tokens"]) for entry in cache_manifest["splits"][split]["slices"]],
        dtype=torch.long,
    )
    token_scale = torch.repeat_interleave(scale.to(torch.float32).cpu(), slice_counts)

    n_rows = int(activation.shape[0])
    out = np.empty((n_rows, prediction_k), dtype=np.uint8)
    with torch.no_grad():
        for start in range(0, n_rows, forward_chunk):
            end = min(start + forward_chunk, n_rows)
            x = (
                activation[start:end].to(torch.float32) * token_scale[start:end, None]
            ).cuda()
            scores = model(x)
            selection = scores if bias is None else scores + bias
            ids = selection.topk(prediction_k, dim=-1).indices
            out[start:end] = ids.to(torch.uint8).cpu().numpy()
    return out


def _write_manifest(
    pred_dir: str,
    training_run_id: str,
    fingerprint: str,
    prediction_k: int,
    apply_bias: bool,
    cache_manifest: dict[str, Any],
    entries: dict[str, Any],
) -> None:
    """Write the prediction-cache manifest describing what is cached."""
    os.makedirs(pred_dir, exist_ok=True)
    payload = {
        "created_at": utc_now(),
        "training_run_id": training_run_id,
        "fingerprint": fingerprint,
        "prediction_k": int(prediction_k),
        "apply_bias": bool(apply_bias),
        "cache_harvest_fingerprint": cache_manifest.get("harvest_fingerprint"),
        "entries": entries,
    }
    with open(os.path.join(pred_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
