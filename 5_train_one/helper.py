"""Data access, config, loss and metrics for the single-variant prototype.

Grounding. ``3_harvest`` writes, per slice, an fp8 ``expanded`` activation
``[T, L, hc_mult, dim]``, an fp8 ``compressed`` activation ``[T, L, dim]`` (the
exact MoE router input), the fp16 reference score vector
``scores = sqrt(softplus(Wx))`` ``[T, L, n_experts]``, and the reference top-k
``topk_ids`` ``[T, L, top_k]``.

A predictor for layer ``L`` reads the activation of layer ``L - distance`` and
predicts layer ``L``'s 256-way score vector. The objective is the explicit KL
between the two normalized *pre-bias* score vectors; the practical metric is the
recall of the reference top-k experts, selecting on ``score + bias`` exactly as
the reference gate does (the bias never enters the KL).

Nothing here imports torch / numpy / yaml at module scope: the local entrypoint
imports ``main`` (and therefore this module) without those stacks.
"""

from __future__ import annotations

import json
import os
from typing import Any

HARVEST_DIR = "/harvest"
TRAINING_DIR = "/training"
MANIFEST_PATH = os.path.join(HARVEST_DIR, "manifest.json")

ARCHS = ("lowrank", "swiglu", "mlp")
INPUT_KINDS = ("compressed", "expanded")
DATASETS = ("yi30-think", "yi30-nothink", "terminus2", "dsh")

CONFIG_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "data": {
        "harvest_dir": HARVEST_DIR,
        "train": {"datasets": None, "max_slices": None, "max_tokens": 72000},
        "eval": {"split": "test", "datasets": None, "max_tokens": 40000},
    },
    "task": {"layer": 20, "distance": 1, "input": "compressed"},
    "model": {"arch": "lowrank", "rank": 128, "hidden": 4096},
    "optim": {
        "lr": 3.0e-4,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "epochs": 20,
        "batch_size": 8192,
        "scheduler": "cosine",
    },
    "eval": {"every": 1, "chunk_tokens": 16384, "ks": [6, 12, 24]},
    "output": {"volume_dir": TRAINING_DIR, "save_best": True},
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_text: str) -> dict[str, Any]:
    """Parse the YAML config over the built-in defaults and sanity-check it."""
    import yaml

    parsed = yaml.safe_load(config_text) or {}
    cfg = _merge(CONFIG_DEFAULTS, parsed)
    if cfg["model"]["arch"] not in ARCHS:
        raise ValueError(
            f"model.arch must be one of {ARCHS}, got {cfg['model']['arch']!r}"
        )
    if cfg["task"]["input"] not in INPUT_KINDS:
        raise ValueError(
            f"task.input must be one of {INPUT_KINDS}, got {cfg['task']['input']!r}"
        )
    if cfg["optim"]["scheduler"] not in ("cosine", "none"):
        raise ValueError("optim.scheduler must be 'cosine' or 'none'")
    return cfg


# --------------------------------------------------------------------------- #
# Manifest / dimensions
# --------------------------------------------------------------------------- #
def _manifest(harvest_dir: str = HARVEST_DIR) -> dict[str, Any]:
    with open(os.path.join(harvest_dir, "manifest.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _dims(manifest: dict[str, Any]) -> dict[str, int]:
    record = manifest["records"][0]
    return {
        "n_layers": int(record["n_layers"]),
        "dim": int(record["dim"]),
        "hc_mult": int(record["hc_mult"]),
        "n_experts": int(record["n_routed_experts"]),
        "top_k": int(record["top_k"]),
        "n_hash_layers": int(manifest.get("n_hash_layers", 0)),
    }


def _select_records(
    manifest: dict[str, Any],
    split: str,
    datasets: list[str] | None,
    max_slices: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Records of one split, optionally filtered by dataset and randomly capped."""
    import random

    records = [r for r in manifest["records"] if r.get("split") == split]
    if datasets is not None:
        records = [r for r in records if r.get("dataset_id") in set(datasets)]
    if max_slices is not None and len(records) > max_slices:
        records = random.Random(seed).sample(records, max_slices)
    return records


def input_dim(input_kind: str, dim: int, hc_mult: int) -> int:
    """Feature width fed to the predictor for one input encoding."""
    return dim * hc_mult if input_kind == "expanded" else dim


def load_router_bias(manifest: dict[str, Any], harvest_dir: str = HARVEST_DIR) -> Any:
    """The per-layer expert-selection bias ``[n_layers, n_experts]`` fp32.

    DeepSeek adds this to the pre-bias score *before* top-k selection (see
    ``checkpoints/inference/model.py`` ``Gate.forward``); it shifts which experts
    are selected but not the routing weights. Zero rows mark the hash layers.
    """
    import torch
    from safetensors.torch import load_file

    path = manifest.get("router_bias_path") or os.path.join(
        harvest_dir, "router_bias.safetensors"
    )
    return load_file(path)["bias"].to(torch.float32)


# --------------------------------------------------------------------------- #
# Slice reads (lazy: only the requested layer is materialized)
# --------------------------------------------------------------------------- #
def _read_activation(record: dict[str, Any], layer: int, input_kind: str) -> Any:
    """One layer's activation for a slice, dequantized to bf16."""
    import torch
    from safetensors import safe_open

    key = "expanded" if input_kind == "expanded" else "compressed"
    scale = (
        record["expanded_scale"]
        if input_kind == "expanded"
        else record["compressed_scale"]
    )
    with safe_open(record["path"], framework="pt") as handle:
        sub = handle.get_slice(key)[:, layer : layer + 1].squeeze(1)
    sub = sub.to(torch.float32) * float(scale)
    if input_kind == "expanded":
        sub = sub.reshape(sub.shape[0], -1)
    return sub.to(torch.bfloat16)


def _read_target(record: dict[str, Any], layer: int) -> tuple[Any, Any]:
    """One layer's target scores (fp32) and true top-k ids (int64)."""
    import torch
    from safetensors import safe_open

    with safe_open(record["path"], framework="pt") as handle:
        scores = (
            handle.get_slice("scores")[:, layer : layer + 1]
            .squeeze(1)
            .to(torch.float32)
        )
        topk = (
            handle.get_slice("topk_ids")[:, layer : layer + 1]
            .squeeze(1)
            .to(torch.int64)
        )
    return scores, topk


def load_frames(
    records: list[dict[str, Any]],
    source_layer: int,
    target_layer: int,
    input_kind: str,
    max_tokens: int | None,
    seed: int,
) -> tuple[Any, Any, Any]:
    """Load and concat every slice's input/target, then cap tokens at random.

    The random token subsample is seeded, so a given ``(records, seed)`` always
    yields the same training (or evaluation) set across epochs and reruns.

    Returns:
        ``(x, y, ids)`` where ``x`` is ``[N, d_in]`` bf16, ``y`` is
        ``[N, n_experts]`` fp32 and ``ids`` is ``[N, top_k]`` int64.
    """
    import torch

    xs: list[Any] = []
    ys: list[Any] = []
    ids: list[Any] = []
    for record in records:
        xs.append(_read_activation(record, source_layer, input_kind))
        y, topk = _read_target(record, target_layer)
        ys.append(y)
        ids.append(topk)
    x = torch.cat(xs)
    y = torch.cat(ys)
    true_ids = torch.cat(ids)

    n = x.shape[0]
    if max_tokens is not None and n > max_tokens:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator)[:max_tokens]
        x = x[perm]
        y = y[perm]
        true_ids = true_ids[perm]
    return x, y, true_ids


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def kl_divergence(pred_scores: Any, target_scores: Any, eps: float = 1e-8) -> Any:
    """KL(target_norm || pred_norm) between two normalized score vectors.

    Both vectors are brought onto the simplex. The target is the raw harvested
    ``sqrt(softplus(Wx))``; normalizing it is what turns the score vector into
    the distribution the KL is defined over.
    """
    q = pred_scores / pred_scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = target_scores / target_scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=-1)
    return kl.mean()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def recall(pred_scores: Any, true_ids: Any, k: int, bias: Any = None) -> float:
    """Mean recall of the true selected experts in the predicted top-k.

    Selection follows the reference gate: when ``bias`` is given the top-k is
    taken from ``pred_scores + bias`` (DeepSeek adds the per-layer selection bias
    before top-k), otherwise from ``pred_scores``. The bias affects only which
    experts are selected, never the routing weights.

    ``1/N * sum_i |needed_i and predicted_i| / |needed_i|`` where ``needed`` is
    the reference top-k set and ``predicted`` is the top-k of the selection
    scores. When ``k`` exceeds the needed-set size this is a coverage measure:
    how many of the needed experts a wider speculative load would capture.
    """
    true_k = int(true_ids.shape[1])
    selection = pred_scores if bias is None else pred_scores + bias
    pred_ids = selection.topk(min(k, int(selection.shape[1])), dim=-1).indices
    match = (pred_ids[:, :, None] == true_ids[:, None, :]).any(dim=1)
    return float(match.sum(dim=1).float().mean() / true_k)


def score_metrics(
    pred_scores: Any,
    target_scores: Any,
    true_ids: Any,
    ks: list[int],
    bias: Any = None,
) -> dict[str, float]:
    """KL on the pre-bias scores, plus selection recall@k for every ``k``.

    The KL target is the harvested pre-bias score vector, so ``bias`` is only
    forwarded to the recall (top-k selection), never to the divergence.
    """
    metrics: dict[str, float] = {
        "kl": float(kl_divergence(pred_scores, target_scores).detach())
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall(pred_scores, true_ids, k, bias=bias)
    return metrics
