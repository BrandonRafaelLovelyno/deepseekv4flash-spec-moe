"""Config, cache, loss and metrics for the train-all routing prototype.

Same two-stage data path as ``5_train_one``, widened from one layer to every
layer. ``3_harvest`` writes, per slice, an fp8 ``expanded``
``[T, L, hc_mult, dim]``, an fp8 ``compressed`` ``[T, L, dim]`` (the exact MoE
router input), the fp16 ``scores = sqrt(softplus(Wx))`` ``[T, L, n_experts]`` and
the reference top-k ``topk_ids`` ``[T, L, top_k]``. Reading one layer out of a
slice is *strided* over the network-mounted harvest volume, so a CPU
``populate_cache`` job re-lays every layer the run needs out contiguously once.

A predictor for target layer ``L`` reads the activation of ``source = L -
min(L, distance)`` -- i.e. it looks back at most ``distance`` layers, clamped at
layer 0 for the early layers -- and predicts ``L``'s 256-way score vector. The
objective is the explicit KL between the two normalized *pre-bias* score
vectors; the practical metric is the recall of the reference top-k experts,
selecting on ``score + bias`` exactly as the reference gate does.

Nothing here imports torch / numpy / yaml at module scope: the local entrypoint
imports ``main`` (and therefore this module) without those stacks.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

HARVEST_DIR = "/harvest"
TRAINING_DIR = "/training"
CACHE_DIR = "/cache"
MANIFEST_PATH = os.path.join(HARVEST_DIR, "manifest.json")

ARCHS = ("lowrank", "swiglu", "mlp")
INPUT_KINDS = ("compressed", "expanded")
SPLITS = ("train", "test")
DATASETS = ("yi30-think", "yi30-nothink", "terminus2", "dsh")
READY_RATIOS = (0.25, 0.5, 0.75)

CONFIG_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "data": {
        "harvest_dir": HARVEST_DIR,
        "hot_experts_path": None,
        "train": {"datasets": None, "max_slices": None, "max_tokens": 72000},
        "eval": {"split": "test", "datasets": None, "max_tokens": 40000},
    },
    "cache": {"dir": CACHE_DIR, "rebuild": False},
    "task": {"layers": None, "distance": 1, "input": "compressed"},
    "model": {"arch": "lowrank", "rank": 128, "hidden": 4096},
    "optim": {
        "lr": 3.0e-4,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "epochs": 20,
        "batch_size": 8192,
        "scheduler": "cosine",
    },
    "eval": {
        "every": 1,
        "chunk_tokens": 16384,
        "ks": [6, 12, 24],
        "ready_ratios": [0.25, 0.5, 0.75],
    },
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
    if int(cfg["task"]["distance"]) < 1:
        raise ValueError("task.distance must be >= 1")
    if cfg["optim"]["scheduler"] not in ("cosine", "none"):
        raise ValueError("optim.scheduler must be 'cosine' or 'none'")
    return cfg


def resolve_tasks(cfg: dict[str, Any], dims: dict[str, int]) -> list[dict[str, Any]]:
    """Resolve and validate the per-layer task geometry from the config.

    ``task.layers`` is ``None`` (every non-hash layer, ``n_hash_layers ..
    n_layers-1``) or an explicit list. Each target layer ``L`` looks back
    ``min(L, distance)`` layers -- clamped so the source never precedes layer 0
    -- giving ``source_layer = max(0, L - distance)``. Returns one task dict per
    target, sorted by layer.
    """
    task = cfg["task"]
    distance = int(task["distance"])
    input_kind = task["input"]
    layers_cfg = task["layers"]
    if layers_cfg is None:
        targets = list(range(dims["n_hash_layers"], dims["n_layers"]))
    else:
        targets = sorted({int(layer) for layer in layers_cfg})

    tasks: list[dict[str, Any]] = []
    for layer in targets:
        if layer >= dims["n_layers"]:
            raise ValueError(f"layer {layer} >= n_layers {dims['n_layers']}")
        if layer < dims["n_hash_layers"]:
            raise ValueError(f"layer {layer} is a hash layer (token-id routing)")
        effective = min(layer, distance)
        tasks.append(
            {
                "layer": layer,
                "distance": effective,
                "input_kind": input_kind,
                "source_layer": layer - effective,
            }
        )
    if not tasks:
        raise ValueError("no target layers resolved; check task.layers")
    return tasks


# --------------------------------------------------------------------------- #
# Harvest manifest / dimensions
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


def _harvest_fingerprint(manifest: dict[str, Any]) -> str:
    """Short digest of the slice list, used to invalidate a stale cache."""
    digest = hashlib.sha256()
    for record in manifest["records"]:
        digest.update(
            f"{record['path']}:{record['n_tokens']}:{record['split']}".encode()
        )
    return digest.hexdigest()[:16]


def _build_splits(manifest: dict[str, Any]) -> dict[str, Any]:
    """Per-split slice table with cumulative token offsets, in manifest order."""
    splits: dict[str, Any] = {}
    for split in SPLITS:
        offset = 0
        slices = []
        for record in manifest["records"]:
            if record["split"] != split:
                continue
            n_tokens = int(record["n_tokens"])
            slices.append(
                {
                    "dataset_id": record["dataset_id"],
                    "document_id": record["document_id"],
                    "n_tokens": n_tokens,
                    "offset": offset,
                }
            )
            offset += n_tokens
        splits[split] = {"n_tokens": offset, "slices": slices}
    return splits


# --------------------------------------------------------------------------- #
# 4_analysis expert counts (hottest-k resident sets)
# --------------------------------------------------------------------------- #
def expert_counts_path(path: str | None = None, harvest_dir: str = HARVEST_DIR) -> str:
    """Resolve the ``4_analysis`` expert-counts ``.npy`` to copy into the cache.

    ``path`` may name the ``train_expert_counts.npy`` file or its directory
    (``.../expert_ranking``). When it is ``None`` the newest
    ``<harvest_dir>/analysis/*/expert_ranking`` run wins.
    """
    import glob

    if path is not None:
        return (
            os.path.join(path, "train_expert_counts.npy")
            if os.path.isdir(path)
            else path
        )
    pattern = os.path.join(
        harvest_dir, "analysis", "*", "expert_ranking", "train_expert_counts.npy"
    )
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"no expert counts under {os.path.dirname(pattern)}; run 4_analysis "
            "first or set data.hot_experts_path"
        )
    return max(candidates, key=os.path.getmtime)


def load_expert_counts(
    path: str | None = None, harvest_dir: str = HARVEST_DIR
) -> Any:
    """Load ``4_analysis``'s per-layer expert usage counts ``[n_layers, n_experts]``."""
    import numpy as np

    return np.load(expert_counts_path(path, harvest_dir))


# --------------------------------------------------------------------------- #
# Cache layout
# --------------------------------------------------------------------------- #
def load_cache_manifest(cache_dir: str) -> dict[str, Any] | None:
    """Load the cache manifest, or ``None`` when the cache has not been built."""
    path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _entry_key(kind: str, input_kind: str | None, split: str | None, layer: int) -> str:
    if kind == "bias":
        return f"bias:{layer}"
    if kind == "expert_counts":
        return f"expert_counts:{layer}"
    if kind == "activation":
        return f"activation:{input_kind}:{split}:{layer}"
    return f"{kind}:{split}:{layer}"


def _entry_path(
    kind: str, input_kind: str | None, split: str | None, layer: int
) -> str:
    if kind == "bias":
        return f"bias/L{layer}.safetensors"
    if kind == "expert_counts":
        return f"expert_counts/L{layer}.safetensors"
    if kind == "activation":
        return f"activation/{input_kind}/{split}/L{layer}.safetensors"
    return f"{kind}/{split}/L{layer}.safetensors"


def populate_cache(config_text: str) -> dict[str, Any]:
    """Re-lay every layer the run needs into the contiguous cache (CPU).

    For all resolved target layers this writes, per split, the source-layer
    activation (fp8 + per-slice scale) and the target-layer scores (fp16) and
    top-k ids (uint8), plus each target layer's selection bias and expert
    counts. Entries already present and matching the harvest fingerprint are
    left alone, so this is additive on a shared cache volume.
    """
    import torch
    from safetensors.torch import load_file, save_file

    cfg = load_config(config_text)
    harvest_dir = cfg["data"]["harvest_dir"]
    cache_dir = cfg["cache"]["dir"]
    manifest = _manifest(harvest_dir)
    dims = _dims(manifest)
    fingerprint = _harvest_fingerprint(manifest)

    tasks = resolve_tasks(cfg, dims)
    input_kind = cfg["task"]["input"]
    target_layers = sorted({task["layer"] for task in tasks})
    source_layers = sorted({task["source_layer"] for task in tasks})

    os.makedirs(cache_dir, exist_ok=True)
    cache = load_cache_manifest(cache_dir)
    reset = (
        cache is None
        or cache.get("harvest_fingerprint") != fingerprint
        or bool(cfg["cache"]["rebuild"])
    )
    if reset:
        cache = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "harvest_fingerprint": fingerprint,
            "dims": dims,
            "splits": _build_splits(manifest),
            "entries": {},
        }

    records_by_split = {
        split: [r for r in manifest["records"] if r["split"] == split]
        for split in SPLITS
    }
    built: list[str] = []
    for split in SPLITS:
        for kind, ik, layers in (
            ("activation", input_kind, source_layers),
            ("scores", None, target_layers),
            ("topk", None, target_layers),
        ):
            for layer in layers:
                key = _entry_key(kind, ik, split, layer)
                if key in cache["entries"] and not reset:
                    continue
                print(f"[cache] building {key}", flush=True)
                _build_entry(
                    cache, cache_dir, kind, ik, split, layer, records_by_split[split]
                )
                built.append(key)

    bias_all = None
    counts_all = None
    counts_source = None
    for layer in target_layers:
        bias_key = _entry_key("bias", None, None, layer)
        if bias_key not in cache["entries"] or reset:
            if bias_all is None:
                bias_path = manifest.get("router_bias_path") or os.path.join(
                    harvest_dir, "router_bias.safetensors"
                )
                bias_all = load_file(bias_path)["bias"]
            print(f"[cache] building {bias_key}", flush=True)
            row = bias_all[layer].to(torch.float32).contiguous()
            relative = _entry_path("bias", None, None, layer)
            target = os.path.join(cache_dir, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            save_file({"bias": row}, target)
            cache["entries"][bias_key] = {
                "path": relative,
                "kind": "bias",
                "dtype": "float32",
                "n_experts": int(row.numel()),
            }
            built.append(bias_key)

        counts_key = _entry_key("expert_counts", None, None, layer)
        if counts_key not in cache["entries"] or reset:
            if counts_all is None:
                counts_source = expert_counts_path(
                    cfg["data"]["hot_experts_path"], harvest_dir
                )
                counts_all = load_expert_counts(counts_source)
                if tuple(counts_all.shape) != (dims["n_layers"], dims["n_experts"]):
                    raise ValueError(
                        f"expert counts shape {tuple(counts_all.shape)} != "
                        f"({dims['n_layers']}, {dims['n_experts']}); "
                        "stale 4_analysis run?"
                    )
            print(f"[cache] building {counts_key}", flush=True)
            row = torch.as_tensor(counts_all[layer], dtype=torch.int64).contiguous()
            relative = _entry_path("expert_counts", None, None, layer)
            target = os.path.join(cache_dir, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            save_file({"counts": row}, target)
            cache["entries"][counts_key] = {
                "path": relative,
                "kind": "expert_counts",
                "dtype": "int64",
                "n_experts": int(row.numel()),
                "source": counts_source,
            }
            built.append(counts_key)

    with open(
        os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(cache, handle, indent=2)

    return {
        "cache_dir": cache_dir,
        "reset": reset,
        "built": built,
        "n_entries": len(cache["entries"]),
    }


def _build_entry(
    cache: dict[str, Any],
    cache_dir: str,
    kind: str,
    input_kind: str | None,
    split: str,
    layer: int,
    records: list[dict[str, Any]],
) -> None:
    """Concatenate one (kind, layer) across a split's slices and write it."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    parts: list[Any] = []
    scales: list[float] = []
    for record in records:
        with safe_open(record["path"], framework="pt") as handle:
            if kind == "activation":
                key = "expanded" if input_kind == "expanded" else "compressed"
                whole = handle.get_tensor(key)
                sub = whole[:, layer]
                if input_kind == "expanded":
                    sub = sub.reshape(sub.shape[0], -1)
                parts.append(sub)
                scale_key = (
                    "expanded_scale" if input_kind == "expanded" else "compressed_scale"
                )
                scales.append(float(record[scale_key]))
                del whole
            elif kind == "scores":
                whole = handle.get_tensor("scores")
                parts.append(whole[:, layer].to(torch.float16))
                del whole
            else:  # topk
                whole = handle.get_tensor("topk_ids")
                parts.append(whole[:, layer].to(torch.uint8))
                del whole

    data = torch.cat(parts).contiguous()
    payload: dict[str, Any] = {"data": data}
    if kind == "activation":
        payload["scale"] = torch.tensor(scales, dtype=torch.float32)
    relative = _entry_path(kind, input_kind, split, layer)
    target = os.path.join(cache_dir, relative)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    save_file(payload, target)
    cache["entries"][_entry_key(kind, input_kind, split, layer)] = {
        "path": relative,
        "kind": kind,
        "dtype": str(data.dtype),
        "n_slices": len(records),
        "n_tokens": int(data.shape[0]),
        "dim": int(data.shape[1]),
    }


# --------------------------------------------------------------------------- #
# Cache reads (GPU training)
# --------------------------------------------------------------------------- #
def load_cached_bias(cache_dir: str, manifest: dict[str, Any], layer: int) -> Any:
    """The cached per-layer expert-selection bias ``[n_experts]`` fp32."""
    import torch
    from safetensors.torch import load_file

    entry = manifest["entries"][_entry_key("bias", None, None, layer)]
    return load_file(os.path.join(cache_dir, entry["path"]))["bias"].to(torch.float32)


def load_cached_expert_counts(
    cache_dir: str, manifest: dict[str, Any], layer: int
) -> Any:
    """The cached target-layer expert usage counts ``[n_experts]`` int64."""
    from safetensors.torch import load_file

    entry = manifest["entries"][_entry_key("expert_counts", None, None, layer)]
    return load_file(os.path.join(cache_dir, entry["path"]))["counts"].to("cpu")


def _select_slices(
    slices: list[dict[str, Any]],
    datasets: list[str] | None,
    max_slices: int | None,
    seed: int,
) -> list[int]:
    """Indices of the slices to use: filter by dataset, random cap (seeded)."""
    import random

    indices = list(range(len(slices)))
    if datasets is not None:
        keep = set(datasets)
        indices = [i for i in indices if slices[i]["dataset_id"] in keep]
    if max_slices is not None and len(indices) > max_slices:
        indices = random.Random(seed).sample(indices, max_slices)
    return indices


def load_cache_frames(
    cache_dir: str,
    manifest: dict[str, Any],
    split: str,
    source_layer: int,
    target_layer: int,
    input_kind: str,
    datasets: list[str] | None,
    max_slices: int | None,
    max_tokens: int | None,
    seed: int,
) -> tuple[Any, Any, Any]:
    """Load one split's source activation and target from the contiguous cache.

    Token-level random subsampling (``max_tokens``) is seeded, so a given
    ``(selection, seed)`` always yields the same rows across epochs and reruns.

    Returns:
        ``(x, y, ids)`` where ``x`` is ``[N, d_in]`` bf16, ``y`` is
        ``[N, n_experts]`` fp32 and ``ids`` is ``[N, top_k]`` int64.
    """
    import torch
    from safetensors.torch import load_file

    slices = manifest["splits"][split]["slices"]
    selected = _select_slices(slices, datasets, max_slices, seed)
    if not selected:
        raise ValueError(f"no {split} slices selected")
    rows = torch.cat(
        [
            torch.arange(
                slices[i]["offset"], slices[i]["offset"] + slices[i]["n_tokens"]
            )
            for i in selected
        ]
    )

    def read(kind: str, ik: str | None, lyr: int) -> dict[str, Any]:
        entry = manifest["entries"][_entry_key(kind, ik, split, lyr)]
        return load_file(os.path.join(cache_dir, entry["path"]))

    activation = read("activation", input_kind, source_layer)
    x = activation["data"][rows].to(torch.float32)
    token_scale = torch.repeat_interleave(
        activation["scale"][selected],
        torch.tensor([slices[i]["n_tokens"] for i in selected], dtype=torch.long),
    )
    x = (x * token_scale[:, None]).to(torch.bfloat16)
    y = read("scores", None, target_layer)["data"][rows].to(torch.float32)
    true_ids = read("topk", None, target_layer)["data"][rows].to(torch.int64)

    n = x.shape[0]
    if max_tokens is not None and n > max_tokens:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator)[:max_tokens]
        x, y, true_ids = x[perm], y[perm], true_ids[perm]
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


def resident_sets(
    counts_layer: Any, ratios: tuple[float, ...] = READY_RATIOS
) -> dict[float, dict[str, Any]]:
    """Hottest-k resident set per ratio for one layer's expert counts.

    Returns ``{ratio: {"k": int, "ids": list[int], "mask": bool tensor}}`` where
    ``ids`` is hottest-first and ``mask`` is ``[n_experts]``. Mirrors
    ``4_analysis``'s ``_resident_masks`` (``k = round(ratio * n_experts)``).
    ``counts_layer`` may be a numpy array or a CPU tensor.
    """
    import numpy as np
    import torch

    counts_layer = np.asarray(counts_layer, dtype=np.int64)
    n_experts = int(counts_layer.shape[0])
    out: dict[float, dict[str, Any]] = {}
    for ratio in ratios:
        k = int(round(ratio * n_experts))
        order = np.argsort(counts_layer)[::-1][:k].copy()
        mask = torch.zeros(n_experts, dtype=torch.bool)
        mask[torch.as_tensor(order, dtype=torch.long)] = True
        out[ratio] = {"k": k, "ids": [int(i) for i in order], "mask": mask}
    return out


def resident_recall(true_ids: Any, resident_mask: Any) -> float:
    """Mean fraction of each token's true experts already resident.

    The prediction-free baseline: how much of the needed set the static hot set
    covers on its own (``k``-independent).
    """
    true_k = int(true_ids.shape[1])
    covered = resident_mask[true_ids]
    return float(covered.sum(dim=1).float().mean() / true_k)


def ready_recall(
    pred_scores: Any, true_ids: Any, resident_mask: Any, k: int, bias: Any = None
) -> float:
    """Recall when the resident hot set is unioned with the predicted top-k.

    Selection follows the reference gate (top-k of ``pred_scores + bias``). The
    retrieved set for a token is that prediction *plus* the layer's resident hot
    experts; the score is the fraction of the token's true experts it covers.
    """
    true_k = int(true_ids.shape[1])
    selection = pred_scores if bias is None else pred_scores + bias
    pred_ids = selection.topk(min(k, int(selection.shape[1])), dim=-1).indices
    in_pred = (pred_ids[:, :, None] == true_ids[:, None, :]).any(dim=1)
    retrieved = in_pred | resident_mask[true_ids]
    return float(retrieved.sum(dim=1).float().mean() / true_k)


def score_metrics(
    pred_scores: Any,
    target_scores: Any,
    true_ids: Any,
    ks: list[int],
    bias: Any = None,
    resident: dict[float, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """KL on the pre-bias scores, plus selection recall@k for every ``k``.

    The KL target is the harvested pre-bias score vector, so ``bias`` is only
    forwarded to the recall (top-k selection), never to the divergence. When
    ``resident`` is given (from :func:`resident_sets`), the per-ratio
    ``ready_recall_r*@k`` (hot set unioned with the prediction) and the
    prediction-free ``resident_recall_r*`` baseline are added too.
    """
    metrics: dict[str, float] = {
        "kl": float(kl_divergence(pred_scores, target_scores).detach())
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall(pred_scores, true_ids, k, bias=bias)
    if resident is not None:
        for ratio, entry in resident.items():
            tag = f"r{int(round(ratio * 100))}"
            metrics[f"resident_recall_{tag}"] = resident_recall(
                true_ids, entry["mask"]
            )
            for k in ks:
                metrics[f"ready_recall_{tag}@{k}"] = ready_recall(
                    pred_scores, true_ids, entry["mask"], k, bias=bias
                )
    return metrics
