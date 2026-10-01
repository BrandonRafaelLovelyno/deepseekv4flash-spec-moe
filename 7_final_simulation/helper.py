"""Config, checkpoint/cache access and shared statistics for the final simulation.

The simulation replays the per-layer routing predictors trained by
``6_train_all`` under a mixed chunked-prefill load. This module owns everything
two or more simulation modules need:

* the config (``load_config``) and the newest-run / per-layer checkpoint
  resolution (``resolve_run_dir`` / ``resolve_layers``);
* reading the contiguous cache -- source activation, target-layer truth top-k,
  selection bias and per-layer expert counts -- plus the harvest ``is_decode``
  labels the chunked-prefill streams are built from;
* the tidy per-combo distribution statistics every artifact writer shares.

torch / numpy / yaml are imported *inside* functions, never at module scope, so
``main.py`` stays importable on a machine without the scientific stack.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, TypedDict

HARVEST_DIR = "/harvest"
TRAINING_DIR = "/training"
CACHE_DIR = "/cache"
SIMULATION_DIR = "/simulation"

SPLITS = ("train", "test")
DATASETS = ("yi30-think", "yi30-nothink", "terminus2", "dsh")
READY_RATIOS = (0.25, 0.5, 0.75)
INPUT_KINDS = ("compressed", "expanded")
DISPERSALS = ("van_der_corput",)
PREFILL_ORDERS = ("sequential_per_session",)

CONFIG_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "training": {"dir": TRAINING_DIR, "run_id": None, "layers": None},
    "cache": {"dir": CACHE_DIR},
    "harvest": {"dir": HARVEST_DIR},
    "data": {"split": ["train", "test"], "datasets": None},
    "simulation": {
        "chunk_sizes": [64, 128, 256, 512],
        "decode_portions": [0.75, 0.5, 0.125],
        "fetch_counts": [0, 8, 10, 20, 40],
        "ready_ratios": [0.25, 0.5, 0.75],
        "prediction_k": 6,
        "apply_bias": True,
        "primary_fetch": 20,
        "forward_chunk": 16384,
        "force_predict": False,
        "decode_dispersal": "van_der_corput",
        "prefill_order": "sequential_per_session",
    },
    "output": {"volume_dir": SIMULATION_DIR},
}


class DistStats(TypedDict):
    """Summary of one per-combo missing-expert distribution."""

    count: int
    mean_missing: float
    portion_fully_resident: float
    missing_p50: int
    missing_p90: int


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

    cfg = _merge(CONFIG_DEFAULTS, yaml.safe_load(config_text) or {})
    sim = cfg["simulation"]

    if not sim["chunk_sizes"] or any(int(b) <= 0 for b in sim["chunk_sizes"]):
        raise ValueError("simulation.chunk_sizes must be a non-empty list of positives")
    if not sim["decode_portions"] or any(
        not 0.0 < float(f) < 1.0 for f in sim["decode_portions"]
    ):
        raise ValueError("simulation.decode_portions must be fractions in (0, 1)")
    if any(int(n) < 0 for n in sim["fetch_counts"]):
        raise ValueError("simulation.fetch_counts must be non-negative")
    if not sim["ready_ratios"] or any(
        not 0.0 < float(r) < 1.0 for r in sim["ready_ratios"]
    ):
        raise ValueError("simulation.ready_ratios must be fractions in (0, 1)")
    if int(sim["prediction_k"]) < 1:
        raise ValueError("simulation.prediction_k must be >= 1")
    if int(sim["primary_fetch"]) < 0:
        raise ValueError("simulation.primary_fetch must be non-negative")
    if sim["decode_dispersal"] not in DISPERSALS:
        raise ValueError(
            f"simulation.decode_dispersal must be one of {DISPERSALS}"
        )
    if sim["prefill_order"] not in PREFILL_ORDERS:
        raise ValueError(f"simulation.prefill_order must be one of {PREFILL_ORDERS}")
    if not cfg["data"]["split"]:
        raise ValueError("data.split must name at least one split")
    return cfg


# --------------------------------------------------------------------------- #
# Manifests and dimensions
# --------------------------------------------------------------------------- #
def load_harvest_manifest(harvest_dir: str = HARVEST_DIR) -> dict[str, Any]:
    """Load the harvest ``manifest.json`` (slice table + static router tables)."""
    path = os.path.join(harvest_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no harvest manifest at {path}; run 3_harvest")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_cache_manifest(cache_dir: str) -> dict[str, Any] | None:
    """Load the contiguous cache manifest, or ``None`` when it is not built."""
    path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def cache_dims(cache_manifest: dict[str, Any]) -> dict[str, int]:
    """The geometry recorded in the cache manifest (n_layers, dim, n_experts...)."""
    return dict(cache_manifest["dims"])


def paired_slices(
    cache_manifest: dict[str, Any],
    harvest_manifest: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    """Pair each cache slice with its harvest record so its labels can be read.

    Both are laid out in harvest-manifest order, so the pairing is positional and
    verified: a mismatch means the cache is stale relative to the harvest.
    """
    records = [r for r in harvest_manifest["records"] if r["split"] == split]
    slices = cache_manifest["splits"][split]["slices"]
    if len(records) != len(slices):
        raise ValueError(
            f"{split}: cache has {len(slices)} slices but harvest has "
            f"{len(records)}; rebuild the cache"
        )
    paired: list[dict[str, Any]] = []
    for record, entry in zip(records, slices):
        if (
            record["dataset_id"] != entry["dataset_id"]
            or record["document_id"] != entry["document_id"]
        ):
            raise ValueError(
                f"{split}: cache/harvest slice mismatch "
                f"({entry['dataset_id']}/{entry['document_id']} vs "
                f"{record['dataset_id']}/{record['document_id']}); rebuild the cache"
            )
        paired.append(
            {
                "dataset_id": record["dataset_id"],
                "document_id": record["document_id"],
                "path": record["path"],
                "offset": int(entry["offset"]),
                "n_tokens": int(entry["n_tokens"]),
            }
        )
    return paired


# --------------------------------------------------------------------------- #
# Cache entry access
# --------------------------------------------------------------------------- #
def _entry_key(
    kind: str, input_kind: str | None, split: str | None, layer: int
) -> str:
    if kind in ("bias", "expert_counts"):
        return f"{kind}:{layer}"
    if kind == "activation":
        return f"activation:{input_kind}:{split}:{layer}"
    return f"{kind}:{split}:{layer}"


def _entry(
    cache_manifest: dict[str, Any], key: str, cache_dir: str
) -> str:
    entry = cache_manifest["entries"].get(key)
    if entry is None:
        raise KeyError(f"cache entry {key!r} missing; run 6_train_all populate_cache")
    return os.path.join(cache_dir, entry["path"])


def read_cached_bias(cache_dir: str, cache_manifest: dict[str, Any], layer: int) -> Any:
    """The cached per-layer expert-selection bias ``[n_experts]`` fp32 as numpy."""
    from safetensors.numpy import load_file

    path = _entry(cache_manifest, _entry_key("bias", None, None, layer), cache_dir)
    return load_file(path)["bias"]


def read_cached_expert_counts(
    cache_dir: str, cache_manifest: dict[str, Any], layer: int
) -> Any:
    """The cached target-layer expert usage counts ``[n_experts]`` int64 as numpy."""
    from safetensors.numpy import load_file

    path = _entry(
        cache_manifest, _entry_key("expert_counts", None, None, layer), cache_dir
    )
    return load_file(path)["counts"]


def read_cached_topk(
    cache_dir: str, cache_manifest: dict[str, Any], split: str, layer: int
) -> Any:
    """The cached target-layer truth top-k ids ``[n_tokens, top_k]`` uint8 as numpy."""
    from safetensors.numpy import load_file

    path = _entry(cache_manifest, _entry_key("topk", None, split, layer), cache_dir)
    return load_file(path)["data"]


def read_cached_activation(
    cache_dir: str,
    cache_manifest: dict[str, Any],
    split: str,
    input_kind: str,
    source_layer: int,
) -> tuple[Any, Any]:
    """The cached source-layer activation: fp8 ``[n_tokens, d]`` and per-slice scale."""
    from safetensors.torch import load_file

    path = _entry(
        cache_manifest,
        _entry_key("activation", input_kind, split, source_layer),
        cache_dir,
    )
    blob = load_file(path)
    return blob["data"], blob["scale"]


def read_is_decode(path: str) -> Any:
    """Read one harvest slice's ``is_decode`` label ``[n_tokens]`` as bool."""
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        return handle.get_tensor("is_decode").astype(bool)


# --------------------------------------------------------------------------- #
# Prediction cache (written by the GPU predictor, read by the CPU sweep)
# --------------------------------------------------------------------------- #
PREDICTIONS_SUBDIR = "predictions"


def prediction_dir(volume_dir: str, training_run_id: str) -> str:
    """Directory holding one training run's cached per-layer predictions."""
    return os.path.join(volume_dir, PREDICTIONS_SUBDIR, training_run_id)


def prediction_fingerprint(
    training_run_id: str,
    cache_manifest: dict[str, Any],
    prediction_k: int,
    apply_bias: bool,
) -> str:
    """Digest of everything a cached prediction depends on.

    Predictions are valid only for this ``(run, cache, prediction_k, bias)``
    combination; a cache rebuild or a knob change therefore invalidates them.
    """
    import hashlib

    digest = hashlib.sha256()
    digest.update(
        "|".join(
            [
                training_run_id,
                str(cache_manifest.get("harvest_fingerprint", "")),
                str(int(prediction_k)),
                str(bool(apply_bias)),
            ]
        ).encode()
    )
    return digest.hexdigest()[:16]


def prediction_file(pred_dir: str, layer: int, split: str) -> str:
    """Path of one layer/split prediction file inside the cache."""
    return os.path.join(pred_dir, f"L{layer:02d}", f"{split}.safetensors")


def load_prediction_manifest(pred_dir: str) -> dict[str, Any] | None:
    """Load the prediction-cache manifest, or ``None`` when it is absent."""
    return _read_json(os.path.join(pred_dir, "manifest.json"))


def read_cached_prediction(pred_dir: str, layer: int, split: str) -> Any:
    """Read cached per-token predicted top-k ``[n_tokens, prediction_k]`` uint8."""
    from safetensors.numpy import load_file

    return load_file(prediction_file(pred_dir, layer, split))["predicted"]


def write_cached_prediction(
    pred_dir: str, layer: int, split: str, predicted: Any
) -> str:
    """Write one layer/split prediction file; return its path."""
    import numpy as np
    from safetensors.numpy import save_file

    path = prediction_file(pred_dir, layer, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({"predicted": np.ascontiguousarray(predicted, dtype=np.uint8)}, path)
    return path


# --------------------------------------------------------------------------- #
# Run / checkpoint resolution
# --------------------------------------------------------------------------- #
def _read_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_run_dir(training_dir: str, run_id: str | None) -> tuple[str, str]:
    """Resolve a run directory (newest when ``run_id`` is null).

    Returns ``(run_dir, run_id)``.
    """
    if run_id:
        run_dir = os.path.join(training_dir, run_id)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"no run {run_id!r} under {training_dir}")
        return run_dir, run_id
    if not os.path.isdir(training_dir):
        raise FileNotFoundError(f"no training volume at {training_dir}")
    candidates = [
        name
        for name in os.listdir(training_dir)
        if os.path.exists(os.path.join(training_dir, name, "summary.json"))
    ]
    if not candidates:
        raise FileNotFoundError(f"no completed runs under {training_dir}")
    newest = max(
        candidates, key=lambda name: os.path.getmtime(os.path.join(training_dir, name))
    )
    return os.path.join(training_dir, newest), newest


def load_run_summary(run_dir: str) -> dict[str, Any]:
    """Load a ``6_train_all`` run's aggregate ``summary.json``."""
    summary = _read_json(os.path.join(run_dir, "summary.json"))
    if summary is None:
        raise FileNotFoundError(f"no summary.json under {run_dir}")
    return summary


def _input_dim(input_kind: str, dims: dict[str, int]) -> int:
    if input_kind == "expanded":
        return int(dims["hc_mult"]) * int(dims["dim"])
    return int(dims["dim"])


def resolve_layers(
    cfg: dict[str, Any],
    run_dir: str,
    run_summary: dict[str, Any],
    dims: dict[str, int],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Resolve the checkpoints to simulate; return ``(metas, skipped_layers)``.

    A layer whose ``checkpoint.safetensors`` is absent is reported in
    ``skipped_layers`` rather than failing the run.
    """
    wanted = cfg["training"]["layers"]
    keep = {int(layer) for layer in wanted} if wanted else None
    run_cfg = run_summary["config"]
    model = run_cfg["model"]
    task = run_cfg["task"]

    metas: list[dict[str, Any]] = []
    skipped: list[int] = []
    for row in sorted(run_summary["layers"], key=lambda item: item["layer"]):
        layer = int(row["layer"])
        if keep is not None and layer not in keep:
            continue
        layer_dir = os.path.join(run_dir, f"L{layer:02d}")
        checkpoint = os.path.join(layer_dir, "checkpoint.safetensors")
        if not os.path.exists(checkpoint):
            skipped.append(layer)
            continue
        metrics = _read_json(os.path.join(layer_dir, "metrics.json")) or {}
        metas.append(
            {
                "layer": layer,
                "source_layer": int(metrics.get("source_layer", row["source_layer"])),
                "distance": int(metrics.get("distance", row["distance"])),
                "d_in": int(metrics.get("d_in", _input_dim(task["input"], dims))),
                "input_kind": task["input"],
                "arch": model["arch"],
                "rank": int(model["rank"]),
                "hidden": int(model["hidden"]),
                "checkpoint": checkpoint,
            }
        )
    if not metas:
        raise ValueError(f"no checkpoints resolved under {run_dir}")
    return metas, skipped


# --------------------------------------------------------------------------- #
# Resident hot sets and distribution statistics
# --------------------------------------------------------------------------- #
def resident_masks(
    counts: Any, ratios: tuple[float, ...]
) -> tuple[dict[float, Any], dict[float, int]]:
    """Hottest-k resident mask per ratio for one layer's expert counts.

    Mirrors ``4_analysis``: ``k = round(ratio * n_experts)``. Returns
    ``(masks, k_by_ratio)`` where ``masks[ratio]`` is a numpy bool ``[n_experts]``.
    """
    import numpy as np

    counts = np.asarray(counts, dtype=np.int64)
    n_experts = int(counts.shape[0])
    masks: dict[float, Any] = {}
    k_by_ratio: dict[float, int] = {}
    for ratio in ratios:
        k = round(float(ratio) * n_experts)
        top = np.argsort(counts)[::-1][:k]
        mask = np.zeros(n_experts, dtype=bool)
        mask[top] = True
        masks[float(ratio)] = mask
        k_by_ratio[float(ratio)] = k
    return masks, k_by_ratio


def _percentile(portion: Any, target: float) -> int:
    import numpy as np

    cdf = np.cumsum(portion)
    idx = int(np.searchsorted(cdf, target))
    return min(idx, portion.shape[0] - 1)


def distribution_stats(hist: Any) -> DistStats:
    """Summarise one per-combo missing-expert count histogram."""
    import numpy as np

    hist = np.asarray(hist, dtype=np.float64)
    total = int(hist.sum())
    if total <= 0:
        return {
            "count": 0,
            "mean_missing": 0.0,
            "portion_fully_resident": 0.0,
            "missing_p50": 0,
            "missing_p90": 0,
        }
    portion = hist / total
    missing = np.arange(hist.shape[0], dtype=np.float64)
    return {
        "count": total,
        "mean_missing": round(float((missing * portion).sum()), 6),
        "portion_fully_resident": round(float(portion[0]), 6),
        "missing_p50": _percentile(portion, 0.5),
        "missing_p90": _percentile(portion, 0.9),
    }


def utc_now() -> str:
    """Timestamp string used in every ``summary.json``."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
