"""Config, cache/harvest access and shared statistics.

This stage measures, for each prefill chunk size and each MoE layer, the
average number of **distinct routed experts** the chunk's tokens activate
together (the union of every token's top-k). It reads only the ground-truth top-k
and the ``is_decode`` labels that ``7_final_simulation`` already replays -- no
predictor, no training run and no GPU. This module owns everything two or more
modules need:

* the config (``load_config``) and the layer list, resolved straight from the
  cache's ``topk:<split>:<layer>`` entries (``resolve_layers``);
* reading the contiguous cache (truth top-k) and the harvest ``is_decode``
  labels the chunked-prefill streams are built from;
* the tidy per-combo distinct-expert statistics every artifact writer shares.

torch / numpy / yaml are imported *inside* functions, never at module scope, so
``main.py`` stays importable on a machine without the scientific stack.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, TypedDict

CACHE_DIR = "/cache"
HARVEST_DIR = "/harvest"
ARTICLE_DIR = "/article"

DISPERSALS = ("van_der_corput",)
PREFILL_ORDERS = ("sequential_per_session",)

CONFIG_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "cache": {"dir": CACHE_DIR},
    "harvest": {"dir": HARVEST_DIR},
    "data": {"split": ["train", "test"], "layers": None},
    "simulation": {
        "chunk_sizes": [16, 32, 64, 128, 256, 512, 1024],
        "decode_portions": [0.75, 0.5, 0.125],
        "decode_dispersal": "van_der_corput",
        "prefill_order": "sequential_per_session",
        "primary_portion": 0.5,
    },
    "output": {"volume_dir": ARTICLE_DIR},
}


class DistinctStats(TypedDict):
    """Summary of one per-combo distinct-expert-per-chunk distribution."""

    count: int
    mean_distinct: float
    distinct_fraction: float
    distinct_min: int
    distinct_p50: int
    distinct_p90: int
    distinct_max: int


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
    if sim["decode_dispersal"] not in DISPERSALS:
        raise ValueError(f"simulation.decode_dispersal must be one of {DISPERSALS}")
    if sim["prefill_order"] not in PREFILL_ORDERS:
        raise ValueError(f"simulation.prefill_order must be one of {PREFILL_ORDERS}")
    if not cfg["data"]["split"]:
        raise ValueError("data.split must name at least one split")
    return cfg


def primary_portion(cfg: dict[str, Any]) -> float:
    """The configured heatmap decode portion, snapped to a configured value."""
    portions = sorted(float(f) for f in cfg["simulation"]["decode_portions"])
    target = float(cfg["simulation"].get("primary_portion", portions[-1]))
    return min(portions, key=lambda portion: abs(portion - target))


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
def _entry_key(kind: str, split: str | None, layer: int) -> str:
    return f"{kind}:{split}:{layer}"


def _entry(cache_manifest: dict[str, Any], key: str, cache_dir: str) -> str:
    entry = cache_manifest["entries"].get(key)
    if entry is None:
        raise KeyError(f"cache entry {key!r} missing; run 6_train_all populate_cache")
    return os.path.join(cache_dir, entry["path"])


def read_cached_topk(
    cache_dir: str, cache_manifest: dict[str, Any], split: str, layer: int
) -> Any:
    """The cached target-layer truth top-k ids ``[n_tokens, top_k]`` uint8 as numpy."""
    from safetensors.numpy import load_file

    path = _entry(cache_manifest, _entry_key("topk", split, layer), cache_dir)
    return load_file(path)["data"]


def read_is_decode(path: str) -> Any:
    """Read one harvest slice's ``is_decode`` label ``[n_tokens]`` as bool."""
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        return handle.get_tensor("is_decode").astype(bool)


# --------------------------------------------------------------------------- #
# Layer resolution
# --------------------------------------------------------------------------- #
def resolve_layers(
    cache_manifest: dict[str, Any],
    splits: list[str],
    wanted: list[int] | None,
) -> tuple[list[int], list[int]]:
    """Resolve the MoE layers to measure; return ``(layers, missing)``.

    A layer is measurable only when the cache holds its truth top-k for *every*
    split being replayed. ``wanted`` (when set) narrows the selection and any
    requested layer without cache coverage is reported in ``missing``.
    """
    available: set[int] | None = None
    for split in splits:
        prefix = f"topk:{split}:"
        layers = {
            int(key[len(prefix) :])
            for key in cache_manifest.get("entries", {})
            if key.startswith(prefix)
        }
        available = layers if available is None else (available & layers)

    available = available or set()
    if not available:
        raise ValueError(
            f"no cached truth top-k for splits {splits}; rebuild the cache"
        )

    keep = {int(layer) for layer in wanted} if wanted else None
    selected = sorted(layer for layer in available if keep is None or layer in keep)
    missing = sorted(keep - available) if keep else []
    if not selected:
        raise ValueError(f"no layers resolved from {sorted(available)} (wanted {wanted})")
    return selected, missing


# --------------------------------------------------------------------------- #
# Distribution statistics
# --------------------------------------------------------------------------- #
def _percentile(portion: Any, target: float) -> int:
    import numpy as np

    cdf = np.cumsum(portion)
    idx = int(np.searchsorted(cdf, target))
    return min(idx, portion.shape[0] - 1)


def distinct_stats(hist: Any) -> DistinctStats:
    """Summarise one per-combo distinct-expert histogram (index = distinct count)."""
    import numpy as np

    hist = np.asarray(hist, dtype=np.float64)
    total = int(hist.sum())
    if total <= 0:
        return {
            "count": 0,
            "mean_distinct": 0.0,
            "distinct_fraction": 0.0,
            "distinct_min": 0,
            "distinct_p50": 0,
            "distinct_p90": 0,
            "distinct_max": 0,
        }
    portion = hist / total
    values = np.arange(hist.shape[0], dtype=np.float64)
    support = np.flatnonzero(hist)
    return {
        "count": total,
        "mean_distinct": round(float((values * portion).sum()), 6),
        "distinct_fraction": round(
            float((values * portion).sum()) / max(hist.shape[0] - 1, 1), 6
        ),
        "distinct_min": int(support[0]),
        "distinct_p50": _percentile(portion, 0.5),
        "distinct_p90": _percentile(portion, 0.9),
        "distinct_max": int(support[-1]),
    }


def utc_now() -> str:
    """Timestamp string used in every ``summary.json``."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
