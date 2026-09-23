"""Shared code for the expert-routing analysis studies.

This module holds everything two or more studies (``analyses/*.py``) need: the
harvest data-access layer, the residency prerequisite (``counts`` / ``masks``),
the shared statistics and figure/CSV plumbing, and the ``AnalysisContext`` that
every ``Analysis.run(ctx)`` reads and writes.

numpy and matplotlib are imported *inside* functions, never at module scope, so
``main.py`` stays importable on a machine without the scientific stack (the
local entrypoint imports this module locally).
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, TypedDict

if TYPE_CHECKING:
    import numpy as np

HARVEST_DIR = "/harvest"
ANALYSIS_DIR = "/harvest/analysis"
MANIFEST_PATH = os.path.join(HARVEST_DIR, "manifest.json")

RATIOS = (0.25, 0.5, 0.75)
SPLITS = ("train", "test")
DATASETS = ("yi30-think", "yi30-nothink", "terminus2", "dsh")

BATCH_SIZES = tuple(range(1, 21))
SESSION_SEED = 33377335
PREFILL_TOKEN_BUDGET = 200_000

COLORS = {0.25: "#4C72B0", 0.5: "#DD8452", 0.75: "#55A868"}


class Event(TypedDict, total=False):
    """One streamed event from a Modal generator: ``start`` / ``log`` /
    ``image`` / ``file``.

    ``text`` carries a log line, ``name`` + ``data`` carry a streamed artifact,
    and ``run_id`` is set only on the ``start`` event.
    """

    type: str
    run_id: str
    text: str
    name: str
    data: bytes


class HarvestRecord(TypedDict, total=False):
    """One slice entry from the harvest ``manifest.json``.

    Fields used here: ``path`` (safetensors file), ``dataset_id``,
    ``document_id``, ``split``, ``n_tokens``, ``n_layers``, ``top_k`` and
    ``n_routed_experts``.
    """

    path: str
    dataset_id: str
    document_id: str
    split: str
    n_tokens: int
    n_layers: int
    top_k: int
    n_routed_experts: int


class DistStats(TypedDict):
    """Summary of a length-``top_k+1`` portion curve."""

    portion_by_missing: list[float]
    portion_fully_resident: float
    portion_needing_fetch: float
    mean_missing: float


class PrefillMetaEntry(TypedDict):
    """Per-batch bookkeeping for the concurrent-prefill study."""

    n_events: int
    tokens_used: int
    mean_coverage: dict[float, float]


class GroupAccumulator(TypedDict):
    """Accumulator for one replay group (pooled or per-dataset).

    ``counts`` is ``[n_ratios, n_layers, top_k+1]`` and ``tokens`` is
    ``[n_ratios, n_layers]``.
    """

    counts: "np.ndarray"
    tokens: "np.ndarray"


# One portion curve per resident ratio, keyed by the ratio (0.25/0.5/0.75).
Curves = dict[float, "np.ndarray"]


@dataclass
class AnalysisContext:
    """Mutable state shared by the studies in the queue.

    Built by ``_prepare`` (records, dims, session pools) and progressively
    filled by the studies in order: ``expert_ranking`` sets ``counts`` /
    ``k_by_ratio`` / ``masks``; ``token_miss`` sets ``pooled`` / ``per_dataset``;
    ``decode_miss`` sets ``batches``; ``prefill_miss`` sets ``prefills`` /
    ``prefill_meta``. Later studies read what earlier ones wrote.
    """

    quick: bool
    run_id: str
    remote_out_dir: str

    logs: list[str] = field(default_factory=list)

    records: list[HarvestRecord] = field(default_factory=list)
    train: list[HarvestRecord] = field(default_factory=list)
    test: list[HarvestRecord] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)

    n_layers: int = 0
    top_k: int = 0
    n_experts: int = 0
    n_train: int = 0
    n_test: int = 0

    by_dataset: dict[str, list[HarvestRecord]] = field(default_factory=dict)

    counts: "np.ndarray | None" = None
    k_by_ratio: dict[float, int] = field(default_factory=dict)
    masks: dict[float, "np.ndarray"] = field(default_factory=dict)

    pooled: dict[str, Curves] = field(default_factory=dict)
    per_dataset: dict[str, dict[str, Curves]] = field(default_factory=dict)
    n_tokens_by_split: dict[str, int] = field(default_factory=dict)

    batches: dict[int, Curves] = field(default_factory=dict)
    prefills: dict[int, Curves] = field(default_factory=dict)
    prefill_meta: dict[int, PrefillMetaEntry] = field(default_factory=dict)

    def out_dir(self, name: str) -> str:
        """Return (creating it) this study's remote output subdirectory."""
        path = os.path.join(self.remote_out_dir, name)
        os.makedirs(path, exist_ok=True)
        return path


# --------------------------------------------------------------------------- #
# Event helpers
# --------------------------------------------------------------------------- #
def _emit(logs: list[str], message: str) -> Event:
    """Record ``message`` in ``logs`` and return it as a streamed log event."""
    logs.append(message)
    return {"type": "log", "text": message}


def _file_event(path: str, name: str) -> Event:
    """Read a written artifact back and return it as a streamed file event."""
    with open(path, "rb") as handle:
        return {"type": "file", "name": name, "data": handle.read()}


def _image_event(name: str, data: bytes) -> Event:
    """Wrap raw PNG bytes as a streamed image event."""
    return {"type": "image", "name": name, "data": data}


def _utc_now() -> str:
    """Timestamp string used in every ``summary.json``."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- #
# Artifact writers
# --------------------------------------------------------------------------- #
def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _write_run_log(out_dir: str, logs: list[str]) -> str:
    """Write this study's ``run.log`` and return its path."""
    path = os.path.join(out_dir, "run.log")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(logs) + "\n")
    return path


def _save_curve_csv(path: str, curves: dict[int, Curves]) -> None:
    """Write a ``batch,ratio,missing,portion`` CSV (decode + prefill share this)."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("batch,ratio,missing,portion\n")
        for batch, by_ratio in curves.items():
            for ratio in RATIOS:
                for value, portion in enumerate(by_ratio[ratio]):
                    handle.write(f"{batch},{ratio},{value},{portion:.8f}\n")


# --------------------------------------------------------------------------- #
# Harvest data access
# --------------------------------------------------------------------------- #
def _read_topk_ids(path: str) -> "np.ndarray":
    """Load one slice's ``topk_ids`` as ``[n_tokens, n_layers, top_k]``."""
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        return handle.get_tensor("topk_ids")


def _decode_ids(path: str) -> "np.ndarray":
    """Load one slice's decode-token routing ids, in step order.

    Decode tokens are the rows with ``is_decode == 1``; they are what a serving
    session emits autoregressively, so they are what concurrent decoding routes.
    """
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        ids = handle.get_tensor("topk_ids")
        is_decode = handle.get_tensor("is_decode")
    return ids[is_decode.astype(bool)]


def _prefill_phases(path: str) -> list["np.ndarray"]:
    """Load a slice's per-turn prefill phases, in turn order.

    Prefill is every non-assistant token (system + tools template, user turns,
    tool results, scaffolding), arriving as contiguous per-turn phases. Returns
    one ``[n_phase_tokens, n_layers, top_k]`` array per phase, ordered by
    ``phase_index``.
    """
    import numpy as np
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        ids = handle.get_tensor("topk_ids")
        is_decode = handle.get_tensor("is_decode").astype(bool)
        phase_index = handle.get_tensor("phase_index")
    prefill = ~is_decode
    return [
        ids[prefill & (phase_index == phase)]
        for phase in np.unique(phase_index[prefill])
    ]


# --------------------------------------------------------------------------- #
# Statistics + plotting plumbing
# --------------------------------------------------------------------------- #
def _dist_stats(distribution: Any) -> DistStats:
    """Summarise a length-``top_k+1`` portion curve."""
    import numpy as np

    distribution = np.asarray(distribution, dtype=np.float64)
    missing = np.arange(distribution.shape[0])
    return {
        "portion_by_missing": [round(float(v), 6) for v in distribution],
        "portion_fully_resident": round(float(distribution[0]), 6),
        "portion_needing_fetch": round(float(1.0 - distribution[0]), 6),
        "mean_missing": round(float((missing * distribution).sum()), 6),
    }


def _pyplot():
    """Return ``matplotlib.pyplot`` with the headless Agg backend selected."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _figure_bytes(fig: Any) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it."""
    import io

    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _batch_grid_figure(
    panels: list[tuple[str, Curves, dict[float, Any], Any]],
    ylabel: str,
    suptitle: str,
    markersize: float = 3.0,
) -> bytes:
    """Render a 4x5 grid of per-batch curves (shared by decode + prefill).

    Args:
        panels: One ``(title, curves, x_by_ratio, xlim)`` per batch. Decode uses
            the same x for every ratio; prefill's curve lengths differ per ratio
            (x capped at ``n_experts - k``), hence ``x_by_ratio`` maps each ratio
            to its own x values. ``xlim`` may be ``None`` to auto-scale.
        ylabel: Label for the first subplot's y axis.
        suptitle: Figure title.
        markersize: Marker size for every curve.

    Returns:
        PNG bytes.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n_cols = 5
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, grid = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    grid = np.atleast_1d(grid).ravel()
    for ax, (title, curves, x_by_ratio, xlim) in zip(grid, panels):
        for ratio in RATIOS:
            ax.plot(
                x_by_ratio[ratio],
                curves[ratio],
                "-o",
                markersize=markersize,
                linewidth=1.2,
                label=f"keep {int(ratio * 100)}%",
                color=COLORS[ratio],
            )
        ax.set_title(title)
        ax.set_xlabel("unique missing experts per layer")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(alpha=0.3)
    for ax in grid[len(panels):]:
        ax.axis("off")
    grid[0].set_ylabel(ylabel)
    grid[0].legend(fontsize=7)
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _figure_bytes(fig)


# --------------------------------------------------------------------------- #
# Residency prerequisite (shared by token_miss / decode_miss / prefill_miss)
# --------------------------------------------------------------------------- #
def _expert_counts(
    train: list[HarvestRecord], n_layers: int, n_experts: int
) -> "np.ndarray":
    """Count expert usage over every training token, per layer, per expert.

    Returns an ``int64`` array ``[n_layers, n_experts]``.
    """
    import numpy as np

    counts = np.zeros((n_layers, n_experts), dtype=np.int64)
    for record in train:
        ids = _read_topk_ids(record["path"])
        for layer in range(n_layers):
            counts[layer] += np.bincount(
                ids[:, layer, :].astype(np.int64).ravel(), minlength=n_experts
            )
    return counts


def _resident_masks(
    counts: "np.ndarray", n_experts: int
) -> tuple[dict[float, int], dict[float, "np.ndarray"]]:
    """Derive the hottest-k resident set per layer for every ratio.

    Returns ``(k_by_ratio, masks)`` where ``masks[ratio]`` is a
    ``[n_layers, n_experts]`` boolean mask of the resident experts.
    """
    import numpy as np

    k_by_ratio = {ratio: int(round(ratio * n_experts)) for ratio in RATIOS}
    masks: dict[float, "np.ndarray"] = {}
    for ratio in RATIOS:
        k = k_by_ratio[ratio]
        mask = np.zeros(counts.shape, dtype=bool)
        for layer in range(counts.shape[0]):
            top = np.argpartition(counts[layer], -k)[-k:]
            mask[layer, top] = True
        masks[ratio] = mask
    return k_by_ratio, masks


# --------------------------------------------------------------------------- #
# Orchestration (called by ``main.analyze``)
# --------------------------------------------------------------------------- #
def _dataset_pools(
    train: list[HarvestRecord], datasets: list[str], seed: int
) -> dict[str, list[HarvestRecord]]:
    """Seeded-shuffle the training sessions of each dataset for round-robin use."""
    rng = random.Random(seed)
    by_dataset: dict[str, list[HarvestRecord]] = {}
    for dataset in datasets:
        pool = [r for r in train if r["dataset_id"] == dataset]
        rng.shuffle(pool)
        by_dataset[dataset] = pool
    return by_dataset


def _prepare(ctx: AnalysisContext) -> Iterator[Event]:
    """Load the harvest manifest and fill ``ctx`` with records, dims and pools.

    Yields a ``log`` event for the slice census (or a terminal message when the
    manifest/records are missing). Leaves ``ctx.records`` empty on failure, which
    ``main.analyze`` uses to stop early.
    """
    if not os.path.exists(MANIFEST_PATH):
        yield _emit(ctx.logs, f"no manifest at {MANIFEST_PATH}; run 3_harvest first")
        return

    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = [r for r in manifest["records"] if r["split"] in SPLITS]
    if ctx.quick:
        records = [r for r in records if r["dataset_id"] == DATASETS[0]]
    if not records:
        yield _emit(ctx.logs, "no harvest records to analyze")
        return

    ctx.records = records
    ctx.n_layers = int(records[0]["n_layers"])
    ctx.top_k = int(records[0]["top_k"])
    ctx.n_experts = int(records[0]["n_routed_experts"])
    ctx.train = [r for r in records if r["split"] == "train"]
    ctx.test = [r for r in records if r["split"] == "test"]
    ctx.n_train = sum(r["n_tokens"] for r in ctx.train)
    ctx.n_test = sum(r["n_tokens"] for r in ctx.test)
    ctx.datasets = [d for d in DATASETS if any(r["dataset_id"] == d for r in records)]
    ctx.n_tokens_by_split = {"train": ctx.n_train, "test": ctx.n_test}
    ctx.by_dataset = _dataset_pools(ctx.train, ctx.datasets, SESSION_SEED)

    yield _emit(
        ctx.logs,
        f"{len(ctx.train)} train slices ({ctx.n_train} tokens), "
        f"{len(ctx.test)} test slices ({ctx.n_test} tokens); "
        f"layers={ctx.n_layers}, top_k={ctx.top_k}, experts={ctx.n_experts}",
    )


def _finalize(ctx: AnalysisContext) -> Iterator[Event]:
    """Write the harness-only ``run.log`` at the run root and stream it back."""
    os.makedirs(ctx.remote_out_dir, exist_ok=True)
    path = _write_run_log(ctx.remote_out_dir, ctx.logs)
    yield _file_event(path, "run.log")
