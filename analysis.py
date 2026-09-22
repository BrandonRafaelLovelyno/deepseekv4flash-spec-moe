"""Measure the speculative-expert-loading miss rate on the harvested routing data.

Reads the per-token expert routing captured by ``harvest.py`` (``topk_ids``,
shape ``[n_tokens, n_layers, top_k]``) and answers one question: if we keep only
the hottest X% of each layer's experts resident in VRAM, how many of a token's
``top_k`` experts are *missing* (and therefore must be fetched)?

Method
    1. Rank experts by **training** frequency, independently within each of the
       43 layers, and derive resident sets for 25% / 50% / 75% of the 256 experts
       per layer (64 / 128 / 192).
    2. Replay every **train** and **test** token. For each token and layer,
       ``missing = top_k - (# of its experts in the resident set)``, in 0..6.
    3. Normalize per layer over that layer's tokens, then average the 43 layer
       curves. The y axis is therefore the **portion of tokens**, not a count,
       and each curve sums to 1.

Test slices are scored against the train-derived resident sets only (no leakage).

Artifacts, under ``/harvest/analysis/<run_id>/`` on the volume and mirrored to
``analysis/output/<run_id>/`` locally:
    * ``run.log``                    -- everything printed
    * ``summary.json``               -- per split/dataset/ratio stats
    * ``distributions.csv``          -- split,dataset,ratio,missing,portion
    * ``train_expert_counts.npy``    -- [n_layers, n_experts] training counts
    * ``hot_experts.csv``            -- layer,ratio,rank,expert_id,count,share
    * ``01_missing_distribution.png``-- pooled train vs test, one bar per ratio
    * ``02_per_dataset.png``         -- 2x4 small multiples
    * ``03_cumulative_missing.png``  -- portion needing at least m fetches

The function is a Modal **generator**: it yields a ``start`` event (the run id),
``log`` events, and each PNG as raw ``image`` bytes, which the local entrypoint
writes to disk. CPU only -- no GPU, no torch.

Usage:
    modal run analysis.py            # full run over the 4 harvested datasets
    modal run analysis.py --quick    # first dataset only, for a smoke test
"""

import json
import os
import time

import modal

APP_NAME = "deepseek-v4-flash-analysis"

HARVEST_DIR = "/harvest"
ANALYSIS_DIR = "/harvest/analysis"
MANIFEST_PATH = os.path.join(HARVEST_DIR, "manifest.json")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "analysis", "output")

RATIOS = (0.25, 0.5, 0.75)
SPLITS = ("train", "test")
DATASETS = ("yi30-think", "yi30-nothink", "terminus2", "dsh")

COLORS = {0.25: "#4C72B0", 0.5: "#DD8452", 0.75: "#55A868"}

harvest_vol = modal.Volume.from_name(
    "deepseek-v4-flash-harvest", create_if_missing=True
)

# Only routing ids are needed, so numpy + safetensors + matplotlib is enough --
# the CUDA / torch stack the harvest image carries would be dead weight.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "safetensors", "matplotlib")
    .env({"MPLBACKEND": "Agg"})
)

app = modal.App(APP_NAME, image=image)


def _read_topk_ids(path: str):
    """Load one slice's ``topk_ids`` as a ``[n_tokens, n_layers, top_k]`` array."""
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        return handle.get_tensor("topk_ids")


def _figure_bytes(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it."""
    import io

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buffer.getvalue()


def _dist_stats(distribution) -> dict:
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


@app.function(
    volumes={HARVEST_DIR: harvest_vol},
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
)
def analyze(quick: bool = False):
    """Rank train experts, replay train/test tokens, stream figures back."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    yield {"type": "start", "run_id": run_id}

    logs = []

    def log(message: str):
        """Record a message and emit it as a streamed log event (once)."""
        logs.append(message)
        yield {"type": "log", "text": message}

    if not os.path.exists(MANIFEST_PATH):
        yield from log(f"no manifest at {MANIFEST_PATH}; run harvest.py first")
        return

    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = [r for r in manifest["records"] if r["split"] in SPLITS]
    if quick:
        records = [r for r in records if r["dataset_id"] == DATASETS[0]]
    if not records:
        yield from log("no harvest records to analyze")
        return

    n_layers = int(records[0]["n_layers"])
    top_k = int(records[0]["top_k"])
    n_experts = int(records[0]["n_routed_experts"])
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    n_train = sum(r["n_tokens"] for r in train)
    n_test = sum(r["n_tokens"] for r in test)
    datasets = [d for d in DATASETS if any(r["dataset_id"] == d for r in records)]
    yield from log(
        f"{len(train)} train slices ({n_train} tokens), "
        f"{len(test)} test slices ({n_test} tokens); "
        f"layers={n_layers}, top_k={top_k}, experts={n_experts}"
    )

    # ---- Pass A: training expert frequency, independently per layer. --------
    counts = np.zeros((n_layers, n_experts), dtype=np.int64)
    for record in train:
        ids = _read_topk_ids(record["path"])
        for layer in range(n_layers):
            counts[layer] += np.bincount(
                ids[:, layer, :].astype(np.int64).ravel(), minlength=n_experts
            )
    yield from log(
        f"ranked experts over {len(train)} train slices; "
        f"hottest expert used {int(counts.max())} times"
    )

    k_by_ratio = {ratio: int(round(ratio * n_experts)) for ratio in RATIOS}
    masks = {}
    for ratio in RATIOS:
        k = k_by_ratio[ratio]
        mask = np.zeros((n_layers, n_experts), dtype=bool)
        for layer in range(n_layers):
            top = np.argpartition(counts[layer], -k)[-k:]
            mask[layer, top] = True
        masks[ratio] = mask
        yield from log(f"keep {int(ratio * 100)}%: top {k} experts/layer resident")

    layer_index = np.arange(n_layers)[None, :, None]

    # ---- Pass B: replay every token, tally missing experts per layer. ------
    # group key -> (split, dataset or None pooled); accumulators are
    # [n_ratios, n_layers, top_k + 1] counts and [n_ratios, n_layers] tokens.
    shape = (len(RATIOS), n_layers, top_k + 1)
    groups = {}

    def accumulator():
        return {"counts": np.zeros(shape, dtype=np.int64), "tokens": np.zeros((len(RATIOS), n_layers))}

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
            yield from log(
                f"replayed {split}/{record['dataset_id']}/{record['document_id']} "
                f"({n_tokens} tokens)"
            )

    def curve(key):
        """Average the per-layer fraction curves into one portion curve per ratio."""
        acc = groups[key]
        out = {}
        for ri, ratio in enumerate(RATIOS):
            tokens = np.maximum(acc["tokens"][ri], 1)[:, None]
            per_layer = acc["counts"][ri] / tokens  # [L, K+1], rows sum to 1
            out[ratio] = per_layer.mean(axis=0)  # [K+1], sums to 1
        return out

    pooled = {split: curve((split, None)) for split in SPLITS}
    per_dataset = {
        split: {ds: curve((split, ds)) for ds in datasets if (split, ds) in groups}
        for split in SPLITS
    }
    n_tokens_by_split = {"train": n_train, "test": n_test}

    for split in SPLITS:
        for ratio in RATIOS:
            stats = _dist_stats(pooled[split][ratio])
            yield from log(
                f"{split:5} keep {int(ratio * 100):>2}%: "
                f"fully resident {stats['portion_fully_resident']:.4f}, "
                f"needs fetch {stats['portion_needing_fetch']:.4f}, "
                f"mean missing {stats['mean_missing']:.4f}"
            )

    # ---- Persist arrays + tabular summaries -------------------------------
    out_dir = os.path.join(ANALYSIS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train_expert_counts.npy"), counts)

    with open(os.path.join(out_dir, "distributions.csv"), "w", encoding="utf-8") as handle:
        handle.write("split,dataset,ratio,missing,portion\n")
        for split in SPLITS:
            for ratio in RATIOS:
                for value, portion in enumerate(pooled[split][ratio]):
                    handle.write(
                        f"{split},pooled,{ratio},{value},{portion:.8f}\n"
                    )
            for ds, curves in per_dataset[split].items():
                for ratio in RATIOS:
                    for value, portion in enumerate(curves[ratio]):
                        handle.write(f"{split},{ds},{ratio},{value},{portion:.8f}\n")

    with open(os.path.join(out_dir, "hot_experts.csv"), "w", encoding="utf-8") as handle:
        handle.write("layer,ratio,rank,expert_id,train_count,share\n")
        total_per_layer = counts.sum(axis=1, keepdims=True)
        for ratio in RATIOS:
            k = k_by_ratio[ratio]
            for layer in range(n_layers):
                order = np.argsort(counts[layer])[::-1][:k]
                for rank, expert in enumerate(order):
                    share = counts[layer, expert] / max(total_per_layer[layer, 0], 1)
                    handle.write(
                        f"{layer},{ratio},{rank},{expert},"
                        f"{int(counts[layer, expert])},{share:.8f}\n"
                    )

    summary = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick": quick,
        "ratios": list(RATIOS),
        "k_by_ratio": {str(r): k_by_ratio[r] for r in RATIOS},
        "n_layers": n_layers,
        "top_k": top_k,
        "n_experts": n_experts,
        "datasets": datasets,
        "splits": {
            split: {
                "n_tokens": n_tokens_by_split[split],
                "pooled": {str(r): _dist_stats(pooled[split][r]) for r in RATIOS},
                "per_dataset": {
                    ds: {str(r): _dist_stats(curves[r]) for r in RATIOS}
                    for ds, curves in per_dataset[split].items()
                },
            }
            for split in SPLITS
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # ---- Figures -----------------------------------------------------------
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
    yield {
        "type": "image",
        "name": "01_missing_distribution.png",
        "data": _figure_bytes(fig),
    }

    if datasets:
        fig2, axes2 = plt.subplots(2, len(datasets), figsize=(4 * len(datasets), 8), squeeze=False)
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
        yield {
            "type": "image",
            "name": "02_per_dataset.png",
            "data": _figure_bytes(fig2),
        }

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
    yield {
        "type": "image",
        "name": "03_cumulative_missing.png",
        "data": _figure_bytes(fig3),
    }

    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(logs) + "\n")
    for name in (
        "summary.json",
        "distributions.csv",
        "hot_experts.csv",
        "train_expert_counts.npy",
    ):
        with open(os.path.join(out_dir, name), "rb") as handle:
            yield {"type": "file", "name": name, "data": handle.read()}
    harvest_vol.commit()
    yield from log(f"artifacts written to {out_dir}")


@app.local_entrypoint()
def main(quick: bool = False) -> None:
    run_id = None
    out_dir = None
    log_lines = []
    for event in analyze.remote_gen(quick=quick):
        kind = event["type"]
        if kind == "start":
            run_id = event["run_id"]
            out_dir = os.path.join(OUT_DIR, run_id)
            os.makedirs(out_dir, exist_ok=True)
            print(f"run_id: {run_id}")
        elif kind == "log":
            print(event["text"])
            log_lines.append(event["text"])
        elif kind in ("image", "file"):
            path = os.path.join(out_dir, event["name"])
            with open(path, "wb") as handle:
                handle.write(event["data"])
            print(f"wrote {path} ({len(event['data']) / 1024:.1f} KiB)")
    if out_dir is not None:
        with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(log_lines) + "\n")
        print(f"\nartifacts mirrored to {out_dir}")
