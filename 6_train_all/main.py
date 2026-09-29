"""Train one expert-routing predictor for every layer, with per-epoch eval.

The train-all counterpart of ``5_train_one``:

* ``build_cache`` (CPU only) re-lays *every* layer the run needs into contiguous
  per-layer files once, so the expensive strided slice reading stays off the GPU.
* ``train_layer`` (small GPU) trains and evaluates exactly one layer, writing its
  ``L{dd}/`` artifacts and committing them.
* ``train_all`` (CPU only) fans one ``train_layer`` container out per layer via
  ``starmap`` and writes the run-level aggregate -- so the GPU is requested only
  where computation happens, never for cache building or orchestration.
* the local entrypoint runs the cache build first (a no-op when fresh), so the
  GPU containers are not created until the data is ready.

A predictor for target layer ``L`` reads the activation of ``source = L -
min(L, distance)`` (look back at most ``distance`` layers, clamped at layer 0)
and predicts ``L``'s score vector.

All knobs live in ``6_train_all/config.yaml`` (copy ``config.example.yaml``).

Artifacts land under ``<output.volume_dir>/<run_id>/`` on the
``deepseek-v4-flash-training`` volume -- one ``L{dd}/`` subdirectory per layer
plus the run-level ``summary.json`` / ``layers.csv`` / ``recall_by_layer.png`` --
and the run-level files are mirrored to ``6_train_all/output/<run_id>/``.

Usage:
    modal run 6_train_all/main.py
    modal run 6_train_all/main.py --build-only
    modal run 6_train_all/main.py --layers 20,21
    modal run 6_train_all/main.py --config 6_train_all/other.yaml
"""

import os
import sys
import time

import modal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from helper import (
    CACHE_DIR,
    HARVEST_DIR,
    TRAINING_DIR,
    load_cache_manifest,
    load_cached_bias,
    load_config,
    populate_cache,
    resolve_tasks,
)
from reporting import (
    mirror_run,
    print_cache_summary,
    print_run_report,
    row_from_summary,
    write_aggregate,
    write_layer_artifacts,
)

APP_NAME = "deepseek-v4-flash-train-all"

OUT_DIR = os.path.join(THIS_DIR, "output")

harvest_vol = modal.Volume.from_name(
    "deepseek-v4-flash-harvest", create_if_missing=True
)
cache_vol = modal.Volume.from_name(
    "deepseek-v4-flash-train-cache", create_if_missing=True
)
training_vol = modal.Volume.from_name(
    "deepseek-v4-flash-training", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "safetensors", "numpy", "matplotlib", "pyyaml")
    .env({"MPLBACKEND": "Agg"})
    .add_local_python_source("helper", "models", "reporting", "plots", "training")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={HARVEST_DIR: harvest_vol, CACHE_DIR: cache_vol},
    cpu=8,
    memory=48 * 1024,
    timeout=2 * 60 * 60,
)
def build_cache(config_text: str) -> dict:
    """CPU-only: re-lay every layer the config needs into the contiguous cache."""
    summary = populate_cache(config_text)
    cache_vol.commit()
    print(
        f"[cache] dir={summary['cache_dir']} reset={summary['reset']} "
        f"built={len(summary['built'])} entries={summary['n_entries']}",
        flush=True,
    )
    return summary


# Concurrent-GPU cap: at most this many per-layer containers run at once.
MAX_PARALLEL = 24


@app.function(
    gpu="T4",
    volumes={CACHE_DIR: cache_vol, TRAINING_DIR: training_vol},
    cpu=4,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
    max_containers=MAX_PARALLEL,
)
def train_layer(config_text: str, run_id: str, task: dict) -> dict:
    """Train exactly one layer on its own small GPU; return its table row."""
    import matplotlib

    matplotlib.use("Agg")
    import torch

    torch.cuda.set_device(0)
    cfg = load_config(config_text)
    cache_dir = cfg["cache"]["dir"]
    manifest = load_cache_manifest(cache_dir)
    if manifest is None:
        raise ValueError(f"no cache at {cache_dir}; run `build_cache` first")
    dims = manifest["dims"]
    ratios = tuple(float(r) for r in cfg["eval"]["ready_ratios"])
    ks = [int(k) for k in cfg["eval"]["ks"]]
    chunk = int(cfg["eval"]["chunk_tokens"])
    out_dir = os.path.join(cfg["output"]["volume_dir"], run_id)

    row = _run_layer(
        cfg,
        dims,
        manifest,
        task,
        int(cfg["seed"]),
        ks,
        ratios,
        chunk,
        config_text,
        run_id,
        out_dir,
    )
    training_vol.commit()
    return row


@app.function(
    volumes={CACHE_DIR: cache_vol, TRAINING_DIR: training_vol},
    cpu=4,
    memory=16 * 1024,
    timeout=4 * 60 * 60,
)
def train_all(config_text: str, run_id: str, layers: str = "") -> dict:
    """Fan one GPU container out per layer, then write the run-level aggregate."""
    import plots

    cfg = load_config(config_text)
    cache_dir = cfg["cache"]["dir"]
    manifest = load_cache_manifest(cache_dir)
    if manifest is None:
        raise ValueError(f"no cache at {cache_dir}; run `build_cache` first")
    dims = manifest["dims"]
    tasks = _select_layers(resolve_tasks(cfg, dims), layers)
    ks = [int(k) for k in cfg["eval"]["ks"]]

    results = train_layer.starmap(
        [(config_text, run_id, task) for task in tasks],
        return_exceptions=True,
    )
    rows = [result for result in results if not isinstance(result, BaseException)]
    failures = [
        str(result) for result in results if isinstance(result, BaseException)
    ]

    out_dir = os.path.join(cfg["output"]["volume_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)
    png = plots.recall_by_layer_png(rows, ks)
    summary = write_aggregate(out_dir, run_id, cfg, rows, png)
    training_vol.commit()

    print(f"[train_all] {len(rows)}/{len(tasks)} layers done", flush=True)
    for failure in failures:
        print(f"[train_all] failed: {failure}", flush=True)
    return {
        "run_id": run_id,
        "rows": rows,
        "summary": summary,
        "png": png,
        "failures": failures,
    }


def _run_layer(
    cfg: dict,
    dims: dict,
    manifest: dict,
    task: dict,
    seed: int,
    ks: list[int],
    ratios: tuple[float, ...],
    chunk: int,
    config_text: str,
    run_id: str,
    out_dir: str,
) -> dict:
    """Train, evaluate and persist one layer; return its aggregate-table row."""
    import torch

    import plots
    import training

    layer = task["layer"]
    cache_dir = cfg["cache"]["dir"]
    torch.manual_seed(seed + layer)

    # Resident hot-expert sets from 4_analysis (copied into the cache) and the
    # reference selection bias; the bias is applied only at top-k selection,
    # never to the pre-bias score the model is trained on.
    bias = load_cached_bias(cache_dir, manifest, layer)
    resident, hot_experts = training.resident_report(
        cache_dir, manifest, layer, ratios, dims["n_experts"]
    )
    train, ev = training.load_frames(cfg, manifest, task, seed)
    n_train = int(train.x.shape[0])
    n_eval = int(ev.x.shape[0])
    d_in = int(train.x.shape[1])
    model, optimizer, scheduler = training.build_training(cfg, dims, d_in, n_train)

    def evaluate() -> dict[str, float]:
        return training.evaluate(model, ev, ks, bias, resident, chunk)

    history, best_state, best_epoch, best_eval_kl = training.run_epochs(
        cfg, model, optimizer, scheduler, train, ks, ratios, evaluate
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate()

    row_id = (
        f"L{layer:02d}_{cfg['model']['arch']}"
        f"_s{task['source_layer']:02d}_{task['input_kind']}"
    )
    layer_dir = os.path.join(out_dir, f"L{layer:02d}")
    os.makedirs(layer_dir, exist_ok=True)
    training.save_checkpoint(model, os.path.join(layer_dir, "checkpoint.safetensors"))
    summary = training.build_summary(
        run_id,
        row_id,
        cfg,
        dims,
        task,
        d_in,
        model,
        n_train,
        n_eval,
        best_epoch,
        best_eval_kl,
        final,
        history,
    )
    write_layer_artifacts(
        layer_dir,
        config_text,
        summary,
        hot_experts,
        plots.training_curves_png(history, ks, row_id),
        plots.ready_recall_png(history, ks, ratios, row_id),
    )
    print(
        f"    L{layer:02d} src=L{task['source_layer']:02d} d={task['distance']} "
        f"best_epoch={best_epoch} eval_kl={best_eval_kl:.5f}",
        flush=True,
    )
    return row_from_summary(summary)


def _select_layers(tasks: list[dict], layers: str) -> list[dict]:
    """Filter the resolved tasks to the comma-separated ``--layers`` subset."""
    wanted = [int(part) for part in layers.split(",") if part.strip()]
    if not wanted:
        return tasks
    keep = set(wanted)
    selected = [task for task in tasks if task["layer"] in keep]
    if not selected:
        raise ValueError(f"none of layers {sorted(keep)} are trainable")
    return selected


def _read_config(config: str) -> tuple[str, str]:
    """Resolve the config path (falling back to the example) and read it."""
    path = config or os.path.join(THIS_DIR, "config.yaml")
    if not os.path.exists(path):
        fallback = os.path.join(THIS_DIR, "config.example.yaml")
        print(f"warning: {path} not found; using {fallback}")
        path = fallback
    with open(path, encoding="utf-8") as handle:
        return path, handle.read()


def _new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


@app.local_entrypoint()
def main(config: str = "", build_only: bool = False, layers: str = "") -> None:
    path, config_text = _read_config(config)
    run_id = _new_run_id()
    print(f"config: {path}\nrun_id: {run_id}")

    # Build/reuse the all-layer cache before the GPU containers are created.
    print_cache_summary(build_cache.remote(config_text=config_text))
    if build_only:
        print("build-only requested; stopping before training.")
        return

    result = train_all.remote(config_text=config_text, run_id=run_id, layers=layers)
    out_dir = mirror_run(
        OUT_DIR, run_id, result["summary"], result["rows"], result["png"]
    )
    ks = [int(k) for k in result["summary"]["config"]["eval"]["ks"]]
    print_run_report(run_id, result["rows"], out_dir, ks, result["failures"])
