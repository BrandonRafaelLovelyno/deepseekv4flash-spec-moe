"""Single-variant routing-predictor prototype with per-epoch evaluation.

Two functions:

* ``build_cache`` (CPU only) re-lays the harvest into contiguous per-layer files
  once, keyed to the config's ``(task.layer, task.distance, task.input)``. This
  is what keeps the expensive strided slice reading off the GPU.
* ``train_variant`` (L4) mounts only the cache, reads two contiguous files,
  trains, and evaluates at the end of every epoch on the held-out split --
  reporting both the KL loss (for tuning) and the practical expert-selection
  recall, so overfitting is visible as it starts.

All knobs live in ``5_train_one/config.yaml`` (copy ``config.example.yaml``).
The local entrypoint runs the cache build first (a no-op when fresh), so the GPU
container is not created until the data is ready.

Artifacts land under ``<output.volume_dir>/<run_id>/`` on the
``deepseek-v4-flash-training`` volume and are mirrored to
``5_train_one/output/<run_id>/``.

Usage:
    modal run 5_train_one/main.py
    modal run 5_train_one/main.py --build-only
    modal run 5_train_one/main.py --config 5_train_one/other.yaml
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
    resolve_task,
)
from reporting import (
    mirror_result,
    print_cache_summary,
    print_run_report,
    write_run_artifacts,
)

APP_NAME = "deepseek-v4-flash-train-one"

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
    """CPU-only: re-lay the layers the config needs into the contiguous cache."""
    summary = populate_cache(config_text)
    cache_vol.commit()
    print(
        f"[cache] dir={summary['cache_dir']} reset={summary['reset']} "
        f"built={len(summary['built'])} entries={summary['n_entries']}",
        flush=True,
    )
    return summary


@app.function(
    gpu="L4",
    volumes={CACHE_DIR: cache_vol, TRAINING_DIR: training_vol},
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
)
def train_variant(config_text: str, run_id: str) -> dict:
    """Train one predictor off the contiguous cache and evaluate every epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import torch

    import plots
    import training

    cfg = load_config(config_text)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.cuda.set_device(0)

    cache_dir = cfg["cache"]["dir"]
    manifest = load_cache_manifest(cache_dir)
    if manifest is None:
        raise ValueError(f"no cache at {cache_dir}; run `build_cache` first")
    dims = manifest["dims"]
    task = resolve_task(cfg, dims)
    layer = task["layer"]
    ratios = tuple(float(r) for r in cfg["eval"]["ready_ratios"])
    ks = [int(k) for k in cfg["eval"]["ks"]]

    # Resident hot-expert sets from 4_analysis (copied into the cache): the
    # hottest-k experts for this layer per ratio. "ready recall" unions them with
    # the model's prediction. The reference selects experts by top-k(score +
    # bias); the bias is applied only at selection, never to the pre-bias score
    # the model is trained on.
    bias = load_cached_bias(cache_dir, manifest, layer)
    resident, hot_experts = training.resident_report(
        cache_dir, manifest, layer, ratios, dims["n_experts"]
    )

    train, ev = training.load_frames(cfg, manifest, task, seed)
    n_train = int(train.x.shape[0])
    n_eval = int(ev.x.shape[0])
    d_in = int(train.x.shape[1])
    model, optimizer, scheduler = training.build_training(cfg, dims, d_in, n_train)
    chunk = int(cfg["eval"]["chunk_tokens"])

    def evaluate() -> dict[str, float]:
        return training.evaluate(model, ev, ks, bias, resident, chunk)

    history, best_state, best_epoch, best_eval_kl = training.run_epochs(
        cfg, model, optimizer, scheduler, train, ks, ratios, evaluate
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate()

    arch = cfg["model"]["arch"]
    row_id = f"L{layer:02d}_{arch}_d{task['distance']}_{task['input_kind']}"
    out_dir = os.path.join(cfg["output"]["volume_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)
    training.save_checkpoint(model, os.path.join(out_dir, "checkpoint.safetensors"))
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
    write_run_artifacts(out_dir, config_text, summary, hot_experts)
    training_vol.commit()

    return {
        "run_id": run_id,
        "row_id": row_id,
        "summary": summary,
        "png": plots.training_curves_png(history, ks, row_id),
        "ready_png": plots.ready_recall_png(history, ks, ratios, row_id),
        "hot_experts": hot_experts,
    }


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
def main(config: str = "", build_only: bool = False) -> None:
    path, config_text = _read_config(config)
    run_id = _new_run_id()
    print(f"config: {path}\nrun_id: {run_id}")

    # Build/reuse the cache before the GPU container is created.
    print_cache_summary(build_cache.remote(config_text=config_text))
    if build_only:
        print("build-only requested; stopping before training.")
        return

    result = train_variant.remote(config_text=config_text, run_id=run_id)
    out_dir = mirror_result(OUT_DIR, run_id, config_text, result)
    print_run_report(result, out_dir)
