"""Final simulation of the trained expert predictor under chunked prefill.

Two stages, split so the GPU is used only where it computes:

* ``predict_all`` (GPU, the only GPU user) forwards each layer's trained routing
  predictor over the harvested source activations and caches the per-token
  predicted top-k experts on the simulation volume.
* ``simulate`` (CPU only) replays the cached predictions under a realistic mixed
  chunked-prefill load. For every chunk it measures how many of the chunk's
  demanded experts are missing after the static hot set is unioned with the top-N
  predicted non-resident experts, and reports the CDF of that miss count.

The static hot set comes from ``4_analysis``'s per-layer expert counts (the
``ready_ratios`` hottest experts); the predictor and its bias come from the
``6_train_all`` run. Both train and test tokens are replayed. Because the
prediction cache depends only on the run, cache and selection knobs, re-tuning
the simulation knobs re-runs the CPU stage alone.

All knobs live in ``7_final_simulation/config.yaml`` (copy
``config.example.yaml``). Artifacts land on the
``deepseek-v4-flash-simulation`` volume under ``<run_id>/`` (and the prediction
cache under ``predictions/<training_run_id>/``); run-level files are mirrored to
``7_final_simulation/output/<run_id>/``.

Usage:
    modal run 7_final_simulation/main.py
    modal run 7_final_simulation/main.py --config 7_final_simulation/other.yaml
    modal run 7_final_simulation/main.py --run-id 20260930T032238Z
    modal run 7_final_simulation/main.py --force-predict
"""

import os
import sys
import time

import modal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from reporting import mirror_run, print_report

APP_NAME = "deepseek-v4-flash-final-simulation"

OUT_DIR = os.path.join(THIS_DIR, "output")

HARVEST_DIR = "/harvest"
CACHE_DIR = "/cache"
TRAINING_DIR = "/training"
SIMULATION_DIR = "/simulation"

harvest_vol = modal.Volume.from_name(
    "deepseek-v4-flash-harvest", create_if_missing=True
)
cache_vol = modal.Volume.from_name(
    "deepseek-v4-flash-train-cache", create_if_missing=True
)
training_vol = modal.Volume.from_name(
    "deepseek-v4-flash-training", create_if_missing=True
)
simulation_vol = modal.Volume.from_name(
    "deepseek-v4-flash-simulation", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "safetensors", "numpy", "matplotlib", "pyyaml")
    .env({"MPLBACKEND": "Agg"})
    .add_local_python_source(
        "helper", "streams", "simulation", "models", "predict", "plots", "reporting"
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="L4",
    volumes={
        CACHE_DIR: cache_vol,
        TRAINING_DIR: training_vol,
        SIMULATION_DIR: simulation_vol,
    },
    cpu=4,
    memory=8 * 1024,
    timeout=2 * 60 * 60,
    scaledown_window=2,
)
def predict_all(config_text: str, force: bool = False) -> dict:
    """GPU only: cache each layer's per-token predicted top-k experts."""
    import predict

    result = predict.predict_all(config_text, force=force)
    simulation_vol.commit()
    return result


@app.function(
    volumes={
        HARVEST_DIR: harvest_vol,
        CACHE_DIR: cache_vol,
        TRAINING_DIR: training_vol,
        SIMULATION_DIR: simulation_vol,
    },
    cpu=8,
    memory=32 * 1024,
    timeout=4 * 60 * 60,
)
def simulate(config_text: str, run_id: str, training_run_id: str = "") -> dict:
    """CPU only: sweep the cached predictions and write the run's artifacts."""
    import matplotlib

    matplotlib.use("Agg")

    import plots
    import reporting
    import simulation

    payload = simulation.simulate(config_text, run_id, training_run_id)
    figures = plots.build_figures(payload)
    out_dir = os.path.join(
        payload["summary"]["config"]["output"]["volume_dir"], run_id
    )
    written = reporting.write_run(out_dir, config_text, payload, figures)
    simulation_vol.commit()
    print(f"[simulate] wrote {len(written)} files to {out_dir}", flush=True)
    return {
        "summary": payload["summary"],
        "distributions": payload["distributions"],
        "figures": figures,
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
def main(config: str = "", run_id: str = "", force_predict: bool = False) -> None:
    path, config_text = _read_config(config)
    sim_run_id = run_id or _new_run_id()
    print(f"config: {path}\nsimulation run_id: {sim_run_id}")

    # GPU stage: forward each layer's predictor once, then release the GPU.
    prediction = predict_all.remote(config_text=config_text, force=force_predict)
    print(
        f"predict: training_run={prediction['training_run_id']} "
        f"predicted={len(prediction['predicted'])} reused={len(prediction['reused'])}"
    )

    # CPU stage: sweep the cached predictions; no GPU is requested.
    result = simulate.remote(
        config_text=config_text,
        run_id=sim_run_id,
        training_run_id=prediction["training_run_id"],
    )
    out_dir = mirror_run(
        OUT_DIR,
        sim_run_id,
        result["summary"],
        result["distributions"],
        result["figures"],
    )
    print_report(sim_run_id, result["summary"], out_dir)
