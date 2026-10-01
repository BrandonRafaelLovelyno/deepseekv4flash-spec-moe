"""Average distinct experts per prefill chunk, per chunk size and layer.

The article companion to ``7_final_simulation``: instead of predicting and
schooling a resident set, it measures the *ground truth* of speculative expert
loading -- for each chunk size ``B`` (and each decode portion of a mixed
chunked-prefill load), how many distinct routed experts a chunk's tokens
activate together, averaged over chunks and reported per layer.

Because the metric is read straight from the cached truth top-k, the whole stage
is CPU only: no predictor, no GPU and no training run. It reuses
``7_final_simulation``'s plan build and replay shape, stripped to the chunk
sweep.

All knobs live in ``8_article_writing/average_used_experts/config.yaml`` (copy
``config.example.yaml``). Artifacts land on the ``deepseek-v4-flash-article``
volume under ``<run_id>/`` and are mirrored to ``output/<run_id>/``.

Usage:
    modal run 8_article_writing/average_used_experts/main.py
    modal run 8_article_writing/average_used_experts/main.py --config <other>.yaml
    modal run 8_article_writing/average_used_experts/main.py --run-id 20261001T000000Z
"""

import os
import sys
import time

import modal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from reporting import mirror_run, print_report

APP_NAME = "deepseek-v4-flash-average-used-experts"

OUT_DIR = os.path.join(THIS_DIR, "output")

HARVEST_DIR = "/harvest"
CACHE_DIR = "/cache"
ARTICLE_DIR = "/article"

harvest_vol = modal.Volume.from_name(
    "deepseek-v4-flash-harvest", create_if_missing=True
)
cache_vol = modal.Volume.from_name(
    "deepseek-v4-flash-train-cache", create_if_missing=True
)
article_vol = modal.Volume.from_name(
    "deepseek-v4-flash-article", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("safetensors", "numpy", "matplotlib", "pyyaml")
    .env({"MPLBACKEND": "Agg"})
    .add_local_python_source("helper", "streams", "usage", "reporting", "plots")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={
        HARVEST_DIR: harvest_vol,
        CACHE_DIR: cache_vol,
        ARTICLE_DIR: article_vol,
    },
    cpu=8,
    memory=32 * 1024,
    timeout=4 * 60 * 60,
)
def analyze(config_text: str, run_id: str) -> dict:
    """CPU only: sweep the cached truth top-k and write the run's artifacts."""
    import matplotlib

    matplotlib.use("Agg")

    import plots
    import reporting
    import usage

    payload = usage.simulate(config_text, run_id)
    figures = plots.build_figures(payload)
    out_dir = os.path.join(
        payload["summary"]["config"]["output"]["volume_dir"], run_id
    )
    written = reporting.write_run(out_dir, config_text, payload, figures)
    article_vol.commit()
    print(f"[analyze] wrote {len(written)} files to {out_dir}", flush=True)
    return {
        "summary": payload["summary"],
        "rows": payload["rows"],
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
def main(config: str = "", run_id: str = "") -> None:
    path, config_text = _read_config(config)
    article_run_id = run_id or _new_run_id()
    print(f"config: {path}\nrun_id: {article_run_id}")

    result = analyze.remote(config_text=config_text, run_id=article_run_id)
    out_dir = mirror_run(
        OUT_DIR,
        article_run_id,
        result["summary"],
        result["rows"],
        result["figures"],
    )
    print_report(article_run_id, result["summary"], out_dir)
