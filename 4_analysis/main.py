"""Measure speculative-expert-loading miss rates on the harvested routing data.

Reads the per-token expert routing captured by ``3_harvest`` (``topk_ids``,
shape ``[n_tokens, n_layers, top_k]``) and answers: if we keep only the hottest
X% of each layer's experts resident in VRAM, how many of a token's ``top_k``
experts are *missing* (and must be fetched)?

This entrypoint is a thin harness. The actual studies live one-per-file under
``analyses/`` and are run as an ordered queue:

    expert_ranking   rank experts by train frequency; build the resident sets
    token_miss       per-token miss distribution, static vs adaptive LFU cache
    decode_miss      concurrent-decode unique missing experts (B=1..20)
    prefill_miss     concurrent-prefill unique missing experts (B=1..20)
    coverage         decode vs prefill coverage of the non-resident set

Each study reads/writes the shared ``AnalysisContext`` (``helper.py``), writes
its own artifacts into its own subdirectory, and streams ``log`` / ``image`` /
``file`` events back to the local entrypoint. Shared code -- the harvest data
access layer, the residency prerequisite, statistics/plot plumbing -- lives in
``helper.py``. Add a study by dropping a module in ``analyses/`` and appending
it to ``ANALYSES`` below.

Artifacts land under ``/harvest/analysis/<run_id>/<study>/`` on the volume and
are mirrored to ``4_analysis/output/<run_id>/<study>/`` locally; PNGs are
streamed to the local mirror only. CPU only -- no GPU, no torch.

Usage:
    modal run 4_analysis/main.py            # full run over the 4 datasets
    modal run 4_analysis/main.py --quick    # first dataset only, for a smoke test
"""

import os
import sys
import time

import modal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from analyses import (  # noqa: E402  (import path set up above)
    CoverageAnalysis,
    DecodeMissAnalysis,
    ExpertRankingAnalysis,
    PrefillMissAnalysis,
    TokenMissAnalysis,
)
from helper import (  # noqa: E402
    ANALYSIS_DIR,
    HARVEST_DIR,
    AnalysisContext,
    _finalize,
    _prepare,
)

APP_NAME = "deepseek-v4-flash-analysis"

OUT_DIR = os.path.join(THIS_DIR, "output")

# The analysis queue: studies run top to bottom, sharing one AnalysisContext.
ANALYSES = (
    ExpertRankingAnalysis(),
    TokenMissAnalysis(),
    DecodeMissAnalysis(),
    PrefillMissAnalysis(),
    CoverageAnalysis(),
)

harvest_vol = modal.Volume.from_name(
    "deepseek-v4-flash-harvest", create_if_missing=True
)

# Only routing ids are needed, so numpy + safetensors + matplotlib is enough --
# the CUDA / torch stack the harvest image carries would be dead weight.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "safetensors", "matplotlib")
    .env({"MPLBACKEND": "Agg"})
    .add_local_python_source("helper", "analyses")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={HARVEST_DIR: harvest_vol},
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
)
def analyze(quick: bool = False):
    """Prepare the context, run the analysis queue, stream every event back."""
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    yield {"type": "start", "run_id": run_id}

    ctx = AnalysisContext(
        quick=quick,
        run_id=run_id,
        remote_out_dir=os.path.join(ANALYSIS_DIR, run_id),
    )
    os.makedirs(ctx.remote_out_dir, exist_ok=True)

    yield from _prepare(ctx)
    if not ctx.records:
        return

    for analysis in ANALYSES:
        yield from analysis.run(ctx)

    yield from _finalize(ctx)
    harvest_vol.commit()


@app.local_entrypoint()
def main(quick: bool = False) -> None:
    out_dir = None
    for event in analyze.remote_gen(quick=quick):
        kind = event["type"]
        if kind == "start":
            out_dir = os.path.join(OUT_DIR, event["run_id"])
            os.makedirs(out_dir, exist_ok=True)
            print(f"run_id: {event['run_id']}")
        elif kind == "log":
            print(event["text"])
        elif kind in ("image", "file"):
            path = os.path.join(out_dir, event["name"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(event["data"])
            print(f"wrote {path} ({len(event['data']) / 1024:.1f} KiB)")
    if out_dir is not None:
        print(f"\nartifacts mirrored to {out_dir}")
