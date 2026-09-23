"""Encode every DeepSeek-V4-Flash trajectory into the DeepSeek-V4 prompt format.

Downloads the full trajectory corpora into a Modal Volume, encodes every
trajectory with DeepSeek's own chat encoder (``encoding/encoding_dsv4.py`` from
the checkpoint), and writes the encoded prompts plus a manifest to a second
volume. One run covers all four sources:

    yi30-think     Yi30/deepseek-v4-swebench-trajectories, ``data/think_high``
    yi30-nothink   same repo, ``data/no_think``
    terminus2      openguardrails/terminal-bench-2.1-deepseek-v4-flash-trajectories,
                   ``trajectories/terminus2/<task>/agent/trajectory.json`` (ATIF)
    dsh            same repo, ``trajectories/dsh/<task>/agent/dsh-session.jsonl``

Each source is encoded in its own native mode (thinking for the reasoning
corpora, chat for ``no_think`` and ``dsh``, which carry no reasoning). Tool
schemas are attached only when the source actually ships them (``dsh``); no
schemas are synthesized for the others.

Reasoning is preserved across the whole transcript: ``encode_messages`` is
called with ``drop_thinking=False`` so the historical ``<think>`` blocks survive
(otherwise the encoder's default discards every reasoning block before the last
user turn, which would defeat the point of these DeepSeek-generated corpora).

No model and no GPU are involved: ``encode_messages`` is pure Python, so this is
a cheap CPU-only data check. No reasoning-effort paragraph is ever injected.

Pipeline (``modal run 2_data_check/main.py``):
    1. ``download_sources`` (CPU, idempotent) caches both full HF repos under the
       ``deepseek-v4-flash-datasets`` volume.
    2. ``encode_all`` (CPU) walks every trajectory, encodes it, and writes
       ``<source>/<key>.txt`` plus ``manifest.json`` to the
       ``deepseek-v4-flash-encoded`` volume.
    3. The local entrypoint writes the manifest to
       ``2_data_check/output/manifest.json`` and prints a summary.

Inspect an individual prompt with:
    modal volume get deepseek-v4-flash-encoded <source>/<key>.txt

Prerequisites:
    modal secret create huggingface-secret HF_TOKEN=hf_...
    modal run 1_weight_download/main.py    # caches the checkpoint in the volume

Usage:
    modal run 2_data_check/main.py
"""

import json
import os
import sys
from dataclasses import asdict

import modal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from helper import (  # noqa: E402  (import path set up above)
    TB_REPO,
    YI30_REPO,
    EncodeResult,
    ErrorRow,
    _encode_item,
    _enumerate_items,
    _summarize_results,
)

APP_NAME = "deepseek-v4-flash-data-check"

BASE_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

DATASET_DIR = "/datasets"
ENCODED_DIR = "/encoded"
CKPT_HF_DIR = "/root/.cache/huggingface"

OUT_DIR = os.path.join(THIS_DIR, "output")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

dataset_vol = modal.Volume.from_name(
    "deepseek-v4-flash-datasets", create_if_missing=True
)
encoded_vol = modal.Volume.from_name(
    "deepseek-v4-flash-encoded", create_if_missing=True
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]")
    .env(
        {
            "HF_HOME": DATASET_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_python_source("helper")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={DATASET_DIR: dataset_vol},
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
    cpu=8,
    memory=16 * 1024,
)
def download_sources() -> dict:
    """Cache the full trajectory repos in the Modal Volume."""
    from huggingface_hub import snapshot_download

    paths = {}
    for repo in (YI30_REPO, TB_REPO):
        paths[repo] = snapshot_download(
            repo_id=repo, repo_type="dataset", max_workers=8
        )
    dataset_vol.commit()
    print(json.dumps(paths, indent=2))
    return paths


@app.function(
    volumes={
        DATASET_DIR: dataset_vol,
        CKPT_HF_DIR: hf_cache_vol,
        ENCODED_DIR: encoded_vol,
    },
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
    cpu=8,
    memory=32 * 1024,
)
def encode_all() -> dict:
    """Encode every trajectory and write the prompts plus a manifest to the volume."""
    import time

    from huggingface_hub import snapshot_download

    try:
        ckpt = snapshot_download(
            repo_id=BASE_REPO,
            revision=BASE_REV,
            cache_dir=CKPT_HF_DIR,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{BASE_REPO} @ {BASE_REV} is not cached in the mounted volume; "
            "run `modal run 1_weight_download/main.py` first."
        ) from exc

    encoding_dir = os.path.join(ckpt, "encoding")
    if encoding_dir not in sys.path:
        sys.path.insert(0, encoding_dir)
    from encoding_dsv4 import encode_messages

    snapshots = {
        repo: snapshot_download(repo_id=repo, repo_type="dataset", local_files_only=True)
        for repo in (YI30_REPO, TB_REPO)
    }
    items = _enumerate_items(snapshots)
    print(f"encoding {len(items)} trajectories")

    results: list[EncodeResult] = []
    errors: list[ErrorRow] = []
    for item in items:
        try:
            result = _encode_item(item, encode_messages, ENCODED_DIR)
        except Exception as exc:  # keep going; report the bad row in the manifest
            errors.append({"source": item.source, "key": item.key, "error": repr(exc)})
            print(f"  FAILED {item.source}/{item.key}: {exc!r}")
            continue
        results.append(result)
        print(
            f"  {result.source}/{result.key}: {result.n_messages} msgs, "
            f"{result.n_chars} chars ({result.n_reasoning_chars} reasoning)"
        )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_items": len(results),
        "n_errors": len(errors),
        "sources": _summarize_results(results),
        "items": [asdict(result) for result in results],
        "errors": errors,
    }
    with open(os.path.join(ENCODED_DIR, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    encoded_vol.commit()

    print(json.dumps(manifest["sources"], indent=2))
    return manifest


@app.local_entrypoint()
def main() -> None:
    print(download_sources.remote())
    manifest = encode_all.remote()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\n===== encoded corpus summary =====")
    for source, stats in manifest["sources"].items():
        if not stats.get("count"):
            print(f"{source:14} : 0 items")
            continue
        print(
            f"{source:14} : {stats['count']:>3} items | "
            f"chars min/med/max {stats['min_chars']}/{stats['median_chars']}/"
            f"{stats['max_chars']} | reasoning {stats['total_reasoning_chars']}"
        )
    print(f"errors         : {manifest['n_errors']}")
    print(f"written to     : {ENCODED_DIR}/<source>/<key>.txt")
    print(f"manifest       : {MANIFEST_PATH}")
