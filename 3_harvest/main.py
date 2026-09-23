"""Harvest per-token expert-routing data from DeepSeek-V4-Flash (MXFP4 base).

Runs a **single whole-sequence prefill** (``start_pos=0``) per trajectory slice
through DeepSeek's own reference implementation (``inference/model.py``) and
records, for every token in every one of the 43 blocks:

    * the expanded residual state -- the Hyper-Connections block input,
      shape ``[hc_mult=4, dim=4096]``;
    * the exact MoE router input -- the post-``ffn_norm`` compressed residual,
      shape ``[dim=4096]``;
    * the full 256-way pre-bias router score vector -- ``sqrt(softplus(Wx))``,
      shape ``[n_routed_experts=256]`` (the reference gate uses
      ``score_func="sqrtsoftplus"``);
    * the reference gate's exact top-k output -- the selected expert ids and
      their routed weights, shape ``[top_k=6]``. These are the authoritative
      routing record; the scores reproduce them only approximately, since they
      are stored in ``float16``.

Static router state. Two model-global tables are written once at the harvest
root, because they are identical for every token and every slice:
``router_bias.safetensors`` holds the per-layer selection bias (zeros on the
hash layers) and ``hash_table.safetensors`` holds the ``tid2eid`` token-id ->
expert-id tables that alone determine routing on layers ``0..n_hash_layers-1``.

Slices. The reference attention only supports a full prefill (``start_pos=0``)
or one-token decode, so every slice is a **prefix** ``[0:n]`` of one session.
For each of the four encoded datasets we pick, with a fixed seed, four training
sessions (24k tokens each) and two held-out test sessions (5k tokens each). The
per-slice context cap stays well inside the Indexer's quadratic prefill budget.

Enrichment. Each output carries ``dataset_id`` and ``document_id``, and every
token is labelled as prefill or decode. A token is **decode** if it lies inside
an assistant turn -- from after ``<｜Assistant｜>`` (skipping the leading
``<think>``/``</think>`` generation marker) through ``<｜end▁of▁sentence｜>``
inclusive, i.e. reasoning + content + tool calls + EOS -- and **prefill**
otherwise (system + tools template, user turns, tool results, scaffolding).
Each token also gets ``token_idx`` (0-based position within its phase) and
``phase_index`` (ordinal of its phase, a maximal run of equal label).

Payload. The residual states are stored as ``float8_e4m3fn`` with a per-tensor
scale (recorded in the sidecar), the router scores as ``float16``, the expert ids
as ``uint8`` and the routed weights as ``float16``, so a slice costs ~0.90 MB/token
instead of ~1.76 MB/token.

Pipeline (``modal run 3_harvest/main.py``):
    1. ``convert_ckpt`` (CPU, idempotent) reuses the converted MXFP4 checkpoint
       under the ``/ckpt`` volume (same artifact ``inference/main.py`` produces).
    2. ``harvest_all`` (single B300) loads the model once and harvests every
       selected slice, writing one safetensors file plus JSON sidecar per slice
       under ``/harvest/<dataset_id>/<split>/``, the two static router tables at
       the harvest root, and a corpus ``manifest.json``.

Prerequisites:
    modal secret create huggingface-secret HF_TOKEN=hf_...
    modal run 1_weight_download/main.py    # caches the MXFP4 base in the volume
    modal run 2_data_check/main.py         # fills the encoded volume
"""

import json
import os
import sys

import modal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from helper import (  # noqa: E402  (import path set up above)
    TEST_SESSIONS,
    TEST_TOKENS,
    TRAIN_SESSIONS,
    TRAIN_TOKENS,
    _extract_router_static,
    _harvest_forward,
    _install_capture,
    _load_streaming,
    _quantize_fp8,
    _read_sessions,
    _select_slices,
    _token_labels,
)

APP_NAME = "deepseek-v4-flash-harvest"

BASE_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

HF_CACHE_DIR = "/root/.cache/huggingface"
CKPT_DIR = "/ckpt"
TILELANG_CACHE_DIR = "/root/.cache/tilelang"
HARVEST_DIR = "/harvest"
ENCODED_DIR = "/encoded"

CKPT_FILE = "model0-mp1.safetensors"
N_EXPERTS = 256
MODEL_PARALLEL = 1

GPU = "B300"
MAX_SEQ_LEN = 32768
SEED = 33377335

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("deepseek-v4-flash-ckpt", create_if_missing=True)
tilelang_cache_vol = modal.Volume.from_name("tilelang-cache", create_if_missing=True)
encoded_vol = modal.Volume.from_name(
    "deepseek-v4-flash-encoded", create_if_missing=True
)
harvest_vol = modal.Volume.from_name(
    "deepseek-v4-flash-harvest", create_if_missing=True
)
hf_secret = modal.Secret.from_name("huggingface-secret")

# Same CUDA *devel* base as ``inference/main.py``: TileLang JIT and
# fast_hadamard_transform both need nvcc, and B300 (sm_103) needs CUDA 13.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "build-essential", "ninja-build", "curl")
    .uv_pip_install(
        "torch==2.14.0+cu130",
        index_url="https://download.pytorch.org/whl/cu130",
        extra_index_url="https://pypi.org/simple",
        extra_options="--index-strategy unsafe-best-match",
    )
    .uv_pip_install(
        "transformers>=5.0.0",
        "safetensors>=0.7.0",
        "tilelang==0.1.8",
        "apache-tvm-ffi==0.1.8.post2",
        "tqdm",
        "setuptools",
        "wheel",
        "ninja",
        "packaging",
    )
    .uv_pip_install(
        "fast_hadamard_transform @ "
        "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0",
        extra_options="--no-build-isolation",
        env={
            "CC": "gcc",
            "CXX": "g++",
            "CUDA_HOME": "/usr/local/cuda",
            "FAST_HADAMARD_TRANSFORM_FORCE_BUILD": "TRUE",
        },
    )
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TILELANG_CACHE_DIR": TILELANG_CACHE_DIR,
        }
    )
    .add_local_python_source("helper")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={HF_CACHE_DIR: hf_cache_vol, CKPT_DIR: ckpt_vol},
    secrets=[hf_secret],
    cpu=16,
    memory=200 * 1024,
    timeout=6 * 60 * 60,
)
def convert_ckpt(force: bool = False) -> str:
    """Convert the HF checkpoint into DeepSeek's MP-reference format."""
    import subprocess

    from huggingface_hub import snapshot_download

    target = os.path.join(CKPT_DIR, CKPT_FILE)
    if os.path.exists(target) and not force:
        print(f"converted checkpoint already present: {target}")
        return target

    snapshot = snapshot_download(
        repo_id=BASE_REPO, revision=BASE_REV, max_workers=16
    )
    convert = os.path.join(snapshot, "inference", "convert.py")
    cmd = [
        sys.executable,
        convert,
        "--hf-ckpt-path",
        snapshot,
        "--save-path",
        CKPT_DIR,
        "--n-experts",
        str(N_EXPERTS),
        "--model-parallel",
        str(MODEL_PARALLEL),
        "--expert-dtype",
        "fp4",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=os.path.dirname(convert))
    ckpt_vol.commit()
    print(f"wrote {target}")
    return target


@app.function(
    gpu=GPU,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        CKPT_DIR: ckpt_vol,
        TILELANG_CACHE_DIR: tilelang_cache_vol,
        ENCODED_DIR: encoded_vol,
        HARVEST_DIR: harvest_vol,
    },
    secrets=[hf_secret],
    cpu=8,
    memory=96 * 1024,
    timeout=6 * 60 * 60,
)
def harvest_all(max_seq_len: int = MAX_SEQ_LEN, seed: int = SEED) -> dict:
    """Load the model once and harvest every selected slice."""
    import random
    import time

    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=BASE_REPO, revision=BASE_REV, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"{BASE_REPO} @ {BASE_REV} is not cached in the mounted volume; "
            "run `modal run 1_weight_download/main.py` first."
        ) from exc
    for sub in ("inference", "encoding"):
        path = os.path.join(snapshot, sub)
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from model import ModelArgs, Transformer

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    torch.cuda.set_device(0)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    if not tokenizer.is_fast:
        raise RuntimeError("need a fast tokenizer for offset mapping")

    sessions = _read_sessions(ENCODED_DIR)
    plan = _select_slices(sessions, tokenizer, random.Random(seed))

    with open(os.path.join(snapshot, "inference", "config.json")) as handle:
        args = ModelArgs(**json.load(handle))
    args.max_batch_size = 1
    args.max_seq_len = max(max_seq_len, TRAIN_TOKENS)
    print(args)

    with torch.device("cuda"):
        model = Transformer(args)
    _load_streaming(model, os.path.join(CKPT_DIR, CKPT_FILE))
    torch.set_default_device("cuda")

    n_layers = len(model.layers)
    router_bias, hash_table = _extract_router_static(model, torch)
    if hash_table is not None:
        lo, hi = int(hash_table.min()), int(hash_table.max())
        if lo < 0 or hi >= args.n_routed_experts:
            raise RuntimeError(
                "hash-layer tid2eid tables look uninitialized: "
                f"range [{lo}, {hi}] is outside [0, {args.n_routed_experts}). "
                "Check that convert.py preserved `tid2eid` and that "
                "_load_streaming loaded `gate.tid2eid`."
            )
    os.makedirs(HARVEST_DIR, exist_ok=True)
    save_file(
        {"bias": router_bias},
        os.path.join(HARVEST_DIR, "router_bias.safetensors"),
    )
    if hash_table is not None:
        save_file(
            {"tid2eid": hash_table},
            os.path.join(HARVEST_DIR, "hash_table.safetensors"),
        )
    harvest_vol.commit()
    print(
        f"router static: bias {tuple(router_bias.shape)}, "
        f"hash tables {'none' if hash_table is None else tuple(hash_table.shape)}"
    )

    buffers = {}
    _install_capture(model, buffers, torch)
    records = []
    errors = []

    for dataset, slices in plan.items():
        for plan_slice in slices:
            session = plan_slice.session
            split = plan_slice.split
            n_tokens = plan_slice.n_tokens
            if n_tokens <= 0:
                continue
            try:
                encoding = tokenizer(
                    session["text"],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                ids = encoding["input_ids"][:n_tokens]
                offsets = encoding["offset_mapping"][:n_tokens]
                n_tokens = len(ids)
                is_decode, turn_index, token_idx, phase_index = _token_labels(
                    session["text"], offsets
                )

                for key in buffers:
                    buffers[key] = None
                buffers["expanded"] = torch.empty(
                    (n_tokens, n_layers, args.hc_mult, args.dim),
                    dtype=torch.bfloat16,
                    device="cpu",
                )
                buffers["compressed"] = torch.empty(
                    (n_tokens, n_layers, args.dim),
                    dtype=torch.bfloat16,
                    device="cpu",
                )
                buffers["scores"] = torch.empty(
                    (n_tokens, n_layers, args.n_routed_experts),
                    dtype=torch.float16,
                    device="cpu",
                )
                buffers["topk_ids"] = torch.empty(
                    (n_tokens, n_layers, args.n_activated_experts),
                    dtype=torch.uint8,
                    device="cpu",
                )
                buffers["topk_weights"] = torch.empty(
                    (n_tokens, n_layers, args.n_activated_experts),
                    dtype=torch.float16,
                    device="cpu",
                )

                input_ids = torch.tensor([ids], dtype=torch.long)
                t0 = time.monotonic()
                _harvest_forward(model, input_ids, torch)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - t0
                torch.cuda.empty_cache()

                expanded_fp8, expanded_scale = _quantize_fp8(
                    buffers["expanded"], torch
                )
                compressed_fp8, compressed_scale = _quantize_fp8(
                    buffers["compressed"], torch
                )

                out_dir = os.path.join(HARVEST_DIR, dataset, split)
                os.makedirs(out_dir, exist_ok=True)
                stem = os.path.join(out_dir, session["document_id"])
                tensors = {
                    "expanded": expanded_fp8,
                    "compressed": compressed_fp8,
                    "scores": buffers["scores"],
                    "topk_ids": buffers["topk_ids"],
                    "topk_weights": buffers["topk_weights"],
                    "token_ids": torch.tensor(ids, dtype=torch.int32),
                    "is_decode": torch.tensor(is_decode, dtype=torch.uint8),
                    "turn_index": torch.tensor(turn_index, dtype=torch.int16),
                    "token_idx": torch.tensor(token_idx, dtype=torch.int32),
                    "phase_index": torch.tensor(phase_index, dtype=torch.int16),
                }
                payload_gb = sum(
                    t.numel() * t.element_size() for t in tensors.values()
                ) / 1e9
                save_file(tensors, f"{stem}.safetensors")

                meta = {
                    "dataset_id": dataset,
                    "document_id": session["document_id"],
                    "split": split,
                    "seed": seed,
                    "n_tokens": n_tokens,
                    "session_total_tokens": session["n"],
                    "token_offset": 0,
                    "truncated": bool(n_tokens < session["n"]),
                    "n_prefill": int(n_tokens - sum(is_decode)),
                    "n_decode": int(sum(is_decode)),
                    "n_turns": int(turn_index[-1]) if turn_index else 0,
                    "n_phases": (int(phase_index[-1]) + 1) if phase_index else 0,
                    "token_idx_base": 0,
                    "expanded_scale": expanded_scale,
                    "compressed_scale": compressed_scale,
                    "expanded_dtype": "float8_e4m3fn",
                    "compressed_dtype": "float8_e4m3fn",
                    "router_scores_dtype": "float16",
                    "topk_ids_dtype": "uint8",
                    "topk_weights_dtype": "float16",
                    "score_func": args.score_func,
                    "route_scale": args.route_scale,
                    "n_hash_layers": args.n_hash_layers,
                    "payload_gb": round(payload_gb, 3),
                    "model_repo": BASE_REPO,
                    "model_revision": BASE_REV,
                    "n_layers": n_layers,
                    "dim": args.dim,
                    "hc_mult": args.hc_mult,
                    "top_k": args.n_activated_experts,
                    "n_routed_experts": args.n_routed_experts,
                    "max_seq_len": args.max_seq_len,
                    "elapsed_seconds": round(elapsed, 2),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with open(f"{stem}.json", "w") as handle:
                    json.dump(meta, handle, indent=2)

                del expanded_fp8, compressed_fp8, tensors
                records.append({"path": f"{stem}.safetensors", **meta})
                print(
                    f"  {dataset}/{split}/{session['document_id']}: "
                    f"{n_tokens} tokens "
                    f"({meta['n_prefill']} prefill / {meta['n_decode']} decode), "
                    f"{payload_gb:.2f} GB, {elapsed:.1f}s"
                )
            except Exception as exc:  # keep going; report the bad slice
                errors.append(
                    {
                        "dataset_id": dataset,
                        "document_id": session["document_id"],
                        "split": split,
                        "error": repr(exc),
                    }
                )
                print(f"  FAILED {dataset}/{session['document_id']}: {exc!r}")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "score_func": args.score_func,
        "route_scale": args.route_scale,
        "n_hash_layers": args.n_hash_layers,
        "n_routed_experts": args.n_routed_experts,
        "router_bias_path": os.path.join(HARVEST_DIR, "router_bias.safetensors"),
        "hash_table_path": (
            os.path.join(HARVEST_DIR, "hash_table.safetensors")
            if hash_table is not None
            else None
        ),
        "train_sessions": TRAIN_SESSIONS,
        "train_tokens": TRAIN_TOKENS,
        "test_sessions": TEST_SESSIONS,
        "test_tokens": TEST_TOKENS,
        "n_slices": len(records),
        "n_errors": len(errors),
        "total_tokens": sum(r["n_tokens"] for r in records),
        "records": records,
        "errors": errors,
    }
    with open(os.path.join(HARVEST_DIR, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    harvest_vol.commit()

    print(json.dumps({k: manifest[k] for k in ("n_slices", "n_errors", "total_tokens")}))
    return manifest


@app.local_entrypoint()
def main(skip_convert: bool = False) -> None:
    if not skip_convert:
        print(f"checkpoint: {convert_ckpt.remote()}")
    manifest = harvest_all.remote()
    print("\n===== harvest summary =====")
    print(
        json.dumps(
            {k: manifest[k] for k in ("n_slices", "n_errors", "total_tokens")},
            indent=2,
        )
    )
