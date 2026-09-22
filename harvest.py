"""Harvest per-token expert-routing data from DeepSeek-V4-Flash (MXFP4 base).

Runs a **single whole-sequence prefill** (``start_pos=0``) per trajectory slice
through DeepSeek's own reference implementation (``inference/model.py``) and
records, for every token in every one of the 43 blocks:

    * the expanded residual state -- the Hyper-Connections block input,
      shape ``[hc_mult=4, dim=4096]``;
    * the exact MoE router input -- the post-``ffn_norm`` compressed residual,
      shape ``[dim=4096]``;
    * the routed expert ids and their top-k weights, shape ``[top_k=6]``.

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
scale (recorded in the sidecar); expert ids are ``uint8`` and weights ``float16``,
so a slice costs ~0.88 MB/token instead of ~1.76 MB/token.

Pipeline (``modal run harvest.py``):
    1. ``convert_ckpt`` (CPU, idempotent) reuses the converted MXFP4 checkpoint
       under the ``/ckpt`` volume (same artifact ``run_inference.py`` produces).
    2. ``harvest_all`` (single B300) loads the model once and harvests every
       selected slice, writing one safetensors file plus JSON sidecar per slice
       under ``/harvest/<dataset_id>/<split>/`` and a corpus ``manifest.json``.

Prerequisites:
    modal secret create huggingface-secret HF_TOKEN=hf_...
    modal run download_weights.py          # caches the MXFP4 base in the volume
    modal run data_check/data_check.py     # fills the encoded volume
"""

import json
import os
import random
import sys
import time

import modal

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

# Four encoded DeepSeek-V4-Flash corpora, as written by data_check.py.
DATASETS = ("yi30-think", "yi30-nothink", "terminus2", "dsh")

TRAIN_SESSIONS = 4
TRAIN_TOKENS = 24000
TEST_SESSIONS = 2
TEST_TOKENS = 5000
SEED = 33377335

# Region markers in the encoded prompt (fullwidth vertical bars, U+FF5C).
# ``<think>``/``</think>`` directly after ``<｜Assistant｜>`` is generation
# scaffolding (thinking vs chat mode), not model output, so it is not decode.
MARK_ASSISTANT = "<｜Assistant｜>"
MARK_END = "<｜end▁of▁sentence｜>"
MARK_THINK_OPEN = "<think>"
MARK_THINK_CLOSE = "</think>"

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

# Same CUDA *devel* base as ``run_inference.py``: TileLang JIT and
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
)

app = modal.App(APP_NAME, image=image)


def _load_streaming(model, path: str) -> None:
    """Copy a safetensors checkpoint into ``model`` one tensor at a time.

    ``safetensors.torch.load_model`` would materialize the whole 167 GB file in
    host RAM before the copy. Streaming sends each tensor straight to the GPU and
    writes it into the matching parameter storage.
    """
    import torch
    from safetensors import safe_open

    state = model.state_dict()
    loaded = 0
    skipped = 0
    with safe_open(path, framework="pt", device="cuda") as handle:
        for name in handle.keys():
            if name not in state:
                skipped += 1
                continue
            tensor = handle.get_tensor(name)
            target = state[name]
            if tuple(tensor.shape) != tuple(target.shape):
                print(
                    f"shape mismatch, skipping {name}: "
                    f"{tuple(tensor.shape)} vs {tuple(target.shape)}"
                )
                skipped += 1
                del tensor
                continue
            with torch.no_grad():
                target.copy_(tensor)
            del tensor
            loaded += 1
    print(f"loaded {loaded} tensors, skipped {skipped}")


def _install_capture(model, buffers, torch) -> None:
    """Monkeypatch the 43 blocks to stream routing data into ``buffers``.

    Each wrapper casts the captured tensor to its storage dtype and copies it to
    its host buffer immediately, so GPU steady-state is one layer rather than
    all 43.

    Installed once, before the slice loop. The wrappers close over the ``buffers``
    dict object, so each slice retargets capture by replacing that dict's values
    -- re-wrapping per slice would nest the wrappers and mismatch token counts.
    """

    def wrap_block(block, layer):
        original = block.forward

        def forward(x, start_pos, input_ids, *args):
            # x: [1, S, hc_mult, dim] -- expanded residual (Hyper-Connections).
            buffers["expanded"][:, layer].copy_(x[0].to(torch.bfloat16))
            return original(x, start_pos, input_ids, *args)

        return forward

    def wrap_moe(block, layer):
        original = block.ffn.forward

        def forward(x, input_ids):
            # x: [1, S, dim] -- exact router input (post-ffn_norm residual).
            flat = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            buffers["compressed"][:, layer].copy_(flat.to(torch.bfloat16))
            return original(x, input_ids)

        return forward

    def wrap_gate(block, layer):
        original = block.ffn.gate.forward

        def forward(x, input_ids=None):
            weights, indices = original(x, input_ids)
            buffers["topk_weights"][:, layer].copy_(
                weights.detach().to(torch.float16)
            )
            buffers["topk_ids"][:, layer].copy_(indices.detach().to(torch.uint8))
            return weights, indices

        return forward

    for layer, block in enumerate(model.layers):
        block.forward = wrap_block(block, layer)
        block.ffn.forward = wrap_moe(block, layer)
        block.ffn.gate.forward = wrap_gate(block, layer)


def _harvest_forward(model, input_ids, torch):
    """Prefill the whole sequence and stop after the last block.

    Deliberately omits ``hc_head``, the LM head, sampling and the DSpark/MTP
    draft blocks, so nothing is decoded.
    """
    with torch.inference_mode():
        h = model.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        n_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            h = layer(h, 0, input_ids)
            print(f"  layer {i + 1}/{n_layers}", flush=True)
    return h


def _token_labels(text: str, offsets):
    """Label tokens as decode (1) or prefill (0), with turn/phase positions.

    A token is decode if it lies in an assistant turn: from after the
    ``<｜Assistant｜>`` marker (skipping an optional leading ``<think>`` or
    ``</think>`` generation marker) through the next
    ``<｜end▁of▁sentence｜>`` inclusive. Everything else is prefill.

    ``offsets`` are the tokenizer's ``(char_start, char_end)`` pairs. Returns
    ``(is_decode, turn_index, token_idx, phase_index)`` where a *phase* is a
    maximal contiguous run of equal ``is_decode``: ``token_idx`` is the position
    within the current phase (0-based, resets at every phase boundary) and
    ``phase_index`` is the global phase ordinal.
    """
    ranges = []
    turns = []
    i = 0
    while True:
        start = text.find(MARK_ASSISTANT, i)
        if start < 0:
            break
        body = start + len(MARK_ASSISTANT)
        if text.startswith(MARK_THINK_OPEN, body):
            body += len(MARK_THINK_OPEN)
        elif text.startswith(MARK_THINK_CLOSE, body):
            body += len(MARK_THINK_CLOSE)
        end = text.find(MARK_END, body)
        end = end + len(MARK_END) if end >= 0 else len(text)
        ranges.append((body, end))
        turns.append(start)
        i = end

    is_decode = [0] * len(offsets)
    turn_index = [0] * len(offsets)
    ti = 0
    for k, (char_start, _) in enumerate(offsets):
        while ti < len(turns) and turns[ti] <= char_start:
            ti += 1
        turn_index[k] = ti
        for range_start, range_end in ranges:
            if range_start <= char_start < range_end:
                is_decode[k] = 1
                break

    token_idx = [0] * len(offsets)
    phase_index = [0] * len(offsets)
    phase = -1
    idx = 0
    prev = None
    for k, dec in enumerate(is_decode):
        if dec != prev:
            phase += 1
            idx = 0
            prev = dec
        token_idx[k] = idx
        phase_index[k] = phase
        idx += 1
    return is_decode, turn_index, token_idx, phase_index


def _quantize_fp8(tensor, torch):
    """Scale a bf16 tensor into e4m3 and return (fp8_tensor, scale)."""
    absmax = max(tensor.max().item(), -tensor.min().item())
    scale = max(absmax / 448.0, 1e-8)
    tensor.div_(scale)
    return tensor.to(torch.float8_e4m3fn), scale


def _read_sessions(encoded_dir: str) -> dict:
    """List every encoded prompt, grouped by dataset_id."""
    sessions = {}
    for dataset in DATASETS:
        directory = os.path.join(encoded_dir, dataset)
        if not os.path.isdir(directory):
            print(f"warning: no encoded prompts for {dataset} ({directory})")
            sessions[dataset] = []
            continue
        keys = sorted(f[: -len(".txt")] for f in os.listdir(directory) if f.endswith(".txt"))
        sessions[dataset] = [
            {
                "dataset_id": dataset,
                "document_id": key,
                "path": os.path.join(directory, f"{key}.txt"),
            }
            for key in keys
        ]
        print(f"{dataset}: {len(keys)} encoded sessions")
    return sessions


def _select_slices(sessions: dict, tokenizer, rng: random.Random) -> dict:
    """Choose 4 train + 2 test sessions per dataset, preferring long ones.

    Each entry is ``(session, split, n_tokens)`` with ``n_tokens`` the prefix
    length actually harvested.
    """
    plan = {}
    for dataset, items in sessions.items():
        for item in items:
            with open(item["path"], encoding="utf-8") as handle:
                item["text"] = handle.read()
            item["n"] = len(
                tokenizer(item["text"], add_special_tokens=False)["input_ids"]
            )

        train_pool = [s for s in items if s["n"] >= TRAIN_TOKENS]
        rng.shuffle(train_pool)
        train = train_pool[:TRAIN_SESSIONS]
        if len(train) < TRAIN_SESSIONS:
            rest = sorted(
                (s for s in items if s not in train), key=lambda s: -s["n"]
            )
            train += rest[: TRAIN_SESSIONS - len(train)]

        chosen = {s["document_id"] for s in train}
        test_pool = [
            s for s in items if s["document_id"] not in chosen and s["n"] >= TEST_TOKENS
        ]
        rng.shuffle(test_pool)
        test = test_pool[:TEST_SESSIONS]
        if len(test) < TEST_SESSIONS:
            rest = sorted(
                (s for s in items if s["document_id"] not in chosen and s not in test),
                key=lambda s: -s["n"],
            )
            test += rest[: TEST_SESSIONS - len(test)]

        slices = [(s, "train", min(TRAIN_TOKENS, s["n"])) for s in train]
        slices += [(s, "test", min(TEST_TOKENS, s["n"])) for s in test]
        plan[dataset] = slices
        picked = ", ".join(
            f"{s['document_id']}({tag},{n})" for s, tag, n in slices
        )
        print(f"{dataset}: {picked}")
    return plan


@app.function(
    volumes={HF_CACHE_DIR: hf_cache_vol, CKPT_DIR: ckpt_vol},
    secrets=[hf_secret],
    cpu=16,
    memory=200 * 1024,
    timeout=6 * 60 * 60,
)
def convert_ckpt(force: bool = False) -> str:
    """Convert the HF checkpoint into DeepSeek's MP-reference format."""
    import os
    import subprocess
    import sys

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
    import json
    import os
    import random
    import sys
    import time

    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=BASE_REPO, revision=BASE_REV, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"{BASE_REPO} @ {BASE_REV} is not cached in the mounted volume; "
            "run `modal run download_weights.py` first."
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
    buffers = {}
    _install_capture(model, buffers, torch)
    records = []
    errors = []

    for dataset, slices in plan.items():
        for session, split, n_tokens in slices:
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
                    "topk_ids_dtype": "uint8",
                    "topk_weights_dtype": "float16",
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
