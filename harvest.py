"""Harvest per-token expert-routing data from DeepSeek-V4-Flash (MXFP4 base).

Runs a **single prefill pass** over the first ``N_TOKENS`` tokens of an opencode
request reconstruction (``opencode_request_ses_f455.txt``, dumped from
``opencode.db``) through DeepSeek's own reference implementation
(``inference/model.py``) and records, for every token in every one of the 43
blocks:

    * the expanded residual state -- the Hyper-Connections block input,
      shape ``[hc_mult=4, dim=4096]``;
    * the exact MoE router input -- the post-``ffn_norm`` compressed residual,
      shape ``[dim=4096]``;
    * the routed expert ids and their top-k weights, shape ``[top_k=6]``.

No decoding is performed: the harvest forward stops after the last block and
never touches the LM head, the sampler, or the DSpark/MTP draft blocks. The
reference attention only supports whole-sequence prefill (``start_pos=0``) or
one-token decode, so the pass is a single forward -- which is also what keeps
full causal context (and therefore realistic routing).

Captured tensors are cast to fp16 and streamed to host memory as each layer
completes, then written to a Modal Volume as one safetensors file plus a JSON
sidecar.

Pipeline (``modal run harvest.py``):
    1. ``convert_ckpt`` (CPU, idempotent) reuses the converted MXFP4 checkpoint
       under the ``/ckpt`` volume (same artifact ``run_inference.py`` produces).
    2. ``harvest`` (single B300) loads the model, encodes the reconstruction as
       a single user turn in thinking mode, prefills the first ``N_TOKENS``
       tokens and writes the captured routing data.

Prerequisites:
    modal secret create huggingface-secret HF_TOKEN=hf_...
    modal run download_weights.py          # caches the MXFP4 base in the volume
"""

import json
import os
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

CKPT_FILE = "model0-mp1.safetensors"
N_EXPERTS = 256
MODEL_PARALLEL = 1

GPU = "B300"
MAX_SEQ_LEN = 16384
N_TOKENS = 10_000

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "opencode_request_ses_f455"
DATA_FILE = "opencode_request_ses_f455.txt"
DATA_REMOTE = f"/data/{DATA_FILE}"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("deepseek-v4-flash-ckpt", create_if_missing=True)
tilelang_cache_vol = modal.Volume.from_name("tilelang-cache", create_if_missing=True)
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
    .add_local_file(os.path.join(THIS_DIR, DATA_FILE), DATA_REMOTE)
)

app = modal.App(APP_NAME, image=image)


def _load_streaming(model, path: str) -> None:
    """Copy a safetensors checkpoint into ``model`` one tensor at a time.

    ``safetensors.torch.load_model`` would materialize the whole file in host
    RAM before the copy. Streaming sends each tensor straight to the GPU and
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


def _install_capture(model, buffers, torch, progress) -> None:
    """Monkeypatch the 43 blocks to stream routing data into ``buffers``.

    Each wrapper casts the captured tensor to fp16 and copies it to its host
    buffer immediately, so GPU steady-state is one layer rather than all 43.
    The block wrapper also advances the progress bar once the layer is done.
    """
    n_layers = len(model.layers)

    def wrap_block(block, layer):
        original = block.forward

        def forward(x, start_pos, input_ids, *args):
            # x: [1, S, hc_mult, dim] -- expanded residual (Hyper-Connections).
            buffers["expanded"][:, layer].copy_(x[0].to(torch.float16))
            out = original(x, start_pos, input_ids, *args)
            progress.layer_done(layer, n_layers, x.shape[1])
            return out

        return forward

    def wrap_moe(block, layer):
        original = block.ffn.forward

        def forward(x, input_ids):
            # x: [1, S, dim] -- exact router input (post-ffn_norm residual).
            flat = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            buffers["compressed"][:, layer].copy_(flat.to(torch.float16))
            return original(x, input_ids)

        return forward

    def wrap_gate(block, layer):
        original = block.ffn.gate.forward

        def forward(x, input_ids=None):
            weights, indices = original(x, input_ids)
            buffers["topk_weights"][:, layer].copy_(
                weights.detach().to(torch.float32)
            )
            buffers["topk_ids"][:, layer].copy_(indices.detach().to(torch.int32))
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
        for layer in model.layers:
            h = layer(h, 0, input_ids)
    return h


class _Progress:
    """Token-unit progress bar that also advances intra-forward, per layer."""

    def __init__(self, total: int, n_layers: int):
        from tqdm import tqdm

        self.total = total
        self.n_layers = n_layers
        self.t0 = time.monotonic()
        self.bar = tqdm(
            total=total,
            unit="tok",
            desc="harvest",
            file=sys.stdout,
            mininterval=0.5,
            dynamic_ncols=True,
        )

    def layer_done(self, layer: int, n_layers: int, segment: int) -> None:
        self.bar.update(segment / n_layers)
        elapsed = time.monotonic() - self.t0
        done = int(segment * (layer + 1) / n_layers)
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        print(
            f"layer {layer + 1:>2}/{n_layers}  "
            f"tokens {done:>6}/{self.total}  "
            f"elapsed {elapsed:6.1f}s  eta {eta:6.1f}s"
        )

    def close(self) -> None:
        self.bar.n = self.total
        self.bar.refresh()
        self.bar.close()


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
        HARVEST_DIR: harvest_vol,
    },
    secrets=[hf_secret],
    cpu=8,
    memory=64 * 1024,
    timeout=2 * 60 * 60,
)
def harvest(n_tokens: int = N_TOKENS, max_seq_len: int = MAX_SEQ_LEN) -> dict:
    """Prefill the session and write per-token routing data to the volume."""
    import json
    import os
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

    from encoding_dsv4 import encode_messages
    from model import ModelArgs, Transformer

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    torch.cuda.set_device(0)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    with open(DATA_REMOTE) as handle:
        session_text = handle.read()

    messages = [{"role": "user", "content": session_text}]
    encoded = encode_messages(
        messages, thinking_mode="thinking", reasoning_effort="high"
    )
    ids = tokenizer.encode(encoded)
    if len(ids) < n_tokens:
        print(f"session only has {len(ids)} tokens; using all of them")
        n_tokens = len(ids)
    ids = ids[:n_tokens]
    n_tokens = len(ids)
    print(f"harvesting {n_tokens} tokens")

    with open(os.path.join(snapshot, "inference", "config.json")) as handle:
        args = ModelArgs(**json.load(handle))
    args.max_batch_size = 1
    args.max_seq_len = max(max_seq_len, n_tokens)
    print(args)

    with torch.device("cuda"):
        model = Transformer(args)

    _load_streaming(model, os.path.join(CKPT_DIR, CKPT_FILE))
    torch.set_default_device("cuda")

    n_layers = len(model.layers)
    buffers = {
        "expanded": torch.empty(
            (n_tokens, n_layers, args.hc_mult, args.dim),
            dtype=torch.float16,
            device="cpu",
        ),
        "compressed": torch.empty(
            (n_tokens, n_layers, args.dim),
            dtype=torch.float16,
            device="cpu",
        ),
        "topk_ids": torch.empty(
            (n_tokens, n_layers, args.n_activated_experts),
            dtype=torch.int32,
            device="cpu",
        ),
        "topk_weights": torch.empty(
            (n_tokens, n_layers, args.n_activated_experts),
            dtype=torch.float32,
            device="cpu",
        ),
    }
    payload_gb = sum(t.numel() * t.element_size() for t in buffers.values()) / 1e9
    print(f"allocated {payload_gb:.2f} GB of host buffers")

    progress = _Progress(n_tokens, n_layers)
    _install_capture(model, buffers, torch, progress)

    input_ids = torch.tensor([ids], dtype=torch.long)
    t0 = time.monotonic()
    try:
        _harvest_forward(model, input_ids, torch)
    finally:
        progress.close()
        torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    out_path = os.path.join(HARVEST_DIR, f"{DATASET_NAME}_{n_tokens}.safetensors")
    tensors = {
        "expanded": buffers["expanded"],
        "compressed": buffers["compressed"],
        "topk_ids": buffers["topk_ids"],
        "topk_weights": buffers["topk_weights"],
        "token_ids": torch.tensor(ids, dtype=torch.int32, device="cpu"),
    }
    print(f"writing {payload_gb:.2f} GB to {out_path}")
    save_file(tensors, out_path)

    meta = {
        "dataset": DATASET_NAME,
        "source_file": DATA_FILE,
        "model_repo": BASE_REPO,
        "model_revision": BASE_REV,
        "encoding": "encode_messages(thinking, high)",
        "serialization": "float16",
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "dim": args.dim,
        "hc_mult": args.hc_mult,
        "top_k": args.n_activated_experts,
        "n_routed_experts": args.n_routed_experts,
        "max_seq_len": args.max_seq_len,
        "elapsed_seconds": round(elapsed, 2),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = out_path.replace(".safetensors", ".json")
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    harvest_vol.commit()

    summary = {
        "path": out_path,
        "meta_path": meta_path,
        "payload_gb": round(payload_gb, 3),
        **meta,
    }
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(
    n_tokens: int = N_TOKENS,
    max_seq_len: int = MAX_SEQ_LEN,
    skip_convert: bool = False,
) -> None:
    if not skip_convert:
        print(f"checkpoint: {convert_ckpt.remote()}")
    summary = harvest.remote(n_tokens, max_seq_len)
    print("\n===== harvest summary =====")
    print(json.dumps(summary, indent=2))
