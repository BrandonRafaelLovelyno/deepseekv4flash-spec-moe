"""Run DeepSeek-V4-Flash (MXFP4 base) inference on a single B300 -- no engine.

This is a pure-PyTorch forward pass. It uses DeepSeek's own reference
implementation shipped inside the checkpoint (``inference/model.py`` and
``inference/kernel.py``); ``inference/generate.py`` already wraps generation in
``torch.inference_mode()``. No vLLM, no SGLang.

The reference code only understands the MXFP4 base release
(``deepseek-ai/DeepSeek-V4-Flash-0731``), whose routed experts use one E8M0
scale per 32 FP4 values. It cannot read the NVIDIA NVFP4 quantization (one E4M3
scale per 16 values plus ``weight_scale_2``/``input_scale``), which is why this
runner downloads and converts the MXFP4 base instead.

Pipeline (``modal run run_inference.py``):
    1. ``convert_ckpt`` (CPU, idempotent) turns the HF shards into the
       reference ``model0-mp1.safetensors`` + tokenizer under the ``/ckpt``
       volume. Cached, so later runs skip it.
    2. ``run_inference`` (single B300) builds the model, streams the converted
       weights to the GPU, encodes an agentic prompt in thinking mode and
       prints the completion.

Prerequisites:
    modal secret create huggingface-secret HF_TOKEN=hf_...
    modal run download_weights.py          # caches the MXFP4 base in the volume
    modal run run_inference.py --prompt "..."
"""

import modal

APP_NAME = "deepseek-v4-flash-infer"

BASE_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

HF_CACHE_DIR = "/root/.cache/huggingface"
CKPT_DIR = "/ckpt"
TILELANG_CACHE_DIR = "/root/.cache/tilelang"

CKPT_FILE = "model0-mp1.safetensors"
N_EXPERTS = 256
MODEL_PARALLEL = 1

GPU = "B300"
MAX_SEQ_LEN = 8192

SYSTEM_PROMPT = (
    "You are an autonomous software engineering agent operating in a Linux "
    "workspace. You can inspect files and run shell commands with the provided "
    "tools. Think step by step about the task, then either call a tool or give "
    "the final answer."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command in the workspace and return its "
                "stdout/stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

DEFAULT_PROMPT = (
    "Plan and write a bash command sequence that finds the largest .safetensors "
    "file under /ckpt, prints how many tensors its header declares, and reports "
    "whether any tensor uses the F4 dtype. Assume python3 and safetensors are "
    "installed. Output only the commands plus a one-line explanation."
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("deepseek-v4-flash-ckpt", create_if_missing=True)
tilelang_cache_vol = modal.Volume.from_name("tilelang-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

# CUDA *devel* base: TileLang JIT and fast_hadamard_transform both need nvcc at
# build/runtime. B300 (sm_103) needs the CUDA 13 family.
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
        # tilelang's bundled TVM targets an older tvm_ffi API; its own bound is
        # loose (>=0.1.2,<0.2.0), so pin the era-matched build explicitly.
        "apache-tvm-ffi==0.1.8.post2",
        "tqdm",
        "setuptools",
        "wheel",
        "ninja",
        "packaging",
    )
    # PyPI sdist for fast-hadamard-transform ships no csrc/ and its wheel URL
    # guesses cu122 for any CUDA>=12, so it always 404s. Build from the git tag
    # instead; force g++ (torch's ABI) and skip the bogus wheel fetch.
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
    host RAM before the copy. Streaming avoids that: each tensor goes straight
    to the GPU and is written into the matching parameter storage.
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
    },
    secrets=[hf_secret],
    cpu=8,
    memory=64 * 1024,
    timeout=2 * 60 * 60,
)
def run_inference(
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 512,
    thinking: bool = True,
) -> str:
    """Run one agentic prompt through the model and return the completion."""
    import json
    import os
    import sys

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
    from transformers import AutoTokenizer

    from encoding_dsv4 import encode_messages
    from generate import generate
    from model import ModelArgs, Transformer

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    torch.cuda.set_device(0)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    with open(os.path.join(snapshot, "inference", "config.json")) as handle:
        args = ModelArgs(**json.load(handle))
    args.max_batch_size = 1
    args.max_seq_len = MAX_SEQ_LEN
    print(args)

    with torch.device("cuda"):
        model = Transformer(args)

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    _load_streaming(model, os.path.join(CKPT_DIR, CKPT_FILE))
    torch.set_default_device("cuda")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT, "tools": TOOLS},
        {"role": "user", "content": prompt},
    ]
    thinking_mode = "thinking" if thinking else "chat"
    prompt_ids = tokenizer.encode(
        encode_messages(
            messages,
            thinking_mode=thinking_mode,
            reasoning_effort="high" if thinking else None,
        )
    )

    completion_ids = generate(
        model, [prompt_ids], max_new_tokens, tokenizer.eos_token_id
    )
    text = tokenizer.decode(completion_ids[0])
    print(text)
    return text


@app.local_entrypoint()
def main(
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 512,
    thinking: bool = True,
    skip_convert: bool = False,
) -> None:
    if not skip_convert:
        print(f"checkpoint: {convert_ckpt.remote()}")
    text = run_inference.remote(prompt, max_new_tokens, thinking)
    print("\n===== completion =====")
    print(text)
