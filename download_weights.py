"""Download and cache DeepSeek-V4-Flash NVFP4 weights into a Modal Volume.

This script only downloads weights -- it never requests a GPU. The HF cache
lives in the Modal Volume ``huggingface-cache`` so later serving apps can mount
it instead of re-downloading from the Hub.

Prerequisites:
    modal setup
    modal secret create huggingface-secret HF_TOKEN=hf_...

Usage:
    modal run download_weights.py

Note: the download runs inside a remote function, not an image ``run_function``
build step. Modal materializes images lazily, so build steps on an app that only
has a local entrypoint never execute. A real function guarantees the volume is
mounted, the download runs, and the cache is committed.
"""

import modal

APP_NAME = "deepseek-v4-flash-download"

HF_CACHE_DIR = "/root/.cache/huggingface"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODELS = {
    "nvidia/DeepSeek-V4-Flash-0731-NVFP4": "f1caa71142bd0be02f728c79f75042ac1e461579",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]")
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={HF_CACHE_DIR: hf_cache_vol},
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
    cpu=8,
)
def download_models() -> list[str]:
    from huggingface_hub import snapshot_download

    paths = []
    for repo_id, revision in MODELS.items():
        path = snapshot_download(repo_id=repo_id, revision=revision, max_workers=16)
        paths.append(path)
    hf_cache_vol.commit()
    return paths


@app.local_entrypoint()
def main() -> None:
    for path in download_models.remote():
        print(f"cached: {path}")
    print("Weights cached in Modal Volume 'huggingface-cache'.")
