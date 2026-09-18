# AGENTS.md

## Project

Serve **DeepSeek-V4-Flash-0731** (NVFP4 weights) on [Modal](https://modal.com),
targeting a **single NVIDIA B200 (180 GB VRAM)**.

The serving engine is a **fork of vLLM** vendored as a git submodule at `vllm/`.
This repository is the orchestration/serving layer around that fork.

## Core novelty: speculative expert loading

DeepSeek-V4 is a huge MoE; the full expert set does not fit in 180 GB. The
project's central idea is **speculative expert loading**: predict which experts a
request will route to and stage them into VRAM ahead of the forward pass, so the
resident expert working set (and therefore peak VRAM) is compressed while
transfer latency stays hidden.
