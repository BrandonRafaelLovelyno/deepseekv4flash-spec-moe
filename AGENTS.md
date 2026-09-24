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

## Local checkpoint reference (`checkpoints/`)

`checkpoints/` is a local, gitignored copy of reference material shipped with the
checkpoint — **the ground truth for prompt format and model architecture**. Consult
it directly rather than reverse-engineering behavior from the vLLM fork.

- `checkpoints/encoding/` — DeepSeek's self-contained prompt encoder and format spec.
  `encoding_dsv4.py` exposes `encode_messages` / `parse_message_from_completion_text`;
  its `README.md` documents special tokens, chat vs. thinking mode, DSML tool calling,
  and reasoning-effort prefixes. `2_data_check/` encodes trajectories with this module.
- `checkpoints/inference/` — DeepSeek's reference inference path: `convert.py` (HF weights
  → MP-reference format), `generate.py` (+ `model.py`, `kernel.py`), and `config.json`
  (expert count, MP, `expert_dtype` fp4/fp8). Authoritative for architecture, MoE routing,
  and expert layout when implementing speculative loading.

Because `checkpoints/` is gitignored, these files are not in the repo — check the
directory locally before assuming a design detail.
