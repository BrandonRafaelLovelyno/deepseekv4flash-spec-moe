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

## Code layout: keep top-level functions at one level of abstraction

Entry points (`main`, a Modal `@app.function`) should read as a short sequence
of named verbs, not a mix of orchestration and mechanics. Keep these rules when
touching the numbered stage directories (`2_data_check`, `5_train_one`, …):

- **No mechanics at the top level.** File IO (`open`/`json.dump`/`write`), plot
  layout (`subplot`/`set_xlabel`), and manual `for`-over-batches loops belong in
  their own module behind a single named call. A 20-line `open(...)` block in a
  function means it is at the wrong level.
- **One module per concern.** Split by role, not by layer: e.g. `reporting.py`
  (artifacts + prints), `plots.py` (figures returning `bytes`), `training.py`
  (loop/eval/checkpoint). Each function does one thing.
- **Unify duplicated logic.** If the same validation/setup appears in two places
  (e.g. task geometry in both `populate_cache` and the trainer), extract it to a
  single function in `helper.py` and call it from both.
- **Respect the import contract.** `helper.py` and anything the local
  entrypoint imports must not import `torch`/`numpy`/`yaml`/`matplotlib` at
  module scope — the local entrypoint runs without those stacks. Put heavy
  imports inside the function (or inside the module's functions) as in
  `training.py` / `plots.py`.
- **Register new modules on the image.** Any new local module must be added to
  `.add_local_python_source(...)` or it will be missing inside the container.
- **Verify without the GPU stack:** `python -m py_compile <files>` plus an
  import test asserting `torch` is absent from `sys.modules`; exercise pure
  helpers (CSV/report writes, plots) with a synthetic run.
