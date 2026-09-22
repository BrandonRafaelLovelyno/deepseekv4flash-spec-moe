"""Pure-Python/torch helpers for the expert-routing harvest.

Nothing here imports ``modal`` and nothing here imports ``torch`` at module
scope: torch is passed in (or imported inside a function) so this module stays
importable without the CUDA stack. The parameter objects passed between these
functions are declared as ``TypedDict``/``dataclass`` above the code that uses
them, so each signature documents its structure instead of an opaque ``dict``.
"""

import os
import random
from dataclasses import dataclass
from typing import Any, TypedDict

# Four encoded DeepSeek-V4-Flash corpora, as written by ``2_data_check``.
DATASETS = ("yi30-think", "yi30-nothink", "terminus2", "dsh")

TRAIN_SESSIONS = 4
TRAIN_TOKENS = 24000
TEST_SESSIONS = 2
TEST_TOKENS = 5000

# Region markers in the encoded prompt (fullwidth vertical bars, U+FF5C).
# ``<think>``/``</think>`` directly after ``<｜Assistant｜>`` is generation
# scaffolding (thinking vs chat mode), not model output, so it is not decode.
MARK_ASSISTANT = "<｜Assistant｜>"
MARK_END = "<｜end▁of▁sentence｜>"
MARK_THINK_OPEN = "<think>"
MARK_THINK_CLOSE = "</think>"

# Maps capture-buffer name (``expanded`` / ``compressed`` / ``topk_ids`` /
# ``topk_weights``) to its host tensor. The dict object is created once and its
# values are replaced per slice, which is what lets the monkeypatched wrappers
# retarget capture without being reinstalled.
CaptureBuffers = dict[str, Any]

# Tokenizer offset pairs: ``(char_start, char_end)`` per token.
Offsets = list[tuple[int, int]]


class Session(TypedDict, total=False):
    """One encoded session (prompt file) selected for harvesting.

    ``_read_sessions`` fills ``dataset_id`` / ``document_id`` / ``path``;
    ``_select_slices`` then adds ``text`` (the decoded prompt) and ``n`` (its
    token length).
    """

    dataset_id: str
    document_id: str
    path: str
    text: str
    n: int


@dataclass(frozen=True)
class Slice:
    """One concrete prefix to harvest.

    Attributes:
        session: The source session (with ``text`` and ``n`` populated).
        split: ``"train"`` or ``"test"``.
        n_tokens: Prefix length actually harvested from the session.
    """

    session: Session
    split: str
    n_tokens: int


def _load_streaming(model: Any, path: str) -> None:
    """Copy a safetensors checkpoint into ``model`` one tensor at a time.

    ``safetensors.torch.load_model`` would materialize the whole 167 GB file in
    host RAM before the copy. Streaming sends each tensor straight to the GPU and
    writes it into the matching parameter storage.

    Args:
        model: The instantiated ``Transformer`` whose ``state_dict`` is the
            copy target.
        path: Path to the converted ``.safetensors`` checkpoint.
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


def _install_capture(model: Any, buffers: CaptureBuffers, torch: Any) -> None:
    """Monkeypatch the 43 blocks to stream routing data into ``buffers``.

    Each wrapper casts the captured tensor to its storage dtype and copies it to
    its host buffer immediately, so GPU steady-state is one layer rather than
    all 43.

    Installed once, before the slice loop. The wrappers close over the ``buffers``
    dict object, so each slice retargets capture by replacing that dict's values
    -- re-wrapping per slice would nest the wrappers and mismatch token counts.

    Args:
        model: The loaded ``Transformer``.
        buffers: Capture-buffer mapping; values are (re)assigned by the caller
            per slice.
        torch: The torch module (passed in to avoid a module-scope import).
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


def _harvest_forward(model: Any, input_ids: Any, torch: Any) -> Any:
    """Prefill the whole sequence and stop after the last block.

    Deliberately omits ``hc_head``, the LM head, sampling and the DSpark/MTP
    draft blocks, so nothing is decoded.

    Args:
        model: The loaded ``Transformer`` with capture wrappers installed.
        input_ids: ``[1, S]`` long tensor of prompt token ids.
        torch: The torch module.

    Returns:
        The final block's hidden state (unused; capture happens via buffers).
    """
    with torch.inference_mode():
        h = model.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        n_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            h = layer(h, 0, input_ids)
            print(f"  layer {i + 1}/{n_layers}", flush=True)
    return h


def _token_labels(
    text: str, offsets: Offsets
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Label tokens as decode (1) or prefill (0), with turn/phase positions.

    A token is decode if it lies in an assistant turn: from after the
    ``<｜Assistant｜>`` marker (skipping an optional leading ``<think>`` or
    ``</think>`` generation marker) through the next
    ``<｜end▁of▁sentence｜>`` inclusive. Everything else is prefill.

    Args:
        text: The decoded prompt string.
        offsets: Tokenizer offset pairs ``(char_start, char_end)``; only
            ``char_start`` is used for labelling.

    Returns:
        ``(is_decode, turn_index, token_idx, phase_index)`` where a *phase* is a
        maximal contiguous run of equal ``is_decode``: ``token_idx`` is the
        position within the current phase (0-based, resets at every phase
        boundary) and ``phase_index`` is the global phase ordinal.
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


def _quantize_fp8(tensor: Any, torch: Any) -> tuple[Any, float]:
    """Scale a bf16 tensor into e4m3 and return (fp8_tensor, scale).

    Mutates ``tensor`` in place (divides by the scale) before casting.

    Args:
        tensor: A bf16 tensor to quantize.
        torch: The torch module.

    Returns:
        ``(fp8_tensor, scale)`` where ``scale`` is the per-tensor absmax/448
        factor recorded in the sidecar.
    """
    absmax = max(tensor.max().item(), -tensor.min().item())
    scale = max(absmax / 448.0, 1e-8)
    tensor.div_(scale)
    return tensor.to(torch.float8_e4m3fn), scale


def _read_sessions(encoded_dir: str) -> dict[str, list[Session]]:
    """List every encoded prompt, grouped by dataset_id.

    Args:
        encoded_dir: Volume directory holding ``<dataset>/<key>.txt`` prompts.

    Returns:
        Maps each dataset name to its sessions, sorted by document id.
    """
    sessions: dict[str, list[Session]] = {}
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


def _select_slices(
    sessions: dict[str, list[Session]], tokenizer: Any, rng: random.Random
) -> dict[str, list[Slice]]:
    """Choose 4 train + 2 test sessions per dataset, preferring long ones.

    Reads every session's text, tokenizes it to fill ``Session.n``, then picks
    the longest-first fallbacks if there are not enough candidates above the
    token thresholds.

    Args:
        sessions: Sessions grouped by dataset, as returned by ``_read_sessions``
            (mutated in place: each session gets ``text`` and ``n``).
        tokenizer: A fast tokenizer used only for length measurement.
        rng: Seeded RNG so selection is reproducible.

    Returns:
        Maps each dataset name to its ``Slice`` plan (train then test).
    """
    plan: dict[str, list[Slice]] = {}
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

        slices = [Slice(s, "train", min(TRAIN_TOKENS, s["n"])) for s in train]
        slices += [Slice(s, "test", min(TEST_TOKENS, s["n"])) for s in test]
        plan[dataset] = slices
        picked = ", ".join(
            f"{sl.session['document_id']}({sl.split},{sl.n_tokens})" for sl in slices
        )
        print(f"{dataset}: {picked}")
    return plan
