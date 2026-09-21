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

Pipeline (``modal run data_check/data_check.py``):
    1. ``download_sources`` (CPU, idempotent) caches both full HF repos under the
       ``deepseek-v4-flash-datasets`` volume.
    2. ``encode_all`` (CPU) walks every trajectory, encodes it, and writes
       ``<source>/<key>.txt`` plus ``manifest.json`` to the
       ``deepseek-v4-flash-encoded`` volume.
    3. The local entrypoint writes the manifest to
       ``data_check/output/manifest.json`` and prints a summary.

Inspect an individual prompt with:
    modal volume get deepseek-v4-flash-encoded <source>/<key>.txt

Prerequisites:
    modal secret create huggingface-secret HF_TOKEN=hf_...
    modal run download_weights.py          # caches the checkpoint in the volume

Usage:
    modal run data_check/data_check.py
"""

import json
import os
import statistics

import modal

APP_NAME = "deepseek-v4-flash-data-check"

BASE_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

YI30_REPO = "Yi30/deepseek-v4-swebench-trajectories"
TB_REPO = "openguardrails/terminal-bench-2.1-deepseek-v4-flash-trajectories"

DATASET_DIR = "/datasets"
ENCODED_DIR = "/encoded"
CKPT_HF_DIR = "/root/.cache/huggingface"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "output")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

SOURCES = {
    "yi30-think": {"repo": YI30_REPO, "adapter": "yi30", "thinking_mode": "thinking"},
    "yi30-nothink": {"repo": YI30_REPO, "adapter": "yi30", "thinking_mode": "chat"},
    "terminus2": {"repo": TB_REPO, "adapter": "terminus2", "thinking_mode": "thinking"},
    "dsh": {"repo": TB_REPO, "adapter": "dsh", "thinking_mode": "chat"},
}

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
)

app = modal.App(APP_NAME, image=image)


def _join_text(blocks) -> str:
    """Join a message's ``text`` content blocks, or pass a string through."""
    if isinstance(blocks, str):
        return blocks
    parts = []
    for block in blocks or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _attach_tools(messages: list[dict], tools: list) -> list[dict]:
    """Attach OpenAI-format tool schemas to the system message.

    ``render_message`` only emits the tools block when a message carries a
    ``tools`` field, so without this the schemas would be dropped. Falls back to
    a fresh system message if the trajectory has none.
    """
    if not tools:
        return messages
    for msg in messages:
        if msg.get("role") == "system":
            msg["tools"] = tools
            return messages
    messages.insert(0, {"role": "system", "content": "", "tools": tools})
    return messages


def _from_yi30(data: dict) -> tuple[list[dict], list]:
    """Yi30 deepseek-v4-swebench trajectories: OpenAI-format messages.

    Assistant turns carry ``reasoning_content``; the trailing ``exit`` row is
    dropped. Tool calls are kept as-is (their ``arguments`` is a JSON string,
    which the encoder parses itself).
    """
    messages = []
    for msg in data["messages"]:
        role = msg.get("role")
        if role == "exit":
            continue
        out = {"role": role, "content": msg.get("content") or ""}
        if role == "assistant":
            reasoning = msg.get("reasoning_content")
            if reasoning:
                out["reasoning_content"] = reasoning
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                out["tool_calls"] = [
                    {
                        "id": call.get("id"),
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"].get("arguments", "{}"),
                        },
                    }
                    for call in tool_calls
                ]
        elif role == "tool":
            out["tool_call_id"] = msg.get("tool_call_id")
        messages.append(out)
    return messages, []


def _from_terminus2(data: dict) -> tuple[list[dict], list]:
    """Terminal-Bench ATIF trajectory: one ``steps[]`` entry per model turn.

    Agent steps carry ``reasoning_content`` and ``tool_calls`` (``bash_command``
    with ``{keystrokes, duration}``); ``observation.results`` are the tool
    results. No tool schemas are available for this scaffold, so none are
    attached.
    """
    messages = []
    for step in data["steps"]:
        source = step.get("source")
        if source == "agent":
            out = {"role": "assistant", "content": step.get("message") or ""}
            reasoning = step.get("reasoning_content")
            if reasoning:
                out["reasoning_content"] = reasoning
            tool_calls = step.get("tool_calls") or []
            if tool_calls:
                out["tool_calls"] = [
                    {
                        "id": call.get("tool_call_id"),
                        "function": {
                            "name": call.get("function_name"),
                            "arguments": call.get("arguments", {}),
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(out)
            observation = step.get("observation")
            if isinstance(observation, dict):
                for result in observation.get("results") or []:
                    messages.append(
                        {
                            "role": "tool",
                            "content": result.get("content", ""),
                            "tool_call_id": result.get("source_call_id"),
                        }
                    )
        else:
            role = "system" if source == "system" else "user"
            messages.append({"role": role, "content": step.get("message") or ""})
    return messages, []


def _from_dsh(lines) -> tuple[list[dict], list, dict]:
    """DeepSeek Harness session log (JSONL event stream).

    ``dsh`` does not persist the model's reasoning, so assistant turns carry no
    ``reasoning``; it does ship the real tool schemas in ``request/header``.
    Returns ``(messages, tools, config)``.
    """
    messages = []
    tools = []
    config = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        kind = event.get("type")
        data = event.get("data", {})
        if kind == "request/header":
            header = data.get("header", {})
            config = header.get("config", {}) or {}
            if not tools:
                tools = [
                    {"type": "function", "function": tool}
                    for tool in header.get("tools") or []
                ]
        elif kind == "system/message":
            content = data.get("message", {}).get("content")
            messages.append({"role": "system", "content": _join_text(content)})
        elif kind == "user/message":
            messages.append({"role": "user", "content": _join_text(data.get("content"))})
        elif kind == "assistant/message":
            blocks = data.get("message", {}).get("content") or []
            out = {"role": "assistant", "content": _join_text(blocks)}
            tool_calls = [
                block
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "tool-call"
            ]
            if tool_calls:
                out["tool_calls"] = [
                    {
                        "id": call.get("id"),
                        "function": {
                            "name": call.get("name"),
                            "arguments": call.get("arguments", "{}"),
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(out)
        elif kind == "tool/result":
            content = data.get("message", {}).get("content") or []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool-result":
                    messages.append(
                        {
                            "role": "tool",
                            "content": _join_text(block.get("content")),
                            "tool_call_id": block.get("toolCallId"),
                        }
                    )
    return messages, tools, config


def _load_trajectory(adapter: str, text: str) -> tuple[list[dict], list, dict]:
    """Dispatch raw file text to its adapter, returning (messages, tools, config)."""
    if adapter == "dsh":
        return _from_dsh(text.splitlines())
    data = json.loads(text)
    if adapter == "yi30":
        messages, tools = _from_yi30(data)
    elif adapter == "terminus2":
        messages, tools = _from_terminus2(data)
    else:
        raise ValueError(f"unknown adapter: {adapter}")
    return messages, tools, {}


def _enumerate_items(snapshots: dict) -> list[dict]:
    """List every trajectory to encode as {source, key, path, adapter, mode}."""
    import glob

    items = []
    for mode, source in (("think_high", "yi30-think"), ("no_think", "yi30-nothink")):
        pattern = os.path.join(snapshots[YI30_REPO], "data", mode, "*.traj.json")
        for path in sorted(glob.glob(pattern)):
            key = os.path.basename(path)[: -len(".traj.json")]
            items.append({"source": source, "key": key, "path": path})

    for source, sub in (("terminus2", "trajectory.json"), ("dsh", "dsh-session.jsonl")):
        pattern = os.path.join(
            snapshots[TB_REPO], "trajectories", source, "*", "agent", sub
        )
        for path in sorted(glob.glob(pattern)):
            key = os.path.basename(os.path.dirname(os.path.dirname(path)))
            items.append({"source": source, "key": key, "path": path})

    for item in items:
        item["adapter"] = SOURCES[item["source"]]["adapter"]
        item["thinking_mode"] = SOURCES[item["source"]]["thinking_mode"]
    return items


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
    import sys
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
            "run `modal run download_weights.py` first."
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

    results = []
    errors = []
    for item in items:
        source, key = item["source"], item["key"]
        try:
            with open(item["path"], encoding="utf-8") as handle:
                text = handle.read()
            messages, tools, _ = _load_trajectory(item["adapter"], text)
            if tools:
                messages = _attach_tools(messages, tools)
            encoded = encode_messages(
                messages, thinking_mode=item["thinking_mode"], drop_thinking=False
            )

            out_dir = os.path.join(ENCODED_DIR, source)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{key}.txt"), "w", encoding="utf-8") as handle:
                handle.write(encoded)

            n_reasoning_chars = sum(
                len(msg.get("reasoning_content") or "") for msg in messages
            )
            results.append(
                {
                    "source": source,
                    "key": key,
                    "thinking_mode": item["thinking_mode"],
                    "n_messages": len(messages),
                    "n_tools": len(tools),
                    "n_reasoning_chars": n_reasoning_chars,
                    "n_chars": len(encoded),
                }
            )
            print(
                f"  {source}/{key}: {len(messages)} msgs, {len(encoded)} chars "
                f"({n_reasoning_chars} reasoning)"
            )
        except Exception as exc:  # keep going; report the bad row in the manifest
            errors.append({"source": source, "key": key, "error": repr(exc)})
            print(f"  FAILED {source}/{key}: {exc!r}")

    by_source = {}
    for source in SOURCES:
        rows = [r for r in results if r["source"] == source]
        if not rows:
            by_source[source] = {"count": 0}
            continue
        chars = [r["n_chars"] for r in rows]
        by_source[source] = {
            "count": len(rows),
            "min_chars": min(chars),
            "median_chars": int(statistics.median(chars)),
            "max_chars": max(chars),
            "total_reasoning_chars": sum(r["n_reasoning_chars"] for r in rows),
            "total_chars": sum(chars),
        }

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_items": len(results),
        "n_errors": len(errors),
        "sources": by_source,
        "items": results,
        "errors": errors,
    }
    with open(os.path.join(ENCODED_DIR, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    encoded_vol.commit()

    print(json.dumps(by_source, indent=2))
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
