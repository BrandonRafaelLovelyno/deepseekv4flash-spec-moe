"""Pure-Python helpers for the DeepSeek-V4-Flash data check.

Nothing in this module imports ``modal``: every function is a plain, CPU-only
transformation over trajectories and their encoded prompts, so the Modal
wrappers in ``main.py`` can stay thin. The parameter objects passed between
these functions are declared as ``TypedDict``/``dataclass`` above the code that
uses them, so each signature documents the exact structure it expects instead
of an opaque ``dict``.
"""

import glob
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, TypedDict

YI30_REPO = "Yi30/deepseek-v4-swebench-trajectories"
TB_REPO = "openguardrails/terminal-bench-2.1-deepseek-v4-flash-trajectories"

# One entry per corpus source. ``repo`` is the HuggingFace dataset id, ``adapter``
# selects the raw-trajectory parser in ``_load_trajectory``, and ``thinking_mode``
# is the mode handed to ``encode_messages`` for every row of that source.
SOURCES: dict[str, dict[str, str]] = {
    "yi30-think": {"repo": YI30_REPO, "adapter": "yi30", "thinking_mode": "thinking"},
    "yi30-nothink": {"repo": YI30_REPO, "adapter": "yi30", "thinking_mode": "chat"},
    "terminus2": {"repo": TB_REPO, "adapter": "terminus2", "thinking_mode": "thinking"},
    "dsh": {"repo": TB_REPO, "adapter": "dsh", "thinking_mode": "chat"},
}


class ContentBlock(TypedDict, total=False):
    """A single content block inside a ``dsh`` message.

    Fields depend on ``type``: ``"text"`` uses ``text``, ``"tool-call"`` uses
    ``id`` / ``name`` / ``arguments``, and ``"tool-result"`` uses
    ``toolCallId`` / ``content``.
    """

    type: str
    text: str
    content: Any
    id: str
    name: str
    arguments: str
    toolCallId: str


class ToolSchema(TypedDict):
    """An OpenAI-format tool schema attached to the system message.

    ``function`` is the raw schema object (``{name, description, parameters}``)
    as shipped by the trajectory source.
    """

    type: str
    function: dict[str, Any]


class ChatMessage(TypedDict, total=False):
    """An OpenAI-format chat message handed to ``encode_messages``.

    Only ``role`` and ``content`` are always present. Assistant turns add
    ``reasoning_content`` and ``tool_calls``; tool results add ``tool_call_id``;
    a system message may carry the attached ``tools`` schemas.
    """

    role: str
    content: str
    reasoning_content: str
    tool_calls: list[dict[str, Any]]
    tool_call_id: str
    tools: list[ToolSchema]


class Yi30Message(TypedDict, total=False):
    """One message from a ``Yi30/deepseek-v4-swebench-trajectories`` file."""

    role: str
    content: Any
    reasoning_content: str
    tool_calls: list[dict[str, Any]]
    tool_call_id: str


class Yi30Trajectory(TypedDict):
    """Raw JSON payload of a ``*.traj.json`` Yi30 trajectory."""

    messages: list[Yi30Message]


class TerminusToolCall(TypedDict, total=False):
    """One tool call recorded in a Terminal-Bench ATIF agent step."""

    tool_call_id: str
    function_name: str
    arguments: Any


class TerminusResult(TypedDict, total=False):
    """One observation result produced by a Terminal-Bench agent step."""

    content: str
    source_call_id: str


class TerminusObservation(TypedDict, total=False):
    """Observation attached to an ATIF agent step; holds its tool results."""

    results: list[TerminusResult]


class TerminusStep(TypedDict, total=False):
    """One entry of a Terminal-Bench ATIF ``steps[]`` array.

    ``agent`` steps carry ``message`` / ``reasoning_content`` / ``tool_calls`` /
    ``observation``; other steps carry only ``source`` and ``message``.
    """

    source: str
    message: str
    reasoning_content: str
    tool_calls: list[TerminusToolCall]
    observation: TerminusObservation


class TerminusTrajectory(TypedDict):
    """Raw JSON payload of a Terminal-Bench ``trajectory.json`` file."""

    steps: list[TerminusStep]


# ``dsh``'s request/header carries the harness config verbatim; its keys are
# harness-defined and not part of this pipeline's contract.
DshConfig = dict[str, Any]

# Maps a HuggingFace repo id to the local snapshot directory it was downloaded to.
SnapshotPaths = Mapping[str, str]


@dataclass(frozen=True)
class TrajectoryItem:
    """One trajectory selected for encoding.

    Attributes:
        source: Key into ``SOURCES`` (e.g. ``"yi30-think"``).
        key: Stable per-source identifier, used as the output file stem.
        path: Absolute path to the raw trajectory file on disk.
        adapter: Parser selector from ``SOURCES`` (``yi30`` / ``terminus2`` /
            ``dsh``).
        thinking_mode: ``encode_messages`` mode from ``SOURCES``.
    """

    source: str
    key: str
    path: str
    adapter: str
    thinking_mode: str


@dataclass
class EncodeResult:
    """Per-trajectory encoding stats written into the manifest."""

    source: str
    key: str
    thinking_mode: str
    n_messages: int
    n_tools: int
    n_reasoning_chars: int
    n_chars: int


class ErrorRow(TypedDict):
    """A trajectory that failed to encode; kept in the manifest errors list."""

    source: str
    key: str
    error: str


class SourceStats(TypedDict, total=False):
    """Aggregate character/message stats for one ``source``."""

    count: int
    min_chars: int
    median_chars: int
    max_chars: int
    total_reasoning_chars: int
    total_chars: int


def _join_text(blocks: str | list[ContentBlock] | None) -> str:
    """Join a message's ``text`` content blocks, or pass a string through.

    Args:
        blocks: The message content. A plain string is returned unchanged; a
            list of content blocks keeps only ``{"type": "text"}`` blocks and
            joins their ``text`` fields with newlines. ``None`` yields ``""``.
    """
    if isinstance(blocks, str):
        return blocks
    parts = []
    for block in blocks or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _attach_tools(
    messages: list[ChatMessage], tools: list[ToolSchema]
) -> list[ChatMessage]:
    """Attach OpenAI-format tool schemas to the system message.

    ``render_message`` only emits the tools block when a message carries a
    ``tools`` field, so without this the schemas would be dropped. Falls back to
    a fresh system message if the trajectory has none.

    Args:
        messages: Chat messages in OpenAI format; mutated in place.
        tools: Tool schemas to attach; when empty, ``messages`` is returned
            unchanged.

    Returns:
        The same ``messages`` list, with the schemas attached to its system
        message.
    """
    if not tools:
        return messages
    for msg in messages:
        if msg.get("role") == "system":
            msg["tools"] = tools
            return messages
    messages.insert(0, {"role": "system", "content": "", "tools": tools})
    return messages


def _from_yi30(data: Yi30Trajectory) -> tuple[list[ChatMessage], list[ToolSchema]]:
    """Yi30 deepseek-v4-swebench trajectories: OpenAI-format messages.

    Assistant turns carry ``reasoning_content``; the trailing ``exit`` row is
    dropped. Tool calls are kept as-is (their ``arguments`` is a JSON string,
    which the encoder parses itself).

    Args:
        data: Decoded ``*.traj.json`` payload.

    Returns:
        ``(messages, tools)``; the Yi30 format ships no tool schemas, so
        ``tools`` is always empty.
    """
    messages: list[ChatMessage] = []
    for msg in data["messages"]:
        role = msg.get("role")
        if role == "exit":
            continue
        out: ChatMessage = {"role": role, "content": msg.get("content") or ""}
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


def _from_terminus2(
    data: TerminusTrajectory,
) -> tuple[list[ChatMessage], list[ToolSchema]]:
    """Terminal-Bench ATIF trajectory: one ``steps[]`` entry per model turn.

    Agent steps carry ``reasoning_content`` and ``tool_calls`` (``bash_command``
    with ``{keystrokes, duration}``); ``observation.results`` are the tool
    results. No tool schemas are available for this scaffold, so none are
    attached.

    Args:
        data: Decoded ``trajectory.json`` payload.

    Returns:
        ``(messages, tools)``; this scaffold ships no tool schemas, so ``tools``
        is always empty.
    """
    messages: list[ChatMessage] = []
    for step in data["steps"]:
        source = step.get("source")
        if source == "agent":
            out: ChatMessage = {
                "role": "assistant",
                "content": step.get("message") or "",
            }
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


def _from_dsh(
    lines: Iterable[str],
) -> tuple[list[ChatMessage], list[ToolSchema], DshConfig]:
    """DeepSeek Harness session log (JSONL event stream).

    ``dsh`` does not persist the model's reasoning, so assistant turns carry no
    ``reasoning_content``; it does ship the real tool schemas in
    ``request/header``.

    Args:
        lines: The raw ``dsh-session.jsonl`` lines; blank lines are skipped.

    Returns:
        ``(messages, tools, config)`` where ``config`` is the harness config
        from the first ``request/header`` event.
    """
    messages: list[ChatMessage] = []
    tools: list[ToolSchema] = []
    config: DshConfig = {}
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
            out: ChatMessage = {"role": "assistant", "content": _join_text(blocks)}
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


def _load_trajectory(
    adapter: str, text: str
) -> tuple[list[ChatMessage], list[ToolSchema], DshConfig]:
    """Dispatch raw file text to its adapter.

    Args:
        adapter: One of ``"yi30"``, ``"terminus2"`` or ``"dsh"`` (from
            ``TrajectoryItem.adapter``).
        text: Full raw contents of the trajectory file. ``dsh`` is parsed as a
            JSONL stream, the others as a single JSON document.

    Returns:
        ``(messages, tools, config)``; ``config`` is only populated by the
        ``dsh`` adapter.

    Raises:
        ValueError: If ``adapter`` is not one of the known parsers.
    """
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


def _enumerate_items(snapshots: SnapshotPaths) -> list[TrajectoryItem]:
    """List every trajectory to encode.

    Args:
        snapshots: Maps each HuggingFace dataset repo id to the local snapshot
            directory it was downloaded into (the return value of
            ``snapshot_download``).

    Returns:
        One ``TrajectoryItem`` per discovered trajectory file, with ``adapter``
        and ``thinking_mode`` filled in from ``SOURCES``.
    """
    items: list[TrajectoryItem] = []
    for mode, source in (("think_high", "yi30-think"), ("no_think", "yi30-nothink")):
        pattern = os.path.join(snapshots[YI30_REPO], "data", mode, "*.traj.json")
        for path in sorted(glob.glob(pattern)):
            key = os.path.basename(path)[: -len(".traj.json")]
            items.append(_make_item(source, key, path))

    for source, sub in (("terminus2", "trajectory.json"), ("dsh", "dsh-session.jsonl")):
        pattern = os.path.join(
            snapshots[TB_REPO], "trajectories", source, "*", "agent", sub
        )
        for path in sorted(glob.glob(pattern)):
            key = os.path.basename(os.path.dirname(os.path.dirname(path)))
            items.append(_make_item(source, key, path))

    return items


def _make_item(source: str, key: str, path: str) -> TrajectoryItem:
    """Build a ``TrajectoryItem``, pulling adapter/mode from ``SOURCES``."""
    spec = SOURCES[source]
    return TrajectoryItem(
        source=source,
        key=key,
        path=path,
        adapter=spec["adapter"],
        thinking_mode=spec["thinking_mode"],
    )


def _encode_item(
    item: TrajectoryItem,
    encode_messages: Callable[..., str],
    encoded_dir: str,
) -> EncodeResult:
    """Read, adapt, encode and persist one trajectory.

    Args:
        item: The trajectory to encode.
        encode_messages: The checkpoint's encoder, invoked as
            ``encode_messages(messages, thinking_mode=..., drop_thinking=False)``.
        encoded_dir: Volume directory the prompt is written under as
            ``<encoded_dir>/<source>/<key>.txt``.

    Returns:
        The ``EncodeResult`` stats for the manifest.
    """
    with open(item.path, encoding="utf-8") as handle:
        text = handle.read()
    messages, tools, _ = _load_trajectory(item.adapter, text)
    if tools:
        messages = _attach_tools(messages, tools)
    encoded = encode_messages(
        messages, thinking_mode=item.thinking_mode, drop_thinking=False
    )

    out_dir = os.path.join(encoded_dir, item.source)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{item.key}.txt"), "w", encoding="utf-8") as handle:
        handle.write(encoded)

    return EncodeResult(
        source=item.source,
        key=item.key,
        thinking_mode=item.thinking_mode,
        n_messages=len(messages),
        n_tools=len(tools),
        n_reasoning_chars=sum(
            len(msg.get("reasoning_content") or "") for msg in messages
        ),
        n_chars=len(encoded),
    )


def _summarize_results(results: list[EncodeResult]) -> dict[str, SourceStats]:
    """Aggregate per-trajectory stats into one row per ``SOURCES`` entry.

    Sources with no successful trajectories get a ``{"count": 0}`` row so the
    summary keys always match ``SOURCES``.

    Args:
        results: Successfully encoded trajectories.

    Returns:
        Maps each source name to its ``SourceStats``.
    """
    by_source: dict[str, SourceStats] = {}
    for source in SOURCES:
        rows = [r for r in results if r.source == source]
        if not rows:
            by_source[source] = {"count": 0}
            continue
        chars = [r.n_chars for r in rows]
        by_source[source] = {
            "count": len(rows),
            "min_chars": min(chars),
            "median_chars": int(statistics.median(chars)),
            "max_chars": max(chars),
            "total_reasoning_chars": sum(r.n_reasoning_chars for r in rows),
            "total_chars": sum(chars),
        }
    return by_source
