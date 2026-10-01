"""Chunked-prefill token-stream construction and chunk diagnostics.

A serving step is modelled as a token budget ``B`` packed from two streams:

* the **decode stream** -- concurrent decode tokens from in-flight requests;
* the **prefill stream** -- the next consecutive tokens of a request being
  prefilled.

Because the harvest only holds a handful of documents, the decode stream is a
*dispersed* permutation of the pooled decode tokens rather than a naive
sequential walk: repeated visits to the same session are spread far apart in the
document so concurrent decode tokens do not share nearby context. Prefill stays
sequential per request, as real chunked prefill does.

numpy is imported inside functions so the module imports without the stack.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np


class SessionTokens(NamedTuple):
    """One harvested session's decode and prefill row offsets.

    ``decode`` is in generation order, ``prefill`` in token order; the values are
    global row offsets into the split's concatenated cache.
    """

    dataset_id: str
    document_id: str
    decode: np.ndarray
    prefill: np.ndarray


class ChunkPlan(NamedTuple):
    """One ``(split, decode_portion, chunk_size)``'s chunked token stream."""

    chunks: list[np.ndarray]
    n_decode: np.ndarray
    n_prefill: np.ndarray


def _radical_inverse(n: int, base: int = 2) -> np.ndarray:
    """The first ``n`` base-``base`` van der Corput values in ``[0, 1)``."""
    import numpy as np

    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    index = np.arange(n, dtype=np.int64)
    denominator = 1.0
    while bool((index > 0).any()):
        index, remainder = np.divmod(index, base)
        denominator *= base
        out += remainder / denominator
    return out


def van_der_corput_order(n: int) -> np.ndarray:
    """A permutation of ``0..n-1`` whose consecutive entries are widely spaced.

    Ordering positions by their van der Corput value makes each new pick land in
    the largest remaining gap, so two tokens taken from one session close in
    stream time are far apart in the document.
    """
    import numpy as np

    if n <= 0:
        return np.empty(0, dtype=np.int64)
    return np.argsort(_radical_inverse(n))


def build_session_tokens(
    slices: list[dict], labels: dict[str, np.ndarray]
) -> list[SessionTokens]:
    """Split each slice's cache rows into decode and prefill offsets."""
    import numpy as np

    sessions: list[SessionTokens] = []
    for entry in slices:
        is_decode = labels[entry["path"]]
        rows = entry["offset"] + np.arange(entry["n_tokens"], dtype=np.int64)
        sessions.append(
            SessionTokens(
                dataset_id=entry["dataset_id"],
                document_id=entry["document_id"],
                decode=rows[is_decode],
                prefill=rows[~is_decode],
            )
        )
    return sessions


def dispersed_decode_stream(sessions: list[SessionTokens], seed: int) -> np.ndarray:
    """Order all decode tokens by balanced session rotation + low-discrepancy picks.

    At each step the least-emitted session (random tie-break) yields its next
    position in van der Corput order. Consecutive tokens therefore rotate across
    sessions, and a session's successive tokens are ~half its length apart.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    streams = [session.decode for session in sessions if len(session.decode)]
    if not streams:
        return np.empty(0, dtype=np.int64)

    lengths = np.array([stream.shape[0] for stream in streams], dtype=np.int64)
    orders = [van_der_corput_order(int(length)) for length in lengths]
    emitted = np.zeros(len(streams), dtype=np.int64)
    total = int(lengths.sum())

    out = np.empty(total, dtype=np.int64)
    for write in range(total):
        active = np.flatnonzero(emitted < lengths)
        fewest = emitted[active].min()
        ties = active[emitted[active] == fewest]
        chosen = int(ties[rng.integers(ties.shape[0])])
        out[write] = streams[chosen][orders[chosen][emitted[chosen]]]
        emitted[chosen] += 1
    return out


def sequential_prefill_stream(sessions: list[SessionTokens], seed: int) -> np.ndarray:
    """Concatenate each session's prefill rows, sessions in a seeded order.

    So consecutive ``B - decode`` blocks are consecutive tokens of one request,
    which is what a real chunked prefill feeds.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sessions))
    parts = [
        sessions[int(i)].prefill
        for i in order
        if len(sessions[int(i)].prefill)
    ]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def iter_chunks(
    decode: np.ndarray,
    prefill: np.ndarray,
    chunk_size: int,
    decode_portion: float,
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield ``(chunk_rows, n_decode, n_prefill)`` for one replay.

    Each chunk targets ``round(decode_portion * chunk_size)`` decode tokens and
    the rest prefill; when one stream runs dry the remainder is filled from the
    other, and the final chunk may be short. Both streams are consumed exactly
    once.
    """
    import numpy as np

    decode_target = round(float(decode_portion) * chunk_size)
    prefill_target = chunk_size - decode_target
    n_decode_total = int(decode.shape[0])
    n_prefill_total = int(prefill.shape[0])
    d = 0
    p = 0
    while d < n_decode_total or p < n_prefill_total:
        take_decode = min(decode_target, n_decode_total - d)
        take_prefill = min(prefill_target, n_prefill_total - p)
        deficit = chunk_size - take_decode - take_prefill
        if deficit > 0:
            extra = min(deficit, n_decode_total - d - take_decode)
            take_decode += extra
            deficit -= extra
        if deficit > 0:
            extra = min(deficit, n_prefill_total - p - take_prefill)
            take_prefill += extra
        chunk = np.concatenate(
            (
                decode[d : d + take_decode],
                prefill[p : p + take_prefill],
            )
        )
        yield chunk, take_decode, take_prefill
        d += take_decode
        p += take_prefill


def chunk_diagnostics(
    plan: ChunkPlan, session_of_row: np.ndarray, is_decode_row: np.ndarray
) -> dict[str, float]:
    """Measure how dispersed the plan's chunks are.

    Reports the mean distinct sessions per chunk, the mean realized decode
    fraction, and the mean minimum within-session position gap among the decode
    tokens of a chunk (a large gap means repeated sessions are far apart).
    """
    import numpy as np

    n_chunks = len(plan.chunks)
    if n_chunks == 0:
        return {
            "chunks": 0,
            "distinct_sessions_per_chunk": 0.0,
            "realized_decode_portion": 0.0,
            "min_within_session_gap": 0.0,
        }

    total = sum(int(chunk.shape[0]) for chunk in plan.chunks)
    distinct = 0.0
    gaps: list[float] = []
    for chunk in plan.chunks:
        distinct += float(np.unique(session_of_row[chunk]).shape[0])
        decode_rows = chunk[is_decode_row[chunk]]
        if decode_rows.shape[0] < 2:
            continue
        by_session = session_of_row[decode_rows]
        chunk_min = None
        for session in np.unique(by_session):
            positions = np.sort(decode_rows[by_session == session])
            if positions.shape[0] < 2:
                continue
            gap = float(np.diff(positions).min())
            chunk_min = gap if chunk_min is None else min(chunk_min, gap)
        if chunk_min is not None:
            gaps.append(chunk_min)

    realized = (
        float(plan.n_decode.sum()) / total if total else 0.0
    )
    return {
        "chunks": n_chunks,
        "distinct_sessions_per_chunk": round(distinct / n_chunks, 4),
        "realized_decode_portion": round(realized, 4),
        "min_within_session_gap": round(float(np.mean(gaps)) if gaps else 0.0, 4),
    }
