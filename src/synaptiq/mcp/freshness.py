"""Compact index-freshness trailer for MCP tool/resource responses (W4.5).

Distills two cheap file reads — ``meta.json`` and ``embeddings_state.json``
— into a single bracketed line, e.g.::

    [index: 4m old · embeddings: complete]
    [index: 12s old · embeddings: encoding 12431/26203]
    [index: 2h old · embeddings: failed]
    [index: age unknown]

so an agent can see at a glance whether the graph it just queried might be
stale, without a separate round trip to ``synaptiq status``.

Appended centrally in ``mcp/server.py`` (:func:`~synaptiq.mcp.server._apply_response_pipeline`
for every tool, and ``dispatch_resource`` for ``synaptiq://overview`` only)
— see those call sites for why that is the single place all response text
flows through, including proxy mode.

Design constraints:
  * Cheap — two small file reads, cached for a few seconds so a burst of
    tool calls (or a chatty agent) doesn't turn into an I/O storm.
  * Never raises — any error (missing file, corrupt JSON, permission
    denied, ...) degrades to a partial trailer or an empty string, never
    an exception. A tool response must never fail because of this.
  * Omitted entirely when there is no index at all (``meta.json`` missing)
    so the existing "no index" error messages are untouched.
  * Disabled process-wide via ``SYNAPTIQ_MCP_FRESHNESS=0``.

Deliberately reads the ``embeddings_state.json`` file format directly
(documented in ``core/embeddings/lazy_worker.py``) rather than importing
its reader — this module's only contract with the embeddings worker is the
on-disk JSON shape, not its Python API.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

# Environment escape hatch: any value other than "0" (including unset)
# leaves the trailer enabled.
_ENV_DISABLE = "SYNAPTIQ_MCP_FRESHNESS"

_STATE_FILENAME = "embeddings_state.json"
_META_FILENAME = "meta.json"

# How long a computed trailer stays valid before the next call re-reads the
# files. Keyed per data_dir so tests (and, in principle, multiple served
# repos in one process) never share a stale entry.
_CACHE_TTL_SECONDS = 5.0
_cache: dict[str, tuple[float, str]] = {}

# Indirection so tests can advance time without a real sleep.
_monotonic = time.monotonic


def _format_age(seconds: float) -> str:
    """Render a non-negative duration as the coarsest readable unit."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s old"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes}m old"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h old"
    days = hours // 24
    return f"{days}d old"


def _index_age_fragment(meta: dict) -> str:
    """``4m old`` from ``meta['last_indexed_at']``, or ``age unknown``."""
    raw = meta.get("last_indexed_at")
    if not raw or not isinstance(raw, str):
        return "age unknown"
    try:
        indexed_at = datetime.fromisoformat(raw)
        now = datetime.now(indexed_at.tzinfo) if indexed_at.tzinfo else datetime.now()  # noqa: DTZ005
        delta_seconds = (now - indexed_at).total_seconds()
    except (ValueError, TypeError):
        return "age unknown"
    return _format_age(delta_seconds)


def _read_json(path: Path) -> dict | None:
    """Parse *path* as JSON, or ``None`` on any read/parse failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _embeddings_fragment(data_dir: Path, meta: dict) -> str | None:
    """``complete`` / ``encoding N/M`` / ``failed`` / ``deferred``.

    Prefers the live worker state file; falls back to the vector count
    stashed in ``meta.json`` by the last full index when the state file is
    absent or unreadable (pre-W4.1 index, or a worker that never ran).
    Returns ``None`` when there is nothing meaningful to report.
    """
    state = _read_json(data_dir / _STATE_FILENAME)
    if state is not None:
        kind = state.get("state")
        if kind == "encoding":
            done = state.get("done", 0)
            total = state.get("total", 0)
            return f"encoding {done}/{total}"
        if kind in ("complete", "failed", "deferred"):
            return kind

    stats = meta.get("stats")
    count = stats.get("embeddings") if isinstance(stats, dict) else None
    if count:
        return f"{count}"
    return None


def _compute(data_dir: Path) -> str:
    """Build the trailer for *data_dir*, or ``""`` when there is no index."""
    meta_path = data_dir / _META_FILENAME
    if not meta_path.exists():
        return ""

    meta = _read_json(meta_path) or {}

    parts = [f"index: {_index_age_fragment(meta)}"]
    embeddings = _embeddings_fragment(data_dir, meta)
    if embeddings:
        parts.append(f"embeddings: {embeddings}")

    return "[" + " · ".join(parts) + "]"


def freshness_trailer(data_dir: Path | None = None) -> str:
    """Return a compact freshness line for *data_dir*, or ``""`` to omit it.

    Args:
        data_dir: Directory containing ``meta.json`` / ``embeddings_state.json``
            (i.e. a repo's ``.synaptiq`` directory). Defaults to
            ``Path.cwd() / ".synaptiq"`` — the same convention every other
            cwd-relative lookup in this codebase uses (``serve`` deliberately
            ``chdir``s to the target repo so this is always correct, even
            under the primary/proxy daemon).

    Never raises: any failure anywhere in the computation is swallowed and
    an empty (or partial) trailer is returned instead.
    """
    try:
        if os.environ.get(_ENV_DISABLE, "1") == "0":
            return ""

        if data_dir is None:
            data_dir = Path.cwd() / ".synaptiq"

        key = str(data_dir)
        now = _monotonic()
        cached = _cache.get(key)
        if cached is not None and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]

        trailer = _compute(data_dir)
        _cache[key] = (now, trailer)
        return trailer
    except Exception:
        return ""
