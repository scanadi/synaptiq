"""Role-aware resource limits for the embedded engine.

Synaptiq runs in two very different modes with opposite resource
contracts:

- **interactive** — one-shot CLI commands (``analyze``, ``query``, ...)
  where the user is waiting in the foreground.  Full machine resources
  (library defaults: all cores, Kuzu's default buffer pool).
- **server** — long-running ``serve``/``mcp``/``watch`` daemons that sit
  beside the user's real work and must stay polite under concurrent
  agent query load.  Strict caps on the Kuzu task-scheduler pool, the
  buffer pool, and ONNX embedding threads.

Entry points declare their role once via :func:`set_profile` before any
storage or embedding model is created; :class:`KuzuBackend` and the
embedders read :func:`current_limits` at creation time.

Environment overrides (positive integers, applied on top of either
profile):

- ``SYNAPTIQ_KUZU_THREADS`` — Kuzu task-scheduler pool size.
- ``SYNAPTIQ_KUZU_MEMORY_MB`` — Kuzu buffer pool size in MB.
- ``SYNAPTIQ_EMBED_THREADS`` — ONNX intra-op threads for fastembed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MB = 1024 * 1024

# Buffer pool cap for the server profile.  Kuzu's library default is a
# large fraction of system RAM — fine for a foreground bulk load, not
# for a sidecar daemon.
SERVER_BUFFER_POOL_MB = 512

_VALID_PROFILES = ("interactive", "server")


@dataclass(frozen=True)
class ResourceLimits:
    """Engine resource caps.  ``0`` for any field means library default."""

    kuzu_threads: int = 0
    kuzu_buffer_bytes: int = 0
    embed_threads: int = 0


def _env_int(name: str) -> int | None:
    """Read a positive integer from the environment, or ``None``."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, raw)
        return None
    return value if value > 0 else None


def resolve_limits(profile: str, cpu_count: int | None = None) -> ResourceLimits:
    """Resolve the limits for *profile*, applying environment overrides.

    Args:
        profile: ``"interactive"`` or ``"server"``.
        cpu_count: Core count to scale against; defaults to the machine's.
    """
    cpus = cpu_count or os.cpu_count() or 4

    if profile == "server":
        threads = max(2, cpus // 4)
        limits = ResourceLimits(
            kuzu_threads=threads,
            kuzu_buffer_bytes=SERVER_BUFFER_POOL_MB * _MB,
            embed_threads=threads,
        )
    else:
        limits = ResourceLimits()

    kuzu_threads = _env_int("SYNAPTIQ_KUZU_THREADS")
    buffer_mb = _env_int("SYNAPTIQ_KUZU_MEMORY_MB")
    embed_threads = _env_int("SYNAPTIQ_EMBED_THREADS")
    if kuzu_threads or buffer_mb or embed_threads:
        limits = ResourceLimits(
            kuzu_threads=kuzu_threads or limits.kuzu_threads,
            kuzu_buffer_bytes=buffer_mb * _MB if buffer_mb else limits.kuzu_buffer_bytes,
            embed_threads=embed_threads or limits.embed_threads,
        )
    return limits


_active_profile = "interactive"


def set_profile(profile: str) -> None:
    """Declare this process's role.

    Call once at the entry point, before any storage backend or
    embedding model is created — limits are read at creation time, not
    re-applied to live objects.
    """
    global _active_profile  # noqa: PLW0603
    if profile not in _VALID_PROFILES:
        raise ValueError(f"Unknown resource profile: {profile!r}")
    _active_profile = profile
    logger.info("Resource profile set to %s: %s", profile, resolve_limits(profile))


def current_limits() -> ResourceLimits:
    """Return the limits for the active profile (env overrides applied)."""
    return resolve_limits(_active_profile)
