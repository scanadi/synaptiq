"""Role-aware resource limits for the embedded engine.

Synaptiq runs in two very different modes with opposite resource
contracts:

- **interactive** — one-shot CLI commands (``analyze``, ``query``, ...)
  where the user is waiting in the foreground.  LadybugDB keeps library
  defaults (all cores, default buffer pool); ONNX embedding threads
  default to a polite ``max(2, cores - 2)`` cap so a foreground index
  doesn't lock up the machine.
- **server** — long-running ``serve``/``mcp``/``watch`` daemons that sit
  beside the user's real work and must stay polite under concurrent
  agent query load.  Strict caps on the LadybugDB task-scheduler pool, the
  buffer pool, ONNX embedding threads, and the walk/parse worker pool (all
  capped to ``max(2, cores//4)`` rather than the interactive
  ``min(8, cores)``).

Entry points declare their role once via :func:`set_profile` before any
storage or embedding model is created; :class:`LadybugBackend` and the
embedders read :func:`current_limits` at creation time.

``synaptiq analyze --jobs N`` layers an explicit, per-invocation cap on
top of the active profile via :func:`set_jobs`, called before storage or
embedder creation just like :func:`set_profile`: ``N > 0`` caps engine
threads, ONNX embedding threads, and the walk/parse worker pools to
*N*; ``N = 0`` is an explicit escape hatch back to uncapped
(all-cores/library-default) engine and embedding threads, overriding even
the environment variables below.  Precedence is **flag > environment
variables > profile defaults**.

Environment overrides (positive integers, applied on top of either
profile, unless overridden by ``--jobs``):

- ``SYNAPTIQ_DB_THREADS`` — LadybugDB task-scheduler pool size.
- ``SYNAPTIQ_DB_MEMORY_MB`` — LadybugDB buffer pool size in MB.
- ``SYNAPTIQ_EMBED_THREADS`` — ONNX intra-op threads for fastembed.

``SYNAPTIQ_KUZU_THREADS`` / ``SYNAPTIQ_KUZU_MEMORY_MB`` remain accepted as
deprecated aliases for one release (they log a one-time warning); the
``DB`` names win when both are set.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MB = 1024 * 1024

# Buffer pool cap for the server profile.  LadybugDB's library default is a
# large fraction of system RAM — fine for a foreground bulk load, not
# for a sidecar daemon.
SERVER_BUFFER_POOL_MB = 512

# Historical hardcoded thread-pool size for the walker/parser phases
# (walker.py, parser_phase.py).  Now a ceiling rather than a fixed
# value: the resolved pool size is ``min(_DEFAULT_POOL_WORKERS,
# cpu_count)`` so small machines don't over-subscribe.
_DEFAULT_POOL_WORKERS = 8

_VALID_PROFILES = ("interactive", "server")


@dataclass(frozen=True)
class ResourceLimits:
    """Engine resource caps.

    ``0`` means library default for ``db_threads``, ``db_buffer_bytes``,
    and ``embed_threads``.  ``pool_workers`` has no such "library default"
    concept — there's no engine-level auto mode for the walk/parse thread
    pools — so it always resolves to a concrete positive worker count
    (``min(8, cores)`` interactive; capped to ``max(2, cores//4)`` under the
    server profile so a sidecar daemon's parsing stays polite — review F3).
    """

    db_threads: int = 0
    db_buffer_bytes: int = 0
    embed_threads: int = 0
    pool_workers: int = _DEFAULT_POOL_WORKERS


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


# Deprecated env-var names already warned about, so the warning fires once per
# process regardless of how often ``resolve_limits`` / ``current_limits`` runs.
_warned_env_aliases: set[str] = set()


def _env_int_aliased(new_name: str, old_name: str) -> int | None:
    """Read *new_name*, falling back to the deprecated *old_name*.

    The ``SYNAPTIQ_KUZU_*`` variables predate the KuzuDB→LadybugDB swap (W2.7);
    they keep working for one release as aliases for the ``SYNAPTIQ_DB_*``
    names but emit a one-time deprecation warning when they are the value that
    takes effect. The new name wins when both are set.
    """
    value = _env_int(new_name)
    if value is not None:
        return value
    legacy = _env_int(old_name)
    if legacy is not None and old_name not in _warned_env_aliases:
        _warned_env_aliases.add(old_name)
        logger.warning(
            "%s is deprecated and will be removed in a future release; use %s instead.",
            old_name,
            new_name,
        )
    return legacy


def resolve_limits(
    profile: str, cpu_count: int | None = None, jobs: int | None = None
) -> ResourceLimits:
    """Resolve the limits for *profile*, applying environment and ``--jobs`` overrides.

    Args:
        profile: ``"interactive"`` or ``"server"``.
        cpu_count: Core count to scale against; defaults to the machine's.
        jobs: Explicit worker/thread cap, e.g. from ``analyze --jobs`` (see
            :func:`set_jobs`).  ``None`` (the default) applies no override —
            environment variables, then profile defaults, resolve as usual.
            ``0`` is an explicit request for uncapped engine/embedding threads
            (the ``--jobs 0`` escape hatch), taking precedence over
            environment variables too.  ``N > 0`` caps ``db_threads``,
            ``embed_threads``, and ``pool_workers`` to *N*.  Never affects
            ``db_buffer_bytes`` (a memory cap, not a worker/thread count).
    """
    cpus = cpu_count or os.cpu_count() or 4
    pool_default = min(_DEFAULT_POOL_WORKERS, cpus)

    if profile == "server":
        threads = max(2, cpus // 4)
        limits = ResourceLimits(
            db_threads=threads,
            db_buffer_bytes=SERVER_BUFFER_POOL_MB * _MB,
            embed_threads=threads,
            # A sidecar daemon must stay polite under concurrent agent load, so
            # the walk/parse worker pool is capped to the same max(2, cores//4)
            # as the engine/embedding threads — not the interactive
            # min(8, cores) default (review F3).
            pool_workers=threads,
        )
    else:
        # interactive: LadybugDB keeps the library default (0); embedding
        # threads get a polite cap so a foreground `analyze` doesn't peg
        # every core, unless overridden by an env var or --jobs below.
        limits = ResourceLimits(
            embed_threads=max(2, cpus - 2),
            pool_workers=pool_default,
        )

    db_threads = _env_int_aliased("SYNAPTIQ_DB_THREADS", "SYNAPTIQ_KUZU_THREADS")
    buffer_mb = _env_int_aliased("SYNAPTIQ_DB_MEMORY_MB", "SYNAPTIQ_KUZU_MEMORY_MB")
    embed_threads = _env_int("SYNAPTIQ_EMBED_THREADS")
    if db_threads or buffer_mb or embed_threads:
        limits = ResourceLimits(
            db_threads=db_threads or limits.db_threads,
            db_buffer_bytes=buffer_mb * _MB if buffer_mb else limits.db_buffer_bytes,
            embed_threads=embed_threads or limits.embed_threads,
            pool_workers=limits.pool_workers,
        )

    if jobs is not None:
        if jobs > 0:
            # Explicit cap: flag wins over both env vars and profile
            # defaults for thread/worker counts (not buffer memory).
            limits = ResourceLimits(
                db_threads=jobs,
                db_buffer_bytes=limits.db_buffer_bytes,
                embed_threads=jobs,
                pool_workers=jobs,
            )
        else:
            # `--jobs 0`: explicit escape hatch back to uncapped
            # engine/embedding threads, overriding even the polite
            # interactive default and any SYNAPTIQ_* env vars.
            limits = ResourceLimits(
                db_threads=0,
                db_buffer_bytes=limits.db_buffer_bytes,
                embed_threads=0,
                pool_workers=pool_default,
            )

    return limits


_active_profile = "interactive"
_jobs_override: int | None = None


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


def set_jobs(jobs: int | None) -> None:
    """Set (or clear) a process-wide ``--jobs`` override on top of the profile.

    Intended for one-shot commands like ``analyze --jobs`` that need a
    per-invocation cap without changing the process's role.  Call once at
    the entry point, before any storage backend or embedding model is
    created — for the same reason as :func:`set_profile`, limits are read
    at creation time, not re-applied to live objects.

    Args:
        jobs: ``None`` clears the override (env vars, then profile
            defaults, resolve as usual).  ``0`` is an explicit request for
            uncapped engine/embedding threads.  ``N > 0`` caps engine threads,
            embedding threads, and I/O worker pools to *N*.
    """
    global _jobs_override  # noqa: PLW0603
    if jobs is not None and jobs < 0:
        raise ValueError(f"jobs must be >= 0, got {jobs}")
    _jobs_override = jobs
    if jobs is not None:
        logger.info(
            "Jobs override set to %d: %s", jobs, resolve_limits(_active_profile, jobs=jobs)
        )


def current_limits() -> ResourceLimits:
    """Return the limits for the active profile (env + ``--jobs`` overrides applied)."""
    return resolve_limits(_active_profile, jobs=_jobs_override)
