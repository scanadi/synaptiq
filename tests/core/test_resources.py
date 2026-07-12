"""Tests for role-aware resource limits."""

from __future__ import annotations

import pytest

from synaptiq.core import resources
from synaptiq.core.resources import (
    SERVER_BUFFER_POOL_MB,
    current_limits,
    resolve_limits,
    set_jobs,
    set_profile,
)

_MB = 1024 * 1024

ENV_VARS = (
    "SYNAPTIQ_DB_THREADS",
    "SYNAPTIQ_DB_MEMORY_MB",
    "SYNAPTIQ_KUZU_THREADS",
    "SYNAPTIQ_KUZU_MEMORY_MB",
    "SYNAPTIQ_EMBED_THREADS",
)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Strip env overrides and restore the interactive profile after each test."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    resources._warned_env_aliases.clear()
    yield
    resources._active_profile = "interactive"
    resources._jobs_override = None
    resources._warned_env_aliases.clear()


def test_interactive_profile_db_stays_library_default():
    limits = resolve_limits("interactive", cpu_count=16)
    assert limits.db_threads == 0
    assert limits.db_buffer_bytes == 0


def test_interactive_profile_embed_threads_polite_default():
    """Embed threads default to max(2, cores - 2) — the W1.4 polite default."""
    limits = resolve_limits("interactive", cpu_count=16)
    assert limits.embed_threads == 14


def test_interactive_profile_embed_default_floor_of_two():
    limits = resolve_limits("interactive", cpu_count=2)
    assert limits.embed_threads == 2  # max(2, 2 - 2) == max(2, 0) == 2

    limits = resolve_limits("interactive", cpu_count=1)
    assert limits.embed_threads == 2  # max(2, 1 - 2) == max(2, -1) == 2


def test_interactive_profile_pool_workers_default_caps_at_eight():
    limits = resolve_limits("interactive", cpu_count=16)
    assert limits.pool_workers == 8


def test_interactive_profile_pool_workers_capped_by_cpu_count():
    limits = resolve_limits("interactive", cpu_count=4)
    assert limits.pool_workers == 4


def test_server_profile_caps_scale_with_cores():
    limits = resolve_limits("server", cpu_count=16)
    assert limits.db_threads == 4
    assert limits.embed_threads == 4
    assert limits.db_buffer_bytes == SERVER_BUFFER_POOL_MB * _MB
    assert limits.pool_workers == 8


def test_server_profile_floor_of_two_threads():
    limits = resolve_limits("server", cpu_count=2)
    assert limits.db_threads == 2
    assert limits.embed_threads == 2


def test_env_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_KUZU_THREADS", "8")
    monkeypatch.setenv("SYNAPTIQ_KUZU_MEMORY_MB", "1024")
    limits = resolve_limits("server", cpu_count=16)
    assert limits.db_threads == 8
    assert limits.db_buffer_bytes == 1024 * _MB
    # Untouched field keeps the profile value.
    assert limits.embed_threads == 4


def test_env_overrides_apply_to_interactive_profile(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_EMBED_THREADS", "2")
    limits = resolve_limits("interactive", cpu_count=16)
    assert limits.embed_threads == 2
    assert limits.db_threads == 0


def test_invalid_env_values_ignored(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_KUZU_THREADS", "lots")
    monkeypatch.setenv("SYNAPTIQ_KUZU_MEMORY_MB", "-5")
    limits = resolve_limits("server", cpu_count=16)
    assert limits.db_threads == 4
    assert limits.db_buffer_bytes == SERVER_BUFFER_POOL_MB * _MB


# ---------------------------------------------------------------------------
# SYNAPTIQ_DB_* env vars + deprecated SYNAPTIQ_KUZU_* aliases (W2.7)
# ---------------------------------------------------------------------------


def test_new_db_env_vars_apply(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_DB_THREADS", "8")
    monkeypatch.setenv("SYNAPTIQ_DB_MEMORY_MB", "1024")
    limits = resolve_limits("server", cpu_count=16)
    assert limits.db_threads == 8
    assert limits.db_buffer_bytes == 1024 * _MB


def test_new_db_var_wins_over_deprecated_alias(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_DB_THREADS", "8")
    monkeypatch.setenv("SYNAPTIQ_KUZU_THREADS", "3")
    limits = resolve_limits("server", cpu_count=16)
    assert limits.db_threads == 8


def test_deprecated_kuzu_alias_still_applies(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_KUZU_THREADS", "7")
    monkeypatch.setenv("SYNAPTIQ_KUZU_MEMORY_MB", "256")
    limits = resolve_limits("server", cpu_count=16)
    assert limits.db_threads == 7
    assert limits.db_buffer_bytes == 256 * _MB


def test_deprecated_kuzu_alias_warns_once(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("SYNAPTIQ_KUZU_THREADS", "7")
    with caplog.at_level(logging.WARNING, logger="synaptiq.core.resources"):
        resolve_limits("server", cpu_count=16)
        resolve_limits("server", cpu_count=16)
    warnings = [
        r for r in caplog.records if "SYNAPTIQ_KUZU_THREADS is deprecated" in r.getMessage()
    ]
    assert len(warnings) == 1, "the deprecation warning must fire exactly once per process"


# ---------------------------------------------------------------------------
# --jobs override (resolve_limits(jobs=...) / set_jobs())
# ---------------------------------------------------------------------------


def test_jobs_positive_caps_db_embed_and_pool_workers():
    limits = resolve_limits("interactive", cpu_count=16, jobs=2)
    assert limits.db_threads == 2
    assert limits.embed_threads == 2
    assert limits.pool_workers == 2


def test_jobs_positive_does_not_affect_buffer_bytes():
    """--jobs caps thread/worker counts, never the engine's buffer-pool memory."""
    interactive = resolve_limits("interactive", cpu_count=16, jobs=2)
    assert interactive.db_buffer_bytes == 0

    server = resolve_limits("server", cpu_count=16, jobs=2)
    assert server.db_buffer_bytes == SERVER_BUFFER_POOL_MB * _MB


def test_jobs_zero_is_explicit_all_cores_escape_hatch():
    """`--jobs 0` restores uncapped engine/embedding threads, overriding even
    the new polite interactive embed-thread default."""
    limits = resolve_limits("interactive", cpu_count=16, jobs=0)
    assert limits.db_threads == 0
    assert limits.embed_threads == 0
    # Pool workers have no "uncapped" concept — falls back to min(8, cpus).
    assert limits.pool_workers == 8


def test_jobs_positive_overrides_env_vars(monkeypatch):
    """Precedence: flag > env vars — --jobs wins even when SYNAPTIQ_* is set."""
    monkeypatch.setenv("SYNAPTIQ_EMBED_THREADS", "3")
    monkeypatch.setenv("SYNAPTIQ_KUZU_THREADS", "3")
    limits = resolve_limits("interactive", cpu_count=16, jobs=5)
    assert limits.db_threads == 5
    assert limits.embed_threads == 5
    assert limits.pool_workers == 5


def test_jobs_zero_overrides_env_vars_too(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_EMBED_THREADS", "3")
    limits = resolve_limits("interactive", cpu_count=16, jobs=0)
    assert limits.embed_threads == 0


def test_jobs_none_falls_back_to_env_vars_then_profile(monkeypatch):
    """No --jobs (None, the default): env vars still apply as before."""
    limits = resolve_limits("interactive", cpu_count=16, jobs=None)
    assert limits.embed_threads == 14  # polite profile default, untouched

    monkeypatch.setenv("SYNAPTIQ_EMBED_THREADS", "3")
    limits = resolve_limits("interactive", cpu_count=16, jobs=None)
    assert limits.embed_threads == 3


def test_jobs_propagates_via_set_jobs_and_current_limits(monkeypatch):
    """--jobs N (via set_jobs) propagates through current_limits(), scaled
    against a monkeypatched cpu_count."""
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 6)
    set_jobs(2)
    limits = current_limits()
    assert limits.db_threads == 2
    assert limits.embed_threads == 2
    assert limits.pool_workers == 2


def test_set_jobs_none_clears_override(monkeypatch):
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 16)
    set_jobs(3)
    assert current_limits().db_threads == 3
    set_jobs(None)
    assert current_limits().db_threads == 0  # interactive default again
    assert current_limits().embed_threads == 14  # polite default again


def test_set_jobs_rejects_negative():
    with pytest.raises(ValueError):
        set_jobs(-1)


def test_set_profile_changes_current_limits():
    set_profile("server")
    assert current_limits().db_threads >= 2
    set_profile("interactive")
    assert current_limits().db_threads == 0


def test_set_profile_rejects_unknown_role():
    with pytest.raises(ValueError):
        set_profile("turbo")
