"""Tests for role-aware resource limits."""

from __future__ import annotations

import pytest

from synaptiq.core import resources
from synaptiq.core.resources import (
    SERVER_BUFFER_POOL_MB,
    ResourceLimits,
    current_limits,
    resolve_limits,
    set_profile,
)

_MB = 1024 * 1024

ENV_VARS = (
    "SYNAPTIQ_KUZU_THREADS",
    "SYNAPTIQ_KUZU_MEMORY_MB",
    "SYNAPTIQ_EMBED_THREADS",
)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Strip env overrides and restore the interactive profile after each test."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    resources._active_profile = "interactive"


def test_interactive_profile_uses_library_defaults():
    assert resolve_limits("interactive", cpu_count=16) == ResourceLimits(0, 0, 0)


def test_server_profile_caps_scale_with_cores():
    limits = resolve_limits("server", cpu_count=16)
    assert limits.kuzu_threads == 4
    assert limits.embed_threads == 4
    assert limits.kuzu_buffer_bytes == SERVER_BUFFER_POOL_MB * _MB


def test_server_profile_floor_of_two_threads():
    limits = resolve_limits("server", cpu_count=2)
    assert limits.kuzu_threads == 2
    assert limits.embed_threads == 2


def test_env_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_KUZU_THREADS", "8")
    monkeypatch.setenv("SYNAPTIQ_KUZU_MEMORY_MB", "1024")
    limits = resolve_limits("server", cpu_count=16)
    assert limits.kuzu_threads == 8
    assert limits.kuzu_buffer_bytes == 1024 * _MB
    # Untouched field keeps the profile value.
    assert limits.embed_threads == 4


def test_env_overrides_apply_to_interactive_profile(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_EMBED_THREADS", "2")
    limits = resolve_limits("interactive", cpu_count=16)
    assert limits.embed_threads == 2
    assert limits.kuzu_threads == 0


def test_invalid_env_values_ignored(monkeypatch):
    monkeypatch.setenv("SYNAPTIQ_KUZU_THREADS", "lots")
    monkeypatch.setenv("SYNAPTIQ_KUZU_MEMORY_MB", "-5")
    limits = resolve_limits("server", cpu_count=16)
    assert limits.kuzu_threads == 4
    assert limits.kuzu_buffer_bytes == SERVER_BUFFER_POOL_MB * _MB


def test_set_profile_changes_current_limits():
    set_profile("server")
    assert current_limits().kuzu_threads >= 2
    set_profile("interactive")
    assert current_limits().kuzu_threads == 0


def test_set_profile_rejects_unknown_role():
    with pytest.raises(ValueError):
        set_profile("turbo")
