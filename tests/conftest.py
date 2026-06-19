"""Shared fixtures for the whole test suite."""

from __future__ import annotations

import pytest

from synaptiq.core import resources


@pytest.fixture(autouse=True)
def _reset_resource_profile():
    """Make tests hermetic against the process-global resource profile.

    CLI tests invoke ``serve``/``mcp``/``watch`` in-process, which switch
    the profile to "server"; without a per-test reset, every later test
    would see capped engine limits (and e.g. embedder call signatures
    change with the runner's core count).
    """
    resources._active_profile = "interactive"
    yield
