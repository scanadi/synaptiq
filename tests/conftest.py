"""Shared fixtures for the whole test suite."""

from __future__ import annotations

import pytest

from synaptiq.core import resources


def pytest_collection_modifyitems(config, items):
    """Exclude the ``equivalence_soak`` (100-script) suite unless it is selected.

    The soak is a heavy opt-in (design §11 / W3.2f): it runs only when the marker
    is named explicitly (``-m equivalence_soak``); otherwise it is skipped so the
    default suite — including ``uv run pytest tests/core/`` — stays fast.
    """
    markexpr = config.getoption("-m", default="")
    if "equivalence_soak" in markexpr:
        return
    skip = pytest.mark.skip(reason="soak excluded by default; select with -m equivalence_soak")
    for item in items:
        if "equivalence_soak" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _reset_resource_profile():
    """Make tests hermetic against the process-global resource profile.

    CLI tests invoke ``serve``/``mcp``/``watch`` in-process, which switch
    the profile to "server"; without a per-test reset, every later test
    would see capped engine limits (and e.g. embedder call signatures
    change with the runner's core count).  Also resets the ``--jobs``
    override (``analyze --jobs`` / ``set_jobs``) for the same reason.
    """
    resources._active_profile = "interactive"
    resources._jobs_override = None
    yield
