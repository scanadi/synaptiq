"""Non-blocking PyPI update check with local caching.

Checks are cached for 24 hours in ``~/.synaptiq/update_check.json`` so the
CLI never blocks on network I/O.  The actual PyPI fetch runs in a daemon
thread started at CLI startup; results are shown on the *next* invocation.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from synaptiq import __version__

CACHE_DIR = Path.home() / ".synaptiq"
CACHE_FILE = CACHE_DIR / "update_check.json"
CACHE_TTL = 86400  # 24 hours
PYPI_URL = "https://pypi.org/pypi/synaptiq/json"


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a semver-ish string into a comparable tuple."""
    return tuple(int(x) for x in v.split("."))


def _fetch_latest_version() -> Optional[str]:
    """Fetch the latest version from PyPI.  Returns ``None`` on any failure."""
    try:
        from urllib.request import Request, urlopen

        req = Request(PYPI_URL, headers={"Accept": "application/json"})
        with urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def _read_cache() -> Optional[dict]:
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if time.time() - data.get("checked_at", 0) < CACHE_TTL:
                return data
    except Exception:
        pass
    return None


def _write_cache(latest_version: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps(
                {
                    "latest_version": latest_version,
                    "checked_at": time.time(),
                    "current_version": __version__,
                }
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def check_for_update_message() -> Optional[str]:
    """Return an update notification string if outdated, else ``None``.

    Reads from cache only — no network I/O.
    """
    cached = _read_cache()
    if cached is None:
        return None
    latest = cached.get("latest_version", "")
    if not latest or latest == __version__:
        return None
    try:
        if _parse_version(latest) > _parse_version(__version__):
            return (
                f"Update available: v{__version__} → v{latest}. "
                f"Run [bold]pip install --upgrade synaptiq[/bold] to update."
            )
    except (ValueError, TypeError):
        pass
    return None


def trigger_background_check() -> None:
    """Spawn a daemon thread to check PyPI and refresh the cache.

    Returns immediately.  Does nothing if the cache is still fresh.
    """
    if _read_cache() is not None:
        return

    def _check():
        latest = _fetch_latest_version()
        if latest:
            _write_cache(latest)

    thread = threading.Thread(target=_check, daemon=True)
    thread.start()
