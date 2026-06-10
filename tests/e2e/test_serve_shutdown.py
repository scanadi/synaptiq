"""End-to-end: serve must exit promptly on stdin hangup and SIGTERM.

The MCP SDK's server.run() neither returns on stdin EOF nor honours task
cancellation, so without the stdin sentinel + shutdown watchdog a serve
process wedges forever while holding the kuzu lock.  These tests run the
real CLI in a subprocess and assert bounded shutdown.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_EXIT_BUDGET = 35.0  # watchdog grace (20s) + teardown margin


def _init_repo(repo: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q", "-b", "main")
    (repo / "a.ts").write_text("export function hello(): void {}\n")
    git("add", ".")
    git("commit", "-qm", "init")


def _spawn_serve(repo: Path) -> subprocess.Popen:
    bootstrap = (
        "import sys;"
        "sys.argv = ['synaptiq', 'serve', '--watch', '--path', sys.argv[1]];"
        "from synaptiq.cli.main import app; app()"
    )
    return subprocess.Popen(
        [sys.executable, "-c", bootstrap, str(repo)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_lock(repo: Path, timeout: float = 90.0) -> None:
    lock = repo / ".synaptiq" / "synaptiq.lock"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if lock.exists():
            return
        time.sleep(0.5)
    pytest.fail("serve never wrote its lock file")


def _assert_exits(proc: subprocess.Popen) -> None:
    try:
        proc.wait(timeout=_EXIT_BUDGET)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail(f"serve still running {_EXIT_BUDGET}s after shutdown signal")


class TestServeShutdown:
    def test_exits_on_stdin_hangup(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        proc = _spawn_serve(tmp_path)
        try:
            _wait_for_lock(tmp_path)
            proc.stdin.close()
            _assert_exits(proc)
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_exits_on_sigterm(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        proc = _spawn_serve(tmp_path)
        try:
            _wait_for_lock(tmp_path)
            proc.send_signal(signal.SIGTERM)
            _assert_exits(proc)
        finally:
            if proc.poll() is None:
                proc.kill()
