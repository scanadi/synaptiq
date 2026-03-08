"""Tests for serve command primary/proxy mode detection logic."""

from __future__ import annotations

import json
import os
from pathlib import Path

from synaptiq.core.daemon.lock import LockManager


class TestServeModeDetection:
    """Verify that LockManager coordination drives primary vs proxy selection."""

    def test_first_instance_becomes_primary(self, tmp_path: Path) -> None:
        """When no lock exists, try_acquire succeeds."""
        lock_mgr = LockManager(tmp_path)
        lock_info = lock_mgr.try_acquire()

        assert lock_info is not None
        assert lock_info.pid == os.getpid()
        assert lock_info.socket == str(tmp_path / "synaptiq.sock")

        lock_mgr.release()

    def test_second_instance_detects_existing_primary(self, tmp_path: Path) -> None:
        """When lock is held, try_acquire fails and read_existing returns info."""
        # First instance acquires the lock.
        primary = LockManager(tmp_path)
        primary_info = primary.try_acquire()
        assert primary_info is not None

        # Second instance tries to acquire — should fail.
        proxy = LockManager(tmp_path)
        proxy_info = proxy.try_acquire()
        assert proxy_info is None

        # But it can read the existing lock info.
        existing = proxy.read_existing()
        assert existing is not None
        assert existing.pid == os.getpid()
        assert existing.socket == str(tmp_path / "synaptiq.sock")

        primary.release()

    def test_stale_lock_gets_cleaned_up(self, tmp_path: Path) -> None:
        """Stale lock (dead PID) is cleaned up, new instance takes over."""
        # Simulate a stale lock by writing a lock file with a non-existent PID.
        lock_path = tmp_path / "synaptiq.lock"
        stale_info = {
            "pid": 99999999,  # Very unlikely to be a real PID
            "socket": str(tmp_path / "synaptiq.sock"),
            "started_at": "2024-01-01T00:00:00+00:00",
        }
        lock_path.write_text(json.dumps(stale_info))

        lock_mgr = LockManager(tmp_path)

        # read_existing should return info with the dead PID.
        existing = lock_mgr.read_existing()
        assert existing is not None
        assert existing.pid == 99999999
        assert existing.is_stale() is True

        # After cleanup, we can acquire.
        lock_mgr.force_cleanup()
        lock_info = lock_mgr.try_acquire()
        assert lock_info is not None
        assert lock_info.pid == os.getpid()

        lock_mgr.release()

    def test_read_existing_returns_none_when_no_lock(self, tmp_path: Path) -> None:
        """read_existing returns None when no lock file exists."""
        lock_mgr = LockManager(tmp_path)
        assert lock_mgr.read_existing() is None

    def test_lock_release_removes_files(self, tmp_path: Path) -> None:
        """After release, lock and socket files are removed."""
        lock_mgr = LockManager(tmp_path)
        lock_mgr.try_acquire()
        lock_mgr.release()

        assert not (tmp_path / "synaptiq.lock").exists()
        assert not (tmp_path / "synaptiq.sock").exists()
