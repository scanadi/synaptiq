"""Tests for synaptiq.core.daemon.lock — LockManager and LockInfo."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from synaptiq.core.daemon.lock import LockInfo, LockManager

# ======================================================================
# TestLockManagerAcquire
# ======================================================================


class TestLockManagerAcquire:
    """Tests for acquiring and releasing the lock."""

    def test_acquire_creates_lock_file(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".synaptiq"
        mgr = LockManager(data_dir)
        info = mgr.try_acquire()
        try:
            assert info is not None
            assert (data_dir / "synaptiq.lock").exists()
        finally:
            mgr.release()

    def test_acquire_writes_valid_json(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".synaptiq"
        mgr = LockManager(data_dir)
        info = mgr.try_acquire()
        try:
            assert info is not None
            data = json.loads((data_dir / "synaptiq.lock").read_text())
            assert data["pid"] == os.getpid()
            assert "socket" in data
            assert "started_at" in data
        finally:
            mgr.release()

    def test_second_acquire_fails(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".synaptiq"
        mgr1 = LockManager(data_dir)
        mgr2 = LockManager(data_dir)
        info1 = mgr1.try_acquire()
        try:
            assert info1 is not None
            info2 = mgr2.try_acquire()
            assert info2 is None
        finally:
            mgr1.release()

    def test_release_removes_lock_and_socket(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".synaptiq"
        mgr = LockManager(data_dir)
        mgr.try_acquire()
        # Create a dummy socket file so release() can remove it.
        mgr.socket_path.touch()
        mgr.release()
        assert not (data_dir / "synaptiq.lock").exists()
        assert not mgr.socket_path.exists()

    def test_creates_data_dir_if_missing(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "deep" / "nested" / ".synaptiq"
        assert not data_dir.exists()
        mgr = LockManager(data_dir)
        info = mgr.try_acquire()
        try:
            assert info is not None
            assert data_dir.exists()
        finally:
            mgr.release()


# ======================================================================
# TestLockManagerRead
# ======================================================================


class TestLockManagerRead:
    """Tests for reading lock info and staleness detection."""

    def test_read_existing_returns_info(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".synaptiq"
        mgr1 = LockManager(data_dir)
        info = mgr1.try_acquire()
        try:
            assert info is not None

            mgr2 = LockManager(data_dir)
            existing = mgr2.read_existing()
            assert existing is not None
            assert existing.pid == os.getpid()
            assert existing.socket == str(mgr1.socket_path)
        finally:
            mgr1.release()

    def test_read_existing_returns_none_when_no_lock(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".synaptiq"
        mgr = LockManager(data_dir)
        assert mgr.read_existing() is None

    def test_is_stale_detects_dead_pid(self) -> None:
        info = LockInfo(pid=99999999, socket="/tmp/fake.sock", started_at="2025-01-01T00:00:00")
        assert info.is_stale() is True

    def test_is_stale_false_for_live_pid(self) -> None:
        info = LockInfo(pid=os.getpid(), socket="/tmp/fake.sock", started_at="2025-01-01T00:00:00")
        assert info.is_stale() is False

    def test_is_stale_returns_false_on_permission_error(self) -> None:
        info = LockInfo(pid=99999, socket="/tmp/fake.sock", started_at="2025-01-01T00:00:00")
        with patch("os.kill", side_effect=PermissionError):
            assert info.is_stale() is False
