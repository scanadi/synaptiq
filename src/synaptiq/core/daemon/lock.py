"""Lock file manager for primary/proxy coordination.

Uses fcntl.flock() for OS-level exclusive locking. The lock file lives at
.synaptiq/synaptiq.lock and contains JSON with PID, socket path, and timestamp.
The OS automatically releases the flock when the process crashes or exits.

Socket Path
-----------
On macOS the AF_UNIX path limit is 104 bytes.  When the data directory path
is long, the socket is placed in ``/tmp/synaptiq-<hash>.sock`` to stay within
the limit.  The socket path is stored in the lock file so proxies always
know where to connect.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# macOS AF_UNIX limit is 104 bytes; leave a safety margin.
_MAX_SOCKET_PATH_LEN = 100


@dataclass
class LockInfo:
    """Information stored in the lock file."""

    pid: int
    socket: str
    started_at: str

    def is_stale(self) -> bool:
        """Check if the PID that owns this lock is still alive."""
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Process exists but we don't have permission to signal it.
            return False
        return False

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "pid": self.pid,
            "socket": self.socket,
            "started_at": self.started_at,
        }


def _compute_socket_path(data_dir: Path) -> Path:
    """Compute a socket path that fits within OS limits.

    Prefers ``<data_dir>/synaptiq.sock`` when it fits.  Falls back to
    ``/tmp/synaptiq-<hash>.sock`` for long paths (macOS 104-byte limit).
    """
    local = data_dir / "synaptiq.sock"
    if len(str(local)) <= _MAX_SOCKET_PATH_LEN:
        return local
    # Use a hash of the data directory to create a short, unique path.
    dir_hash = hashlib.sha256(str(data_dir).encode()).hexdigest()[:12]
    return Path(f"/tmp/synaptiq-{dir_hash}.sock")


class LockManager:
    """Manages the .synaptiq/synaptiq.lock file using fcntl.flock().

    Parameters
    ----------
    data_dir:
        Path to the ``.synaptiq`` directory (will be created if missing).
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._lock_path = data_dir / "synaptiq.lock"
        self._socket_path = _compute_socket_path(data_dir)
        self._fd: int | None = None

    @property
    def socket_path(self) -> Path:
        """Return the path where the UNIX domain socket should be created."""
        return self._socket_path

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def try_acquire(self) -> LockInfo | None:
        """Try to acquire an exclusive flock on the lock file.

        Returns a :class:`LockInfo` on success, or ``None`` if the lock is
        already held by another process.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)

        fd = os.open(
            str(self._lock_path),
            os.O_CREAT | os.O_RDWR,
            0o644,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None

        info = LockInfo(
            pid=os.getpid(),
            socket=str(self._socket_path),
            started_at=datetime.now(tz=timezone.utc).isoformat(),
        )

        payload = json.dumps(info.to_dict()).encode()
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)

        self._fd = fd
        return info

    def read_existing(self) -> LockInfo | None:
        """Read the lock file written by another process.

        Returns ``None`` when the lock file does not exist or cannot be
        parsed.
        """
        try:
            data = self._lock_path.read_text()
        except FileNotFoundError:
            return None

        try:
            obj = json.loads(data)
            return LockInfo(
                pid=obj["pid"],
                socket=obj["socket"],
                started_at=obj["started_at"],
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def release(self) -> None:
        """Release the flock, close the fd, and remove lock/socket files."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

        self._remove_files()

    def force_cleanup(self) -> None:
        """Remove stale lock and socket files without holding the lock."""
        self._remove_files()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_files(self) -> None:
        for p in (self._lock_path, self._socket_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
