"""Persistent memory store for agent-discovered facts.

Stores key-value facts in ``.synaptiq/memory.json`` so they survive
re-indexing and persist across agent sessions.

Concurrency: every read-modify-write cycle holds an exclusive ``flock``
on a dedicated lock file (``memory.lock``), and saves go through a
uniquely-named temp file followed by an atomic rename.  Locking the data
file itself would not work — ``open(..., "w")`` truncates *before* the
lock can be acquired.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Fact:
    """A single remembered fact."""

    key: str
    value: str
    category: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class MemoryStore:
    """File-backed key-value store for agent facts.

    Parameters
    ----------
    synaptiq_dir:
        Path to the ``.synaptiq`` directory (created if missing).
    """

    def __init__(self, synaptiq_dir: Path | None = None) -> None:
        if synaptiq_dir is None:
            synaptiq_dir = Path.cwd() / ".synaptiq"
        self._path = synaptiq_dir / "memory.json"
        self._lock_path = synaptiq_dir / "memory.lock"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(self, key: str, value: str, category: str = "") -> Fact:
        """Create or update a fact.

        If a fact with the same *key* already exists its value, category,
        and ``updated_at`` timestamp are updated.
        """
        with self._locked():
            data = self._load()
            now = time.time()
            existing = data.get(key)
            if existing:
                existing["value"] = value
                existing["category"] = category or existing.get("category", "")
                existing["updated_at"] = now
                fact = Fact(**existing)
            else:
                fact = Fact(
                    key=key, value=value, category=category, created_at=now, updated_at=now
                )
                data[key] = asdict(fact)
            self._save(data)
        return fact

    def recall(self, query: str, limit: int = 10) -> list[Fact]:
        """Search facts by query.

        Exact key match is returned first.  Otherwise facts are ranked by
        word-overlap between the query and the fact's key + value + category.
        """
        data = self._load()
        if not data:
            return []

        # Exact key match.
        if query in data:
            return [Fact(**data[query])]

        query_words = set(query.lower().split())
        scored: list[tuple[float, Fact]] = []
        for entry in data.values():
            fact = Fact(**entry)
            text_words = set(
                f"{fact.key} {fact.value} {fact.category}".lower().split()
            )
            overlap = len(query_words & text_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((score, fact))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [f for _, f in scored[:limit]]

    def forget(self, key: str) -> bool:
        """Remove a fact by key.  Returns ``True`` if it existed."""
        with self._locked():
            data = self._load()
            if key not in data:
                return False
            del data[key]
            self._save(data)
        return True

    def list_all(self, category: str | None = None) -> list[Fact]:
        """Return all facts, optionally filtered by category."""
        data = self._load()
        facts = [Fact(**v) for v in data.values()]
        if category:
            facts = [f for f in facts if f.category == category]
        facts.sort(key=lambda f: f.updated_at, reverse=True)
        return facts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive flock on the lock file for a read-modify-write."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name + atomic rename: concurrent writers can never
        # truncate each other's in-progress file.
        fd, tmp_name = tempfile.mkstemp(
            prefix="memory.", suffix=".tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp_name, self._path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
