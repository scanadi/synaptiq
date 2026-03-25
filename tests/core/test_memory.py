"""Tests for the persistent memory store."""

import json

from synaptiq.core.memory import MemoryStore


class TestMemoryStore:
    def test_remember_and_recall(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.remember("auth-flow", "Uses JWT tokens via auth middleware", "architecture")

        facts = store.recall("auth-flow")
        assert len(facts) == 1
        assert facts[0].key == "auth-flow"
        assert facts[0].value == "Uses JWT tokens via auth middleware"
        assert facts[0].category == "architecture"

    def test_remember_updates_existing(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.remember("key1", "value1")
        store.remember("key1", "updated value")

        facts = store.recall("key1")
        assert len(facts) == 1
        assert facts[0].value == "updated value"

    def test_fuzzy_recall(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.remember("auth-flow", "Uses JWT tokens via auth middleware", "architecture")
        store.remember("db-pattern", "Repository pattern with SQLAlchemy", "pattern")

        facts = store.recall("JWT authentication")
        assert len(facts) >= 1
        assert facts[0].key == "auth-flow"

    def test_recall_no_results(self, tmp_path):
        store = MemoryStore(tmp_path)
        facts = store.recall("nonexistent")
        assert len(facts) == 0

    def test_forget(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.remember("key1", "value1")
        assert store.forget("key1") is True
        assert store.forget("key1") is False
        assert store.recall("key1") == []

    def test_list_all(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.remember("k1", "v1", "cat-a")
        store.remember("k2", "v2", "cat-b")
        store.remember("k3", "v3", "cat-a")

        all_facts = store.list_all()
        assert len(all_facts) == 3

        cat_a = store.list_all(category="cat-a")
        assert len(cat_a) == 2

    def test_persistence_across_instances(self, tmp_path):
        store1 = MemoryStore(tmp_path)
        store1.remember("persistent", "this should survive")

        store2 = MemoryStore(tmp_path)
        facts = store2.recall("persistent")
        assert len(facts) == 1
        assert facts[0].value == "this should survive"

    def test_empty_store(self, tmp_path):
        store = MemoryStore(tmp_path)
        assert store.list_all() == []
        assert store.recall("anything") == []
        assert store.forget("anything") is False

    def test_json_file_created(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.remember("key", "value")
        assert (tmp_path / "memory.json").exists()
        data = json.loads((tmp_path / "memory.json").read_text())
        assert "key" in data
