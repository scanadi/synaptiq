"""Benchmark: Token reduction measurement.

Compares the token count of Synaptiq tool responses versus reading
the equivalent raw file content, demonstrating how much context
the knowledge graph saves.
"""

from __future__ import annotations

import pytest

from synaptiq.mcp.token_budget import count_tokens
from synaptiq.mcp.tools import handle_context, handle_file_context, handle_query


@pytest.mark.benchmark
class TestTokenReduction:
    """Token reduction benchmarks."""

    def test_query_vs_raw_files(self, indexed_small_repo):
        """Compare search result tokens vs reading all matched files."""
        repo_dir, storage, graph, _ = indexed_small_repo

        # Synaptiq query response.
        query_result = handle_query(storage, "process")
        query_tokens = count_tokens(query_result)

        # Equivalent raw content: read all Python files in the repo.
        raw_tokens = 0
        for py_file in repo_dir.rglob("*.py"):
            try:
                raw_tokens += count_tokens(py_file.read_text())
            except Exception:
                pass

        reduction = (1.0 - query_tokens / raw_tokens) * 100 if raw_tokens > 0 else 0

        print("\n[query vs raw]")
        print(f"  Query response: {query_tokens} tokens")
        print(f"  Raw file content: {raw_tokens} tokens")
        print(f"  Reduction: {reduction:.1f}%")

        assert reduction > 50, f"Token reduction too low: {reduction:.1f}%"

    def test_context_vs_raw_file(self, indexed_small_repo):
        """Compare context response tokens vs the full source file."""
        repo_dir, storage, graph, _ = indexed_small_repo

        # Context for a specific symbol.
        context_result = handle_context(storage, "process_0")
        context_tokens = count_tokens(context_result)

        # Raw file content for the file containing the symbol.
        raw_file = repo_dir / "src" / "app" / "mod_0.py"
        raw_tokens = count_tokens(raw_file.read_text()) if raw_file.exists() else 1

        print("\n[context vs raw file]")
        print(f"  Context response: {context_tokens} tokens")
        print(f"  Raw file: {raw_tokens} tokens")
        # Context may be larger than a single file (includes callers/callees
        # from other files), so we just report the ratio.
        ratio = context_tokens / raw_tokens if raw_tokens > 0 else 0
        print(f"  Ratio: {ratio:.1f}x (context/raw)")

    def test_file_context_vs_raw(self, indexed_small_repo):
        """Compare file_context response vs reading the raw file."""
        _, storage, _, _ = indexed_small_repo

        fc_result = handle_file_context(storage, "src/app/mod_0.py")
        fc_tokens = count_tokens(fc_result)

        # The file_context tool returns structured metadata, not raw source.
        print("\n[file_context]")
        print(f"  Response: {fc_tokens} tokens")
        print("  (Structured metadata — no raw source comparison needed)")

        assert fc_tokens > 0
