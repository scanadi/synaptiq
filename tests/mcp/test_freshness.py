"""Tests for the MCP index-freshness trailer (W4.5).

Covers the standalone helper in ``mcp/freshness.py`` directly (age
formatting, embeddings-state precedence, omission rules, the env escape
hatch, the never-raises contract, and the TTL cache) plus the two places it
is wired in centrally: ``dispatch_tool`` (every tool) and
``dispatch_resource`` (``synaptiq://overview`` only). A final test proves
the proxy-mode claim in ``dispatch_tool``'s docstring end-to-end over a real
socket instead of just asserting it.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synaptiq.core.daemon.socket_client import SocketClient
from synaptiq.core.daemon.socket_server import SocketServer
from synaptiq.mcp import freshness as freshness_module
from synaptiq.mcp.freshness import freshness_trailer
from synaptiq.mcp.server import dispatch_resource, dispatch_tool
from synaptiq.mcp.token_budget import strip_metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def _write_meta(data_dir: Path, **meta: object) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_state(data_dir: Path, **state: object) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "embeddings_state.json").write_text(json.dumps(state), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_cache():
    """The TTL cache is process-global; keep tests hermetic regardless."""
    freshness_module._cache.clear()
    yield
    freshness_module._cache.clear()


# ---------------------------------------------------------------------------
# freshness_trailer() — omission rules
# ---------------------------------------------------------------------------


class TestOmission:
    def test_omitted_when_meta_missing(self, tmp_path):
        assert freshness_trailer(tmp_path / ".synaptiq") == ""

    def test_omitted_when_data_dir_missing(self, tmp_path):
        assert freshness_trailer(tmp_path / "does-not-exist" / ".synaptiq") == ""


# ---------------------------------------------------------------------------
# freshness_trailer() — index age formatting
# ---------------------------------------------------------------------------


class TestAgeFormatting:
    def test_seconds(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(30))
        trailer = freshness_trailer(data_dir)
        assert "30s old" in trailer

    def test_minutes(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(4 * 60))
        trailer = freshness_trailer(data_dir)
        assert "4m old" in trailer

    def test_hours(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(3 * 3600))
        trailer = freshness_trailer(data_dir)
        assert "3h old" in trailer

    def test_days(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(2 * 86400))
        trailer = freshness_trailer(data_dir)
        assert "2d old" in trailer

    def test_age_unknown_when_timestamp_key_missing(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir)  # no last_indexed_at at all
        assert freshness_trailer(data_dir) == "[index: age unknown]"

    def test_age_unknown_when_timestamp_unparseable(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at="not-a-real-timestamp")
        assert "age unknown" in freshness_trailer(data_dir)


# ---------------------------------------------------------------------------
# freshness_trailer() — embeddings fragment
# ---------------------------------------------------------------------------


class TestEmbeddingsFragment:
    def test_complete(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))
        _write_state(data_dir, state="complete", done=100, total=100)
        assert "embeddings: complete" in freshness_trailer(data_dir)

    def test_encoding_in_progress(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))
        _write_state(data_dir, state="encoding", done=12431, total=26203)
        assert "embeddings: encoding 12431/26203" in freshness_trailer(data_dir)

    def test_failed(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))
        _write_state(data_dir, state="failed", error="boom")
        assert "embeddings: failed" in freshness_trailer(data_dir)

    def test_deferred(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))
        _write_state(data_dir, state="deferred", detail="index locked")
        assert "embeddings: deferred" in freshness_trailer(data_dir)

    def test_falls_back_to_meta_stats_when_state_file_absent(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10), stats={"embeddings": 4242})
        assert "embeddings: 4242" in freshness_trailer(data_dir)

    def test_fragment_omitted_when_nothing_to_show(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))
        trailer = freshness_trailer(data_dir)
        assert trailer == "[index: 10s old]"
        assert "embeddings" not in trailer


# ---------------------------------------------------------------------------
# freshness_trailer() — never raises
# ---------------------------------------------------------------------------


class TestNeverRaises:
    def test_corrupt_state_file_yields_partial_trailer(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "embeddings_state.json").write_text("{not valid json", encoding="utf-8")

        trailer = freshness_trailer(data_dir)  # must not raise

        assert trailer == "[index: 10s old]"

    def test_corrupt_meta_json_yields_age_unknown(self, tmp_path):
        data_dir = tmp_path / ".synaptiq"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "meta.json").write_text("{not valid json", encoding="utf-8")

        trailer = freshness_trailer(data_dir)  # must not raise

        assert "age unknown" in trailer

    def test_unexpected_filesystem_error_is_swallowed(self, tmp_path, monkeypatch):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))

        def _boom(self):
            raise OSError("simulated failure")

        # Not caught by any inner guard — exercises the outer try/except.
        monkeypatch.setattr(Path, "exists", _boom)

        assert freshness_trailer(data_dir) == ""


# ---------------------------------------------------------------------------
# freshness_trailer() — env escape hatch
# ---------------------------------------------------------------------------


class TestEnvDisable:
    def test_disabled_via_env(self, tmp_path, monkeypatch):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))
        monkeypatch.setenv("SYNAPTIQ_MCP_FRESHNESS", "0")
        assert freshness_trailer(data_dir) == ""

    def test_enabled_when_unset(self, tmp_path, monkeypatch):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))
        monkeypatch.delenv("SYNAPTIQ_MCP_FRESHNESS", raising=False)
        assert freshness_trailer(data_dir) != ""

    def test_only_the_literal_string_zero_disables(self, tmp_path, monkeypatch):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))
        monkeypatch.setenv("SYNAPTIQ_MCP_FRESHNESS", "false")
        assert freshness_trailer(data_dir) != ""


# ---------------------------------------------------------------------------
# freshness_trailer() — TTL cache
# ---------------------------------------------------------------------------


class TestCaching:
    def test_stays_cached_within_ttl(self, tmp_path, monkeypatch):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))

        fake_now = [1_000.0]
        monkeypatch.setattr(freshness_module, "_monotonic", lambda: fake_now[0])

        first = freshness_trailer(data_dir)
        assert first == "[index: 10s old]"

        # Rewrite meta.json with a very different age; clock hasn't moved.
        _write_meta(data_dir, last_indexed_at=_iso(4 * 60))
        second = freshness_trailer(data_dir)

        assert second == first  # served from cache, not re-read

    def test_refreshes_once_ttl_elapses(self, tmp_path, monkeypatch):
        data_dir = tmp_path / ".synaptiq"
        _write_meta(data_dir, last_indexed_at=_iso(10))

        fake_now = [1_000.0]
        monkeypatch.setattr(freshness_module, "_monotonic", lambda: fake_now[0])

        freshness_trailer(data_dir)

        _write_meta(data_dir, last_indexed_at=_iso(4 * 60))
        fake_now[0] += freshness_module._CACHE_TTL_SECONDS + 1.0

        assert "4m old" in freshness_trailer(data_dir)

    def test_distinct_data_dirs_do_not_share_a_cache_entry(self, tmp_path, monkeypatch):
        dir_a = tmp_path / "a" / ".synaptiq"
        dir_b = tmp_path / "b" / ".synaptiq"
        _write_meta(dir_a, last_indexed_at=_iso(10))
        _write_meta(dir_b, last_indexed_at=_iso(4 * 60))

        monkeypatch.setattr(freshness_module, "_monotonic", lambda: 1_000.0)

        assert "10s old" in freshness_trailer(dir_a)
        assert "4m old" in freshness_trailer(dir_b)


# ---------------------------------------------------------------------------
# dispatch_tool — centrally wired for every tool
# ---------------------------------------------------------------------------


class TestDispatchToolIntegration:
    def test_trailer_present_for_query(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_meta(tmp_path / ".synaptiq", last_indexed_at=_iso(4 * 60))
        with patch("synaptiq.mcp.server.handle_query", return_value="1. Foo (Function)"):
            result = dispatch_tool("synaptiq_query", {"query": "foo"}, MagicMock())
        assert "1. Foo (Function)" in result
        assert "[index: 4m old]" in result

    def test_trailer_present_for_context(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_meta(tmp_path / ".synaptiq", last_indexed_at=_iso(4 * 60))
        with patch("synaptiq.mcp.server.handle_context", return_value="Symbol: Foo (Class)"):
            result = dispatch_tool("synaptiq_context", {"symbol": "Foo"}, MagicMock())
        assert "Symbol: Foo (Class)" in result
        assert "[index: 4m old]" in result

    def test_trailer_present_for_dead_code(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_meta(tmp_path / ".synaptiq", last_indexed_at=_iso(4 * 60))
        with patch("synaptiq.mcp.server.handle_dead_code", return_value="No dead code detected."):
            result = dispatch_tool("synaptiq_dead_code", {}, MagicMock())
        assert "No dead code detected." in result
        assert "[index: 4m old]" in result

    def test_trailer_omitted_when_no_index(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no .synaptiq at all
        with patch("synaptiq.mcp.server.handle_dead_code", return_value="No dead code detected."):
            result = dispatch_tool("synaptiq_dead_code", {}, MagicMock())
        assert "No dead code detected." in result
        assert "[index:" not in result

    def test_trailer_disabled_via_env(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_meta(tmp_path / ".synaptiq", last_indexed_at=_iso(4 * 60))
        monkeypatch.setenv("SYNAPTIQ_MCP_FRESHNESS", "0")
        with patch("synaptiq.mcp.server.handle_dead_code", return_value="No dead code detected."):
            result = dispatch_tool("synaptiq_dead_code", {}, MagicMock())
        assert "[index:" not in result

    def test_trailer_survives_strip_metadata(self, tmp_path, monkeypatch):
        """CLI reads use ``strip_metadata`` to drop the tokens footer for
        display — the trailer must sit BEFORE that footer (not after) or
        the regex in ``token_budget.strip_metadata`` stops matching and the
        raw ``--- tokens: N ---`` footer leaks into CLI output."""
        monkeypatch.chdir(tmp_path)
        _write_meta(tmp_path / ".synaptiq", last_indexed_at=_iso(4 * 60))
        with patch("synaptiq.mcp.server.handle_dead_code", return_value="No dead code detected."):
            result = dispatch_tool("synaptiq_dead_code", {}, MagicMock())

        assert result.rstrip().endswith("---")  # tokens footer is still the last thing

        stripped = strip_metadata(result)

        assert "--- tokens:" not in stripped
        assert "[index: 4m old]" in stripped


# ---------------------------------------------------------------------------
# dispatch_resource — overview only
# ---------------------------------------------------------------------------


class TestDispatchResourceIntegration:
    def test_overview_has_trailer(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_meta(tmp_path / ".synaptiq", last_indexed_at=_iso(4 * 60))
        with patch("synaptiq.mcp.server.get_overview", return_value="Synaptiq Codebase Overview"):
            result = dispatch_resource("synaptiq://overview", MagicMock())
        assert "Synaptiq Codebase Overview" in result
        assert "[index: 4m old]" in result

    def test_dead_code_resource_has_no_trailer(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_meta(tmp_path / ".synaptiq", last_indexed_at=_iso(4 * 60))
        with patch("synaptiq.mcp.server.get_dead_code_list", return_value="No dead code detected."):
            result = dispatch_resource("synaptiq://dead-code", MagicMock())
        assert "No dead code detected." in result
        assert "[index:" not in result

    def test_schema_resource_has_no_trailer(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_meta(tmp_path / ".synaptiq", last_indexed_at=_iso(4 * 60))
        result = dispatch_resource("synaptiq://schema", MagicMock())
        assert "[index:" not in result

    def test_overview_trailer_omitted_when_no_index(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no .synaptiq at all
        with patch("synaptiq.mcp.server.get_overview", return_value="Synaptiq Codebase Overview"):
            result = dispatch_resource("synaptiq://overview", MagicMock())
        assert "[index:" not in result


# ---------------------------------------------------------------------------
# Proxy mode — the primary's dispatch already carries the trailer, so a
# proxy relaying the raw socket response gets it for free. Proven end to
# end over a real Unix socket rather than just asserted in a docstring.
# ---------------------------------------------------------------------------


class TestProxyPassthrough:
    @pytest.mark.asyncio
    async def test_trailer_rides_through_socket_from_primary_dispatch(self, monkeypatch):
        tmpdir = tempfile.mkdtemp()  # short path — AF_UNIX has a ~104 byte limit on macOS
        try:
            repo_dir = Path(tmpdir)
            monkeypatch.chdir(repo_dir)
            _write_meta(repo_dir / ".synaptiq", last_indexed_at=_iso(4 * 60))

            def _primary_dispatch(method: str, params: dict) -> str:
                """Mirrors cli.main._PrimaryRuntime.start()'s dispatch closure:
                calls the real dispatch_tool/dispatch_resource directly."""
                if method == "tool":
                    with patch(
                        "synaptiq.mcp.server.handle_dead_code",
                        return_value="No dead code detected.",
                    ):
                        return dispatch_tool(
                            params.get("name", ""), params.get("arguments", {}), MagicMock()
                        )
                if method == "resource":
                    with patch(
                        "synaptiq.mcp.server.get_overview",
                        return_value="Synaptiq Codebase Overview",
                    ):
                        return dispatch_resource(params.get("uri", ""), MagicMock())
                return "unknown"

            socket_path = repo_dir / "s.sock"
            server = SocketServer(socket_path, _primary_dispatch)
            await server.start()

            client = SocketClient(socket_path)
            await client.connect()
            try:
                tool_result = await client.call_tool("synaptiq_dead_code", {})
                resource_result = await client.read_resource("synaptiq://overview")
            finally:
                await client.close()
                await server.stop()

            # The proxy did nothing but relay these strings verbatim — the
            # trailer already rode along inside the primary's dispatch.
            assert "[index: 4m old]" in tool_result
            assert "[index: 4m old]" in resource_result
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
