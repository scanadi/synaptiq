"""Tests for proxy mode dispatch in the MCP server."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import synaptiq.mcp.server as mcp_server_module
from synaptiq.core.daemon.socket_client import PrimaryPromotedError
from synaptiq.mcp.server import call_tool, read_resource, set_proxy_client


class TestProxyDispatch:
    @pytest.mark.asyncio
    async def test_call_tool_uses_proxy(self):
        """When proxy client is set, call_tool forwards through it."""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = '{"result": "proxied"}'
        set_proxy_client(mock_client)
        try:
            result = await call_tool("synaptiq_query", {"query": "hello"})
            mock_client.call_tool.assert_awaited_once_with("synaptiq_query", {"query": "hello"})
            assert len(result) == 1
            assert result[0].text == '{"result": "proxied"}'
        finally:
            set_proxy_client(None)

    @pytest.mark.asyncio
    async def test_read_resource_uses_proxy(self):
        """When proxy client is set, read_resource forwards through it."""
        mock_client = AsyncMock()
        mock_client.read_resource.return_value = "proxied overview"
        set_proxy_client(mock_client)
        try:
            result = await read_resource("synaptiq://overview")
            mock_client.read_resource.assert_awaited_once_with("synaptiq://overview")
            assert result == "proxied overview"
        finally:
            set_proxy_client(None)

    @pytest.mark.asyncio
    async def test_call_tool_falls_back_to_local_when_no_proxy(
        self, monkeypatch, tmp_path
    ):
        """Without proxy client, tools dispatch locally.

        chdir to an empty directory so local dispatch gets a bare storage
        backend instead of whatever ``.synaptiq`` index the repo running
        the tests happens to contain.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(mcp_server_module, "_storage", None)
        set_proxy_client(None)
        try:
            result = await call_tool("synaptiq_list_repos", {})
            assert len(result) == 1
            # Should return a TextContent with some text (local dispatch)
            assert result[0].type == "text"
            assert isinstance(result[0].text, str)
        finally:
            set_proxy_client(None)

    @pytest.mark.asyncio
    async def test_call_tool_redispatches_locally_after_promotion(
        self, monkeypatch, tmp_path
    ):
        """PrimaryPromotedError from the proxy falls through to local dispatch."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(mcp_server_module, "_storage", None)
        mock_client = AsyncMock()
        mock_client.call_tool.side_effect = PrimaryPromotedError("promoted")
        set_proxy_client(mock_client)
        try:
            result = await call_tool("synaptiq_list_repos", {})
            mock_client.call_tool.assert_awaited_once()
            assert len(result) == 1
            assert result[0].type == "text"
        finally:
            set_proxy_client(None)
