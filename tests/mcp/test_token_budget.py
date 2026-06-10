"""Tests for token counting and budget-aware truncation."""

from synaptiq.mcp.token_budget import count_tokens, truncate_response, wrap_with_metadata


class TestCountTokens:
    def test_basic_count(self):
        assert count_tokens("hello world") == max(1, len("hello world") // 4)

    def test_empty_string_returns_one(self):
        assert count_tokens("") == 1

    def test_long_string(self):
        text = "a" * 400
        assert count_tokens(text) == 100


class TestWrapWithMetadata:
    def test_appends_token_count(self):
        result = wrap_with_metadata("hello")
        assert "--- tokens:" in result
        assert "hello" in result

    def test_metadata_format(self):
        result = wrap_with_metadata("test")
        assert result.endswith("---")


class TestTruncateResponse:
    def test_no_truncation_when_under_budget(self):
        text = "short text"
        result = truncate_response(text, 1000)
        assert result == text

    def test_no_truncation_when_budget_zero(self):
        text = "any text"
        result = truncate_response(text, 0)
        assert result == text

    def test_list_truncation(self):
        items = "\n".join(f"{i}. Item number {i}" for i in range(1, 20))
        result = truncate_response(items, 20)
        assert "showing" in result
        assert "items to fit token budget" in result

    def test_section_truncation(self):
        sections = (
            "=== Section 1 ===\nContent 1\n\n"
            "=== Section 2 ===\nContent 2\n\n"
            "=== Section 3 ===\nContent 3"
        )
        result = truncate_response(sections, 15)
        assert "sections to fit token budget" in result

    def test_hard_truncation_fallback(self):
        text = "a" * 1000
        result = truncate_response(text, 10)
        assert "[... truncated to fit token budget]" in result
        assert len(result) < 1000


class TestToolSchemaBudgets:
    def test_verbose_tools_declare_max_tokens(self):
        from synaptiq.mcp.server import TOOLS

        budgeted = {
            "synaptiq_communities",
            "synaptiq_dead_code",
            "synaptiq_cycles",
            "synaptiq_impact",
            "synaptiq_query",
            "synaptiq_context",
            "synaptiq_export",
        }
        found = {t.name for t in TOOLS if t.name in budgeted}
        assert found == budgeted
        for tool in TOOLS:
            if tool.name in budgeted:
                assert "max_tokens" in tool.inputSchema["properties"], tool.name
