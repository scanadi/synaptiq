"""Tests for the pre-tool routing suggest module."""

from synaptiq.mcp.suggest import suggest_tools


class TestClassification:
    def test_dead_code_question(self):
        suggestions = suggest_tools("is there any dead code?")
        assert suggestions[0].tool_name == "synaptiq_dead_code"

    def test_impact_question(self):
        suggestions = suggest_tools("what is the impact of changing `UserService`?")
        assert suggestions[0].tool_name == "synaptiq_impact"
        assert suggestions[0].arguments.get("symbol") == "UserService"

    def test_caller_question(self):
        suggestions = suggest_tools("who calls `validate_user`?")
        assert suggestions[0].tool_name == "synaptiq_context"
        assert suggestions[0].arguments.get("symbol") == "validate_user"

    def test_coupling_question(self):
        suggestions = suggest_tools("what files are coupled with src/auth.py?")
        assert suggestions[0].tool_name == "synaptiq_coupling"
        assert "auth.py" in suggestions[0].arguments.get("file_path", "")

    def test_cycles_question(self):
        suggestions = suggest_tools("are there any circular dependencies?")
        assert suggestions[0].tool_name == "synaptiq_cycles"

    def test_communities_question(self):
        suggestions = suggest_tools("show me the community clusters")
        assert suggestions[0].tool_name == "synaptiq_communities"

    def test_explain_question(self):
        suggestions = suggest_tools("explain what `UserService` does")
        assert any(s.tool_name == "synaptiq_explain" for s in suggestions)

    def test_file_context_question(self):
        suggestions = suggest_tools("what's in file src/models/user.py?")
        assert suggestions[0].tool_name == "synaptiq_file_context"

    def test_fallback_to_query(self):
        suggestions = suggest_tools("something vague and generic")
        assert suggestions[0].tool_name == "synaptiq_query"

    def test_empty_question(self):
        suggestions = suggest_tools("")
        assert len(suggestions) > 0

    def test_call_path_with_two_symbols(self):
        suggestions = suggest_tools("path from `UserService` to `DatabasePool`")
        assert suggestions[0].tool_name == "synaptiq_call_path"

    def test_test_impact_question(self):
        suggestions = suggest_tools("which tests are affected by `validate_user`?")
        assert suggestions[0].tool_name == "synaptiq_test_impact"


class TestSymbolExtraction:
    def test_extracts_quoted_symbol(self):
        suggestions = suggest_tools("what calls `process_data`?")
        assert any(
            s.arguments.get("symbol") == "process_data" for s in suggestions
        )

    def test_extracts_camel_case(self):
        suggestions = suggest_tools("explain UserService")
        assert any(
            s.arguments.get("symbol") == "UserService"
            or s.arguments.get("query") == "UserService"
            for s in suggestions
        )

    def test_extracts_snake_case(self):
        suggestions = suggest_tools("impact of validate_user_input")
        assert any(
            s.arguments.get("symbol") == "validate_user_input" for s in suggestions
        )

    def test_what_breaks_with_lower_camel_case(self):
        """The flagship workflow phrasing must route to impact, not query."""
        suggestions = suggest_tools("what breaks if I change enqueueBillingCountSync")
        assert suggestions[0].tool_name == "synaptiq_impact"
        assert suggestions[0].arguments.get("symbol") == "enqueueBillingCountSync"

    def test_impact_without_symbol_still_recommends_impact(self):
        suggestions = suggest_tools("what breaks if I change this?")
        tool_names = [s.tool_name for s in suggestions]
        assert "synaptiq_impact" in tool_names

    def test_extracts_pascal_case_with_acronym_and_digits(self):
        suggestions = suggest_tools("what is the impact of changing KPIData?")
        assert suggestions[0].tool_name == "synaptiq_impact"
        assert suggestions[0].arguments.get("symbol") == "KPIData"

        suggestions = suggest_tools("impact of Base64Encoder")
        assert suggestions[0].arguments.get("symbol") == "Base64Encoder"

    def test_pure_acronym_not_extracted_unless_quoted(self):
        suggestions = suggest_tools("what is the impact of USA on the economy?")
        assert all(s.arguments.get("symbol") != "USA" for s in suggestions)

        suggestions = suggest_tools("what is the impact of `USA`?")
        assert suggestions[0].arguments.get("symbol") == "USA"
