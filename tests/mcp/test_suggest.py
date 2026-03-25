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
