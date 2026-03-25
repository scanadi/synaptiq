"""Tests for the MCP secret scanner."""

from synaptiq.mcp.secret_scanner import redact, scan


class TestScan:
    def test_detects_aws_key(self):
        matches = scan("key = AKIAIOSFODNN7EXAMPLE1")
        assert len(matches) == 1
        assert matches[0].secret_type == "AWS_KEY"

    def test_detects_jwt(self):
        token = (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
            ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )
        matches = scan(f"auth = {token}")
        assert any(m.secret_type == "JWT_TOKEN" for m in matches)

    def test_detects_private_key(self):
        matches = scan("-----BEGIN RSA PRIVATE KEY-----")
        assert len(matches) == 1
        assert matches[0].secret_type == "PRIVATE_KEY"

    def test_detects_connection_string(self):
        matches = scan("db = postgresql://user:pass@host:5432/mydb")
        assert len(matches) == 1
        assert matches[0].secret_type == "CONNECTION_STRING"

    def test_detects_github_token(self):
        token = "ghp_" + "A" * 36
        matches = scan(f"token = {token}")
        assert any(m.secret_type == "GITHUB_TOKEN" for m in matches)

    def test_detects_openai_key(self):
        key = "sk-" + "a" * 40
        matches = scan(f"key = {key}")
        assert any(m.secret_type == "OPENAI_KEY" for m in matches)

    def test_detects_password_assignment(self):
        matches = scan("password = 'mysecretpassword123'")
        assert any(m.secret_type == "PASSWORD_ASSIGNMENT" for m in matches)

    def test_no_false_positive_on_normal_code(self):
        code = 'def hello():\n    return "world"\n\nclass User:\n    pass'
        matches = scan(code)
        assert len(matches) == 0

    def test_no_false_positive_on_short_strings(self):
        matches = scan("name = 'hello'")
        assert len(matches) == 0


class TestRedact:
    def test_redacts_single_secret(self):
        text = "key = AKIAIOSFODNN7EXAMPLE1"
        result, count = redact(text)
        assert count == 1
        assert "AKIAIOSFODNN7EXAMPLE1" not in result
        assert "[REDACTED: AWS_KEY]" in result

    def test_redacts_multiple_secrets(self):
        text = "aws = AKIAIOSFODNN7EXAMPLE1\ndb = postgresql://u:p@host:5432/db"
        result, count = redact(text)
        assert count == 2
        assert "[REDACTED: AWS_KEY]" in result
        assert "[REDACTED: CONNECTION_STRING]" in result

    def test_no_redaction_on_clean_text(self):
        text = "def process(data): return data"
        result, count = redact(text)
        assert count == 0
        assert result == text

    def test_handles_empty_string(self):
        result, count = redact("")
        assert count == 0
        assert result == ""
