"""Tests for REST endpoint linking."""

from synaptiq.core.ingestion.rest_linking import (
    _match_confidence,
    _normalize_url,
    extract_rest_info_from_source,
)


class TestUrlNormalization:
    def test_fastapi_params(self):
        assert _normalize_url("/users/{id}") == "/users/{param}"

    def test_express_params(self):
        assert _normalize_url("/users/:id") == "/users/{param}"

    def test_django_params(self):
        assert _normalize_url("/users/<int:id>") == "/users/{param}"

    def test_trailing_slash_stripped(self):
        assert _normalize_url("/users/") == "/users"

    def test_no_params(self):
        assert _normalize_url("/users") == "/users"


class TestMatchConfidence:
    def test_exact_match(self):
        assert _match_confidence("/users", "/users") == 1.0

    def test_normalized_match(self):
        assert _match_confidence("/users/:id", "/users/{user_id}") == 0.9

    def test_with_host_prefix(self):
        assert _match_confidence("/users", "https://api.com/users") == 1.0

    def test_no_match(self):
        assert _match_confidence("/users", "/items") == 0.0

    def test_prefix_match(self):
        assert _match_confidence("/api/items", "/api/items/123") == 0.5


class TestPythonExtraction:
    def test_fastapi_endpoints(self):
        code = (
            '@app.get("/users/{user_id}")\ndef get_user(): pass\n\n'
            '@app.post("/users")\ndef create(): pass'
        )
        eps, hcs = extract_rest_info_from_source(code, "api.py", "python")
        assert len(eps) == 2
        assert eps[0].http_method == "get"
        assert eps[0].url_pattern == "/users/{user_id}"
        assert eps[1].http_method == "post"

    def test_requests_calls(self):
        code = 'response = requests.get("https://api.com/users")\ndata = requests.post("https://api.com/items")'
        eps, hcs = extract_rest_info_from_source(code, "client.py", "python")
        assert len(hcs) == 2
        assert hcs[0].http_method == "get"
        assert hcs[0].receiver == "requests"
        assert hcs[1].http_method == "post"

    def test_flask_route(self):
        code = '@app.route("/health")\ndef health(): pass'
        eps, hcs = extract_rest_info_from_source(code, "app.py", "python")
        assert len(eps) == 1
        assert eps[0].http_method == "get"  # Default


class TestTypescriptExtraction:
    def test_express_endpoints(self):
        code = 'app.get("/api/items", getItems);\napp.post("/api/items/:id", updateItem);'
        eps, hcs = extract_rest_info_from_source(code, "app.ts", "typescript")
        assert len(eps) == 2
        assert eps[0].url_pattern == "/api/items"
        assert eps[1].url_pattern == "/api/items/:id"

    def test_fetch_calls(self):
        code = 'const res = fetch("https://api.com/data");'
        eps, hcs = extract_rest_info_from_source(code, "client.ts", "typescript")
        assert len(hcs) == 1
        assert hcs[0].receiver == "fetch"

    def test_axios_calls(self):
        code = 'const res = axios.post("https://api.com/users");'
        eps, hcs = extract_rest_info_from_source(code, "client.ts", "typescript")
        assert len(hcs) == 1
        assert hcs[0].http_method == "post"
        assert hcs[0].receiver == "axios"

    def test_skips_middleware(self):
        code = 'app.use("/api", middleware);\napp.all("/health", handler);'
        eps, hcs = extract_rest_info_from_source(code, "app.ts", "typescript")
        assert len(eps) == 0  # use and all are skipped
