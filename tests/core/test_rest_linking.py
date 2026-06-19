"""Tests for REST endpoint linking."""

from synaptiq.core.ingestion.rest_linking import (
    _detect_language,
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


class TestRubyExtraction:
    def test_sinatra_route(self):
        code = 'get "/users/:id" do\n  user\nend'
        eps, hcs = extract_rest_info_from_source(code, "app.rb", "ruby")
        assert len(eps) == 1
        assert eps[0].http_method == "get"
        assert eps[0].url_pattern == "/users/:id"
        assert eps[0].line == 1

    def test_rails_route_hash_rocket(self):
        code = 'get "/users/:id" => "users#show"\npost "/users" => "users#create"'
        eps, hcs = extract_rest_info_from_source(code, "routes.rb", "ruby")
        assert len(eps) == 2
        assert eps[0].http_method == "get"
        assert eps[0].url_pattern == "/users/:id"
        assert eps[1].http_method == "post"
        assert eps[1].url_pattern == "/users"

    def test_rails_route_to_option(self):
        code = 'delete "/users/:id", to: "users#destroy"'
        eps, hcs = extract_rest_info_from_source(code, "routes.rb", "ruby")
        assert len(eps) == 1
        assert eps[0].http_method == "delete"
        assert eps[0].url_pattern == "/users/:id"

    def test_indented_route(self):
        code = 'namespace :api do\n  get "/health" do\n  end\nend'
        eps, hcs = extract_rest_info_from_source(code, "app.rb", "ruby")
        assert len(eps) == 1
        assert eps[0].url_pattern == "/health"
        assert eps[0].line == 2

    def test_httparty_call(self):
        code = 'HTTParty.get("https://api.com/users")'
        eps, hcs = extract_rest_info_from_source(code, "client.rb", "ruby")
        assert len(hcs) == 1
        assert hcs[0].http_method == "get"
        assert hcs[0].receiver == "HTTParty"
        assert hcs[0].url == "https://api.com/users"

    def test_faraday_call(self):
        code = 'Faraday.post("https://api.com/items")'
        eps, hcs = extract_rest_info_from_source(code, "client.rb", "ruby")
        assert len(hcs) == 1
        assert hcs[0].http_method == "post"
        assert hcs[0].receiver == "Faraday"

    def test_net_http_call(self):
        code = 'Net::HTTP.get("http://example.com/data")'
        eps, hcs = extract_rest_info_from_source(code, "client.rb", "ruby")
        assert len(hcs) == 1
        assert hcs[0].http_method == "get"
        assert hcs[0].receiver == "Net::HTTP"

    def test_rest_client_call(self):
        code = 'RestClient.delete("https://api.com/users/1")'
        eps, hcs = extract_rest_info_from_source(code, "client.rb", "ruby")
        assert len(hcs) == 1
        assert hcs[0].http_method == "delete"
        assert hcs[0].receiver == "RestClient"


class TestRubyExtractionEdgeCases:
    def test_non_route_dsl_not_misdetected(self):
        # Model/controller DSL methods that take strings/symbols must not be
        # mistaken for routes.
        code = (
            "has_many :posts\n"
            "validates :name, presence: true\n"
            'render "template"\n'
            "before_action :authenticate\n"
        )
        eps, hcs = extract_rest_info_from_source(code, "model.rb", "ruby")
        assert eps == []

    def test_method_call_with_receiver_not_a_route(self):
        # ``user.get`` / ``File.delete`` have a receiver, so the verb is not at
        # the start of the line — not a route definition.
        code = 'File.delete "/tmp/cache"\nuser.post "/x"'
        eps, hcs = extract_rest_info_from_source(code, "app.rb", "ruby")
        assert eps == []

    def test_non_literal_http_url_ignored(self):
        # A non-string URL (variable / URI object) is not captured.
        code = "HTTParty.get(url)\nNet::HTTP.get(uri)"
        eps, hcs = extract_rest_info_from_source(code, "client.rb", "ruby")
        assert hcs == []


class TestDetectLanguage:
    def test_ruby_extensions(self):
        assert _detect_language("app.rb") == "ruby"
        assert _detect_language("Rakefile.rake") == "ruby"
        assert _detect_language("config.ru") == "ruby"
        assert _detect_language("foo.gemspec") == "ruby"
        assert _detect_language("sig.rbi") == "ruby"

    def test_other_languages(self):
        assert _detect_language("a.py") == "python"
        assert _detect_language("a.ts") == "typescript"
        assert _detect_language("a.js") == "javascript"
        assert _detect_language("a.go") is None
