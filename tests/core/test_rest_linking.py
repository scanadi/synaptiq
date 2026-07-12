"""Tests for REST endpoint linking."""

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel, generate_id
from synaptiq.core.ingestion.parser_phase import FileParseData
from synaptiq.core.ingestion.rest_linking import (
    _PY_ENDPOINT_DECORATOR,
    _PY_HTTP_CALL,
    _PY_HTTP_CALL_FSTRING,
    _RB_ENDPOINT,
    _RB_HTTP_CALL,
    _TS_AXIOS,
    _TS_ENDPOINT,
    _TS_FETCH,
    _detect_language,
    _match_confidence,
    _normalize_url,
    _passes_prefilter,
    extract_rest_info_from_source,
    process_rest_linking,
)
from synaptiq.core.parsers.base import EndpointInfo, HttpCallInfo, ParseResult


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


# ---------------------------------------------------------------------------
# Golden end-to-end test (W1.5 safety net)
#
# Builds a synthetic multi-language graph + parse_data_list exercising every
# extractor family (Python FastAPI-decorator + Flask default-GET route +
# `requests` client incl. an f-string call; TypeScript Express routes +
# `fetch`/`axios` clients; Ruby Sinatra block route + Rails hash-rocket
# route + `HTTParty` client) plus the *direct* (non-regex) endpoint/http_call
# channel that exercises the `ep.function_name` branch of
# `_find_endpoint_handler`. process_rest_linking is language-agnostic when
# matching URLs, so a couple of edges deliberately land cross-language
# (e.g. a Python client call resolving against a Ruby route) — that is the
# real, intended "cross-service" behavior the module docstring describes.
#
# The expected edge set below was captured by running process_rest_linking
# against this exact fixture on the pre-optimization implementation
# (commit c1b025b). W1.5's perf changes (indexed match loop, endpoint
# handler dict, regex pre-filter) must reproduce it byte-for-byte.
# ---------------------------------------------------------------------------


# -- Python: FastAPI + Flask endpoints, requests client ---------------------
_GOLDEN_PY_API = (
    "from fastapi import FastAPI\n"
    "\n"
    "app = FastAPI()\n"
    "\n"
    "\n"
    '@app.get("/users/{user_id}")\n'
    "def get_user(user_id: str):\n"
    "    return db.get(user_id)\n"
    "\n"
    "\n"
    '@app.route("/health")\n'
    "def health_check():\n"
    '    return "ok"\n'
)
_GOLDEN_PY_CLIENT = (
    "import requests\n"
    "\n"
    "\n"
    "def fetch_user(uid):\n"
    '    return requests.get(f"https://api.example.com/users/{uid}")\n'
    "\n"
    "\n"
    "def check_health():\n"
    '    return requests.get("https://api.example.com/health")\n'
)

# -- TypeScript: Express endpoints, fetch/axios client -----------------------
_GOLDEN_TS_ROUTES = (
    'import express from "express";\n'
    "\n"
    "const router = express.Router();\n"
    "\n"
    "\n"
    'router.get("/api/items", getItems);\n'
    "\n"
    "function getItems(req, res) {\n"
    "  res.json(items);\n"
    "}\n"
    "\n"
    "\n"
    'router.post("/api/items/:id", updateItem);\n'
    "\n"
    "function updateItem(req, res) {\n"
    "  res.json({ ok: true });\n"
    "}\n"
)
_GOLDEN_TS_CLIENT = (
    "async function loadItems() {\n"
    '  return fetch("https://svc.example.com/api/items");\n'
    "}\n"
    "\n"
    "\n"
    "async function updateItemRemote() {\n"
    '  return axios.post("https://svc.example.com/api/items/:id");\n'
    "}\n"
)

# -- Ruby: Sinatra block route + Rails hash-rocket route, HTTParty client ----
_GOLDEN_RB_ROUTES = (
    'require "sinatra"\n'
    "\n"
    "\n"
    'get "/users/:id" do\n'
    "  show_user\n"
    "end\n"
    "\n"
    "\n"
    'post "/users" => "users#create"\n'
)
_GOLDEN_RB_CLIENT = (
    "class UserClient\n"
    "  def fetch(id)\n"
    '    HTTParty.get("https://api.example.com/users/#{id}")\n'
    "  end\n"
    "\n"
    "  def create_user(payload)\n"
    '    HTTParty.post("https://api.example.com/users")\n'
    "  end\n"
    "end\n"
)


def _add_symbol(
    graph: KnowledgeGraph,
    label: NodeLabel,
    file_path: str,
    name: str,
    start_line: int,
    end_line: int,
    class_name: str = "",
) -> str:
    """Add a symbol node (no DEFINES edge — rest_linking never looks at those)."""
    symbol_name = f"{class_name}.{name}" if label == NodeLabel.METHOD and class_name else name
    node_id = generate_id(label, file_path, symbol_name)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=label,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            class_name=class_name,
        )
    )
    return node_id


def _build_golden_fixture() -> tuple[KnowledgeGraph, list[FileParseData]]:
    graph = KnowledgeGraph()

    _add_symbol(graph, NodeLabel.FUNCTION, "services/user_api.py", "get_user", 7, 8)
    _add_symbol(graph, NodeLabel.FUNCTION, "services/user_api.py", "health_check", 12, 13)
    _add_symbol(graph, NodeLabel.FUNCTION, "services/user_client.py", "fetch_user", 4, 5)
    _add_symbol(graph, NodeLabel.FUNCTION, "services/user_client.py", "check_health", 8, 9)

    _add_symbol(graph, NodeLabel.FUNCTION, "services/itemsRoutes.ts", "getItems", 8, 10)
    _add_symbol(graph, NodeLabel.FUNCTION, "services/itemsRoutes.ts", "updateItem", 15, 17)
    _add_symbol(graph, NodeLabel.FUNCTION, "services/itemsClient.ts", "loadItems", 1, 3)
    _add_symbol(graph, NodeLabel.FUNCTION, "services/itemsClient.ts", "updateItemRemote", 6, 8)

    _add_symbol(
        graph, NodeLabel.METHOD, "app/routes.rb", "show_user", 5, 6, class_name="RoutesApp"
    )
    _add_symbol(
        graph, NodeLabel.METHOD, "app/routes.rb", "create", 10, 12, class_name="UsersController"
    )
    _add_symbol(
        graph,
        NodeLabel.METHOD,
        "app/services/user_client.rb",
        "fetch",
        2,
        4,
        class_name="UserClient",
    )
    _add_symbol(
        graph,
        NodeLabel.METHOD,
        "app/services/user_client.rb",
        "create_user",
        6,
        8,
        class_name="UserClient",
    )

    # -- Direct (non-regex) parse_result.endpoints/http_calls channel -------
    # Exercises the `ep.function_name` branch of _find_endpoint_handler: a
    # same-named "decoy_handler" node sits closer to the decorator line, so
    # only an exact (file_path, function_name) index — not proximity — can
    # resolve this correctly.
    _add_symbol(graph, NodeLabel.FUNCTION, "services/direct.py", "named_handler", 20, 25)
    _add_symbol(graph, NodeLabel.FUNCTION, "services/direct.py", "decoy_handler", 21, 22)
    direct_pr = ParseResult(
        endpoints=[
            EndpointInfo(
                url_pattern="/direct",
                http_method="get",
                function_name="named_handler",
                line=19,
            )
        ],
    )
    _add_symbol(graph, NodeLabel.FUNCTION, "services/direct_client.py", "call_direct", 1, 3)
    direct_client_pr = ParseResult(
        http_calls=[HttpCallInfo(url="/direct", http_method="get", line=2, receiver="requests")],
    )

    parse_data = [
        FileParseData("services/user_api.py", "python", ParseResult(), _GOLDEN_PY_API),
        FileParseData("services/user_client.py", "python", ParseResult(), _GOLDEN_PY_CLIENT),
        FileParseData(
            "services/itemsRoutes.ts", "typescript", ParseResult(), _GOLDEN_TS_ROUTES
        ),
        FileParseData(
            "services/itemsClient.ts", "typescript", ParseResult(), _GOLDEN_TS_CLIENT
        ),
        FileParseData("app/routes.rb", "ruby", ParseResult(), _GOLDEN_RB_ROUTES),
        FileParseData("app/services/user_client.rb", "ruby", ParseResult(), _GOLDEN_RB_CLIENT),
        FileParseData("services/direct.py", "python", direct_pr, ""),
        FileParseData("services/direct_client.py", "python", direct_client_pr, ""),
    ]
    return graph, parse_data


# The exact edge set produced by the pre-optimization implementation
# (source_id, target_id, confidence, properties["rest_link"]), sorted.
_GOLDEN_EDGES = [
    (
        "function:services/direct_client.py:call_direct",
        "function:services/direct.py:named_handler",
        1.0,
        True,
    ),
    (
        "function:services/itemsClient.ts:loadItems",
        "function:services/itemsRoutes.ts:getItems",
        1.0,
        True,
    ),
    (
        "function:services/itemsClient.ts:loadItems",
        "function:services/itemsRoutes.ts:updateItem",
        0.5,
        True,
    ),
    (
        "function:services/itemsClient.ts:updateItemRemote",
        "function:services/itemsRoutes.ts:updateItem",
        1.0,
        True,
    ),
    (
        "function:services/user_client.py:check_health",
        "function:services/user_api.py:health_check",
        1.0,
        True,
    ),
    (
        "function:services/user_client.py:fetch_user",
        "function:services/user_api.py:get_user",
        1.0,
        True,
    ),
    (
        "function:services/user_client.py:fetch_user",
        "method:app/routes.rb:RoutesApp.show_user",
        1.0,
        True,
    ),
    (
        "function:services/user_client.py:fetch_user",
        "method:app/routes.rb:UsersController.create",
        0.5,
        True,
    ),
    (
        "method:app/services/user_client.rb:UserClient.create_user",
        "method:app/routes.rb:UsersController.create",
        1.0,
        True,
    ),
    (
        "method:app/services/user_client.rb:UserClient.fetch",
        "method:app/routes.rb:UsersController.create",
        0.5,
        True,
    ),
]


def _edges_as_sorted_tuples(graph: KnowledgeGraph) -> list[tuple[str, str, float, bool]]:
    rows = [
        (
            rel.source,
            rel.target,
            round(rel.properties.get("confidence", -1.0), 4),
            rel.properties.get("rest_link"),
        )
        for rel in graph.iter_relationships()
    ]
    rows.sort()
    return rows


class TestGoldenRestLinks:
    def test_edge_count_matches_golden_set(self):
        graph, parse_data = _build_golden_fixture()
        edges_created = process_rest_linking(parse_data, graph)
        assert edges_created == len(_GOLDEN_EDGES)

    def test_edge_set_matches_golden_set_byte_for_byte(self):
        graph, parse_data = _build_golden_fixture()
        process_rest_linking(parse_data, graph)
        assert _edges_as_sorted_tuples(graph) == _GOLDEN_EDGES

    def test_no_edges_without_endpoints_or_calls(self):
        # Guards the early-return at the top of process_rest_linking: no
        # endpoints or no http_calls at all must short-circuit to 0 edges.
        graph = KnowledgeGraph()
        _add_symbol(graph, NodeLabel.FUNCTION, "a.py", "solo", 1, 2)
        parse_data = [FileParseData("a.py", "python", ParseResult(), "def solo(): pass\n")]
        assert process_rest_linking(parse_data, graph) == 0


# ---------------------------------------------------------------------------
# Pre-filter conservativeness (W1.5 item c)
#
# The pre-filter in rest_linking.py is only safe to skip per-line regex
# scanning if it is a *superset* of every file any extractor regex could
# possibly match. This test checks that property directly against the real
# compiled regex objects — for every sample line that a regex matches, the
# file containing it must still pass `_passes_prefilter` — over a broad mix
# of fixture content: every source sample already used elsewhere in this
# file (including deliberate non-matching samples) plus the golden
# multi-language fixtures.
# ---------------------------------------------------------------------------

_REGEXES_BY_LANGUAGE: dict[str, tuple] = {
    "python": (_PY_ENDPOINT_DECORATOR, _PY_HTTP_CALL, _PY_HTTP_CALL_FSTRING),
    "typescript": (_TS_ENDPOINT, _TS_FETCH, _TS_AXIOS),
    "javascript": (_TS_ENDPOINT, _TS_FETCH, _TS_AXIOS),
    "ruby": (_RB_ENDPOINT, _RB_HTTP_CALL),
}

# (content, language) samples — the full set of source snippets exercised
# by TestPythonExtraction/TestTypescriptExtraction/TestRubyExtraction(EdgeCases)
# above, plus the golden multi-language fixtures. Deliberately includes
# samples with zero regex matches too (the property under test only claims
# "match implies pass", never "pass implies match").
_PREFILTER_SAMPLES: list[tuple[str, str]] = [
    (
        '@app.get("/users/{user_id}")\ndef get_user(): pass\n\n'
        '@app.post("/users")\ndef create(): pass',
        "python",
    ),
    (
        'response = requests.get("https://api.com/users")\n'
        'data = requests.post("https://api.com/items")',
        "python",
    ),
    ('@app.route("/health")\ndef health(): pass', "python"),
    ('app.get("/api/items", getItems);\napp.post("/api/items/:id", updateItem);', "typescript"),
    ('const res = fetch("https://api.com/data");', "typescript"),
    ('const res = axios.post("https://api.com/users");', "typescript"),
    ('app.use("/api", middleware);\napp.all("/health", handler);', "typescript"),
    ('get "/users/:id" do\n  user\nend', "ruby"),
    ('get "/users/:id" => "users#show"\npost "/users" => "users#create"', "ruby"),
    ('delete "/users/:id", to: "users#destroy"', "ruby"),
    ('namespace :api do\n  get "/health" do\n  end\nend', "ruby"),
    ('HTTParty.get("https://api.com/users")', "ruby"),
    ('Faraday.post("https://api.com/items")', "ruby"),
    ('Net::HTTP.get("http://example.com/data")', "ruby"),
    ('RestClient.delete("https://api.com/users/1")', "ruby"),
    (
        "has_many :posts\nvalidates :name, presence: true\n"
        'render "template"\nbefore_action :authenticate\n',
        "ruby",
    ),
    ('File.delete "/tmp/cache"\nuser.post "/x"', "ruby"),
    ("HTTParty.get(url)\nNet::HTTP.get(uri)", "ruby"),
    (_GOLDEN_PY_API, "python"),
    (_GOLDEN_PY_CLIENT, "python"),
    (_GOLDEN_TS_ROUTES, "typescript"),
    (_GOLDEN_TS_CLIENT, "typescript"),
    (_GOLDEN_RB_ROUTES, "ruby"),
    (_GOLDEN_RB_CLIENT, "ruby"),
    # Prose containing verb words as substrings of unrelated words — no
    # regex should match (no quoted URL literal follows), included as a
    # sanity check that the sample set has real negative cases too.
    ("# widget budget forget compute output\nx = 1\n", "python"),
]


def test_prefilter_conservativeness():
    for content, language in _PREFILTER_SAMPLES:
        regexes = _REGEXES_BY_LANGUAGE[language]
        any_line_matches = any(
            regex.search(line) for line in content.split("\n") for regex in regexes
        )
        if any_line_matches:
            assert _passes_prefilter(content, language), (
                f"pre-filter dropped a file with a real regex match "
                f"(language={language!r}): {content!r}"
            )
