"""REST endpoint linking phase — cross-service call detection.

Detects REST API endpoint definitions (Flask, FastAPI, Express decorators)
and HTTP client call sites (requests, fetch, axios), then creates CALLS
edges between them with URL-pattern-based confidence scoring.

Phase 5b in the ingestion pipeline, runs after call tracing (Phase 5).
"""

from __future__ import annotations

import bisect
import logging
import re

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphRelationship, RelType
from synaptiq.core.parsers.base import EndpointInfo, HttpCallInfo

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Decorator-based endpoint extraction (post-parse)
# ------------------------------------------------------------------

# Python: @app.get("/users/{id}"), @router.post("/items"), @app.route("/path", methods=["GET"])
_PY_ENDPOINT_DECORATOR = re.compile(
    r"@\w+\.(get|post|put|delete|patch|route)\s*\(\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)

# Python HTTP client calls: requests.get("url"), httpx.post("url"), aiohttp...
_PY_HTTP_CALL = re.compile(
    r"\b(requests|httpx|aiohttp|urllib3)\.(get|post|put|delete|patch|head|options)"
    r"\s*\(\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_PY_HTTP_CALL_FSTRING = re.compile(
    r"\b(requests|httpx)\.(get|post|put|delete|patch)\s*\(\s*f[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)

# TypeScript/JS: app.get("/path", handler), router.post("/path", ...)
_TS_ENDPOINT = re.compile(
    r"\b\w+\.(get|post|put|delete|patch|all|use)\s*\(\s*[\"'`]([^\"'`]+)[\"'`]",
    re.IGNORECASE,
)

# TypeScript/JS HTTP client calls: fetch("url"), axios.get("url")
_TS_FETCH = re.compile(
    r"\bfetch\s*\(\s*[\"'`]([^\"'`]+)[\"'`]",
    re.IGNORECASE,
)
_TS_AXIOS = re.compile(
    r"\baxios\.(get|post|put|delete|patch)\s*\(\s*[\"'`]([^\"'`]+)[\"'`]",
    re.IGNORECASE,
)

# Ruby route DSL: Sinatra ``get "/x" do`` / Rails ``get "/x" => "c#a"`` /
# ``get "/x", to: "c#a"``.  The verb must open the (optionally indented) line so
# bare method calls like ``render "tpl"`` or ``has_many :x`` are not misread as
# routes.
_RB_ENDPOINT = re.compile(
    r"^\s*(get|post|put|delete|patch|head|options)\s+[\"']([^\"']+)[\"']",
)

# Ruby HTTP client calls: HTTParty.get("url"), Faraday.get("url"),
# RestClient.post("url"), Net::HTTP.get("url") — only string-literal URLs.
_RB_HTTP_CALL = re.compile(
    r"\b(HTTParty|Faraday|RestClient|Typhoeus|Net::HTTP)"
    r"\.(get|post|put|delete|patch|head)\s*\(?\s*[\"']([^\"']+)[\"']",
)

# Go route registration: net/http ``mux.HandleFunc("/x", h)`` / ``http.Handle``
# and the gin/echo/chi/gorilla router DSL ``r.GET("/x", h)`` / ``r.Get("/x", h)``.
# Group 1 is the receiver (used to reject ``http.Get`` — a *client* call, see
# below — while keeping ``http.HandleFunc``), group 2 the verb, group 3 the
# path. Go string literals are double-quoted or raw-backtick-quoted (single
# quotes are runes, never URLs).
_GO_ENDPOINT = re.compile(
    r'\b(\w+)\.(Get|Post|Put|Delete|Patch|Head|Options|HandleFunc|Handle)'
    r'\s*\(\s*["`]([^"`]+)["`]',
    re.IGNORECASE,
)

# Go net/http client helpers: ``http.Get("url")`` / ``http.Post(...)`` /
# ``http.Head`` / ``http.PostForm``. Case-sensitive — the stdlib API is
# capitalised.
_GO_HTTP_CALL = re.compile(
    r'\bhttp\.(Get|Post|Head|PostForm)\s*\(\s*["`]([^"`]+)["`]',
)

# Go request builder: ``http.NewRequest("GET", "url", body)`` and
# ``http.NewRequestWithContext(ctx, "GET", "url", body)`` — the optional leading
# ctx arg is skipped by ``(?:[^,]+,\s*)?``. Group 1 is the method, group 2 the
# URL.
_GO_HTTP_NEWREQUEST = re.compile(
    r'\bhttp\.NewRequest\w*\s*\(\s*(?:[^,]+,\s*)?'
    r'["`](GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)["`]\s*,\s*["`]([^"`]+)["`]',
)


# ------------------------------------------------------------------
# Cheap pre-filter — skip the per-line regex pass entirely for files that
# cannot possibly contain a REST endpoint/HTTP-call match for their
# language.
#
# Soundness requirement: a token set is safe as long as it is a *necessary*
# condition for its regex(es) to match — i.e. every string any one of the
# regexes below can match must contain at least one of that language's
# tokens literally. The set does not need to be tight/minimal; extra false
# positives (files that pass the filter but produce no matches) are fine —
# "when in doubt, include the file". `test_prefilter_conservativeness` in
# tests/core/test_rest_linking.py checks this holds for every extractor
# regex against a broad sample of fixture lines.
#
# Derivation, regex by regex:
#   Python  — `_PY_ENDPOINT_DECORATOR` always requires a literal "@" right
#             before the receiver (`@app.get(...)`) — "@" alone is a sound,
#             cheap anchor regardless of which verb follows.
#             `_PY_HTTP_CALL` / `_PY_HTTP_CALL_FSTRING` always require one
#             of the literal receivers `requests`/`httpx`/`aiohttp`/
#             `urllib3` right before the dot; the f-string variant's
#             receiver alternation (`requests`/`httpx`) is already a subset
#             of the plain variant's, so it needs no extra token.
#   TS/JS   — `_TS_ENDPOINT` matches `<identifier>.<verb>(` where the
#             receiver is an arbitrary identifier (no distinctive anchor is
#             possible) and `<verb>` is a closed 7-word set — every one of
#             get/post/put/delete/patch/all/use must be a token.
#             `_TS_FETCH` always requires the literal `fetch`.
#             `_TS_AXIOS` always requires the literal `axios` (its verb
#             alternation is already covered by the endpoint set above).
#   Ruby    — `_RB_ENDPOINT` matches `^\s*<verb>\s+["']` — again a closed
#             7-word verb set with no other anchor:
#             get/post/put/delete/patch/head/options.
#             `_RB_HTTP_CALL` always requires one of the literal receivers
#             `HTTParty`/`Faraday`/`RestClient`/`Typhoeus`/`Net::HTTP`.
#   Go      — `_GO_ENDPOINT` matches `<identifier>.<verb>(` where `<verb>` is a
#             closed set get/post/put/delete/patch/head/options/handlefunc/
#             handle (no distinctive receiver anchor is possible — routers are
#             named `r`/`router`/`mux`/`e`/…). Every verb must be a token;
#             `handle` is a literal substring of `handlefunc`, so one token
#             covers both. `_GO_HTTP_CALL` and `_GO_HTTP_NEWREQUEST` always
#             require the literal receiver `http` right before the dot, so the
#             single token `http` anchors both.
#
# The check is done against `content.lower()` with lower-cased tokens. Both
# Ruby regexes and the Go client regexes are case-sensitive in their real
# matching, so lower-casing only ever *adds* candidate files versus the
# exact-case regex — never drops one — which keeps the filter sound.
# ------------------------------------------------------------------

_PY_PREFILTER_TOKENS = ("@", "requests", "httpx", "aiohttp", "urllib3")
_TS_PREFILTER_TOKENS = (
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "all",
    "use",
    "fetch",
    "axios",
)
_RB_PREFILTER_TOKENS = (
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "head",
    "options",
    "httparty",
    "faraday",
    "restclient",
    "typhoeus",
    "net::http",
)
_GO_PREFILTER_TOKENS = (
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "head",
    "options",
    "handle",  # substring of "handlefunc" too
    "http",
)

_PREFILTER_TOKENS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "python": _PY_PREFILTER_TOKENS,
    "typescript": _TS_PREFILTER_TOKENS,
    "javascript": _TS_PREFILTER_TOKENS,
    "ruby": _RB_PREFILTER_TOKENS,
    "go": _GO_PREFILTER_TOKENS,
}


def _passes_prefilter(content: str, language: str) -> bool:
    """Return False only when *content* provably cannot match any of
    *language*'s extractor regexes (see derivation above)."""
    tokens = _PREFILTER_TOKENS_BY_LANGUAGE.get(language)
    if tokens is None:
        return True
    lowered = content.lower()
    return any(tok in lowered for tok in tokens)


def extract_rest_info_from_source(
    content: str, file_path: str, language: str
) -> tuple[list[EndpointInfo], list[HttpCallInfo]]:
    """Extract REST endpoints and HTTP calls from raw source content.

    This runs as a supplementary pass after the main tree-sitter parse,
    using regex to catch decorator patterns that the AST parser may
    not explicitly extract.
    """
    endpoints: list[EndpointInfo] = []
    http_calls: list[HttpCallInfo] = []

    if not _passes_prefilter(content, language):
        return endpoints, http_calls

    lines = content.split("\n")

    if language in ("python",):
        for i, line in enumerate(lines, 1):
            # Endpoint decorators.
            for m in _PY_ENDPOINT_DECORATOR.finditer(line):
                method = m.group(1).lower()
                if method == "route":
                    method = "get"  # Default route method
                endpoints.append(
                    EndpointInfo(
                        url_pattern=m.group(2),
                        http_method=method,
                        function_name="",  # Will be resolved from symbols
                        line=i,
                    )
                )
            # HTTP client calls.
            for m in _PY_HTTP_CALL.finditer(line):
                http_calls.append(
                    HttpCallInfo(
                        url=m.group(3),
                        http_method=m.group(2).lower(),
                        line=i,
                        receiver=m.group(1),
                    )
                )
            for m in _PY_HTTP_CALL_FSTRING.finditer(line):
                http_calls.append(
                    HttpCallInfo(
                        url=m.group(3),
                        http_method=m.group(2).lower(),
                        line=i,
                        receiver=m.group(1),
                    )
                )

    elif language in ("typescript", "javascript"):
        for i, line in enumerate(lines, 1):
            # Express-style endpoints.
            for m in _TS_ENDPOINT.finditer(line):
                method = m.group(1).lower()
                if method in ("use", "all"):
                    continue  # Skip middleware
                endpoints.append(
                    EndpointInfo(
                        url_pattern=m.group(2),
                        http_method=method,
                        function_name="",
                        line=i,
                    )
                )
            # fetch() calls.
            for m in _TS_FETCH.finditer(line):
                http_calls.append(
                    HttpCallInfo(url=m.group(1), http_method="get", line=i, receiver="fetch")
                )
            # axios calls.
            for m in _TS_AXIOS.finditer(line):
                http_calls.append(
                    HttpCallInfo(
                        url=m.group(2),
                        http_method=m.group(1).lower(),
                        line=i,
                        receiver="axios",
                    )
                )

    elif language == "ruby":
        for i, line in enumerate(lines, 1):
            # Sinatra/Rails route DSL.
            for m in _RB_ENDPOINT.finditer(line):
                endpoints.append(
                    EndpointInfo(
                        url_pattern=m.group(2),
                        http_method=m.group(1).lower(),
                        function_name="",
                        line=i,
                    )
                )
            # HTTP client calls.
            for m in _RB_HTTP_CALL.finditer(line):
                http_calls.append(
                    HttpCallInfo(
                        url=m.group(3),
                        http_method=m.group(2).lower(),
                        line=i,
                        receiver=m.group(1),
                    )
                )

    elif language == "go":
        for i, line in enumerate(lines, 1):
            # Route registration (net/http HandleFunc/Handle + gin/echo/chi
            # router verbs).
            for m in _GO_ENDPOINT.finditer(line):
                receiver = m.group(1).lower()
                verb = m.group(2).lower()
                # ``http.Get``/``http.Post``/… is a client call (below), not a
                # route; only ``http.HandleFunc``/``http.Handle`` are routes.
                if receiver == "http" and verb not in ("handlefunc", "handle"):
                    continue
                # HandleFunc/Handle are method-agnostic — default to GET, which
                # process_rest_linking treats as ambiguous (matches any method).
                method = "get" if verb in ("handlefunc", "handle") else verb
                endpoints.append(
                    EndpointInfo(
                        url_pattern=m.group(3),
                        http_method=method,
                        function_name="",
                        line=i,
                    )
                )
            # net/http client helpers.
            for m in _GO_HTTP_CALL.finditer(line):
                method = m.group(1).lower()
                if method == "postform":
                    method = "post"
                http_calls.append(
                    HttpCallInfo(url=m.group(2), http_method=method, line=i, receiver="http")
                )
            # http.NewRequest / NewRequestWithContext.
            for m in _GO_HTTP_NEWREQUEST.finditer(line):
                http_calls.append(
                    HttpCallInfo(
                        url=m.group(2),
                        http_method=m.group(1).lower(),
                        line=i,
                        receiver="http",
                    )
                )

    return endpoints, http_calls


# ------------------------------------------------------------------
# URL pattern matching
# ------------------------------------------------------------------

# Normalise path parameters: {id}, :id, <id> → {param}
_PARAM_PATTERNS = [
    re.compile(r"\{[^}]+\}"),  # Flask/FastAPI: {id}
    re.compile(r":([a-zA-Z_]\w*)"),  # Express: :id
    re.compile(r"<[^>]+>"),  # Django: <int:id>
]


def _normalize_url(url: str) -> str:
    """Normalise a URL pattern by replacing all parameter placeholders with {param}."""
    result = url.rstrip("/")
    for pat in _PARAM_PATTERNS:
        result = pat.sub("{param}", result)
    return result


def _process_call_url(call_url: str) -> str:
    """Apply the same host-stripping/interpolation-marker cleanup to a call
    URL that :func:`_match_confidence` uses, so index lookups
    (:func:`_candidate_endpoints`) key off exactly the same string
    :func:`_match_confidence` will itself compare against.
    """
    cu = call_url.rstrip("/")

    # Remove any base URL / host prefix from the call URL.
    if "://" in cu:
        # Strip protocol + host: https://host.com/path → /path
        parts = cu.split("://", 1)
        if len(parts) == 2:
            slash_idx = parts[1].find("/")
            if slash_idx >= 0:
                cu = parts[1][slash_idx:]
            else:
                cu = "/"

    # Remove f-string interpolation markers.
    return re.sub(r"\{[^}]*\}", "{param}", cu)


def _match_confidence(endpoint_url: str, call_url: str) -> float:
    """Compute matching confidence between an endpoint URL and a call URL.

    Returns:
        1.0 for exact match, 0.9 for normalised match, 0.5 for prefix match,
        0.0 for no match.
    """
    ep = endpoint_url.rstrip("/")
    cu = _process_call_url(call_url)

    if ep == cu:
        return 1.0

    norm_ep = _normalize_url(ep)
    norm_cu = _normalize_url(cu)

    if norm_ep == norm_cu:
        return 0.9

    # Prefix match.
    if norm_cu.startswith(norm_ep) or norm_ep.startswith(norm_cu):
        return 0.5

    return 0.0


# ------------------------------------------------------------------
# Endpoint indexing for the match loop
#
# `_match_confidence` only returns a positive score in three cases, all
# expressible in terms of the *normalised* endpoint/call URLs:
#   1.0  ep            == cu            (raw-equal ⟹ normalised-equal too,
#                                         since normalisation is a pure
#                                         function of the string)
#   0.9  norm_ep        == norm_cu
#   0.5  norm_cu.startswith(norm_ep)  or  norm_ep.startswith(norm_cu)
#
# So for a fixed call, *every* endpoint that can score > 0 has a normalised
# URL that is either a prefix of the call's normalised URL, or has the
# call's normalised URL as a prefix of itself. `_candidate_endpoints` finds
# exactly that set without scanning every endpoint:
#
#   - endpoint-is-prefix-of-call: every prefix of the call's normalised URL
#     (there are only len(url) of them) is tried as an exact key into the
#     `normalized_url -> [endpoints]` bucket built by
#     `_build_endpoint_index`. Bounded by URL length, not endpoint count.
#   - call-is-prefix-of-endpoint: bucket keys are kept sorted, so every key
#     that starts with the call's normalised URL forms one contiguous range
#     starting at `bisect_left`. Scanning forward and stopping at the first
#     key that no longer starts with the call's URL is sound: sorted order
#     guarantees no later key can start with it either once one fails.
#
# The result is a superset-safe (never-miss) candidate list; the real
# `_match_confidence` call in the caller remains the authority on the
# actual score. HTTP-method bucketing was deliberately left out — the
# "get" call is ambiguous and must be checked against endpoints of *any*
# method (see process_rest_linking), so a compound (method, url) key would
# still need to fan out across methods for that common case. Method
# filtering stays a cheap inline check on the (already narrow) candidate
# list, exactly as it was before this index existed.
# ------------------------------------------------------------------


def _build_endpoint_index(
    all_endpoints: list[tuple[EndpointInfo, str]],
) -> tuple[dict[str, list[tuple[EndpointInfo, str]]], list[str]]:
    """Bucket endpoints by normalised URL. Returns the bucket dict plus its
    sorted key list (used by :func:`_candidate_endpoints` for the
    call-is-prefix-of-endpoint direction).
    """
    by_norm_url: dict[str, list[tuple[EndpointInfo, str]]] = {}
    for ep, ep_file in all_endpoints:
        key = _normalize_url(ep.url_pattern.rstrip("/"))
        by_norm_url.setdefault(key, []).append((ep, ep_file))
    return by_norm_url, sorted(by_norm_url)


def _candidate_endpoints(
    call_url: str,
    by_norm_url: dict[str, list[tuple[EndpointInfo, str]]],
    sorted_keys: list[str],
) -> list[tuple[EndpointInfo, str]]:
    """Return every endpoint that could score > 0 against *call_url*,
    without scanning the full endpoint list. See the module comment above
    for the derivation."""
    norm_cu = _normalize_url(_process_call_url(call_url))

    seen: set[int] = set()
    candidates: list[tuple[EndpointInfo, str]] = []

    def _add_bucket(key: str) -> None:
        for pair in by_norm_url.get(key, ()):
            pair_id = id(pair)
            if pair_id not in seen:
                seen.add(pair_id)
                candidates.append(pair)

    # Direction 1: endpoint's normalised URL is a prefix of the call's
    # (covers the 1.0 and 0.9 tiers too — the full-length "prefix" is an
    # exact match).
    for i in range(len(norm_cu) + 1):
        prefix = norm_cu[:i]
        if prefix in by_norm_url:
            _add_bucket(prefix)

    # Direction 2: call's normalised URL is a prefix of the endpoint's.
    idx = bisect.bisect_left(sorted_keys, norm_cu)
    for key in sorted_keys[idx:]:
        if not key.startswith(norm_cu):
            break
        _add_bucket(key)

    return candidates


# ------------------------------------------------------------------
# Phase entry point
# ------------------------------------------------------------------


def process_rest_linking(
    parse_data_list: list,
    graph: KnowledgeGraph,
) -> int:
    """Create CALLS edges between REST endpoints and HTTP client calls.

    Parameters
    ----------
    parse_data_list:
        List of FileParseData objects from the parser phase.
    graph:
        The knowledge graph to add edges to.

    Returns
    -------
    int
        Number of REST linking edges created.
    """
    # Collect endpoints and HTTP calls from all files.
    all_endpoints: list[tuple[EndpointInfo, str]] = []  # (endpoint, file_path)
    all_http_calls: list[tuple[HttpCallInfo, str]] = []  # (call, file_path)

    for fpd in parse_data_list:
        file_path = fpd.file_path
        language = _detect_language(file_path)
        if not language:
            continue

        # Use endpoints/http_calls from ParseResult if populated.
        if hasattr(fpd, "parse_result"):
            pr = fpd.parse_result
            if hasattr(pr, "endpoints"):
                for ep in pr.endpoints:
                    all_endpoints.append((ep, file_path))
            if hasattr(pr, "http_calls"):
                for hc in pr.http_calls:
                    all_http_calls.append((hc, file_path))

        # Supplementary regex-based extraction from source.
        if hasattr(fpd, "content") and fpd.content:
            eps, hcs = extract_rest_info_from_source(fpd.content, file_path, language)
            for ep in eps:
                all_endpoints.append((ep, file_path))
            for hc in hcs:
                all_http_calls.append((hc, file_path))

    if not all_endpoints or not all_http_calls:
        return 0

    # Build per-file node index to avoid linear scans.
    file_node_index: dict[str, list[tuple[int, int, str]]] = {}
    # (file_path, symbol_name) -> node_id, first-occurrence-wins in
    # graph.iter_nodes() order — replaces the full node scan that used to
    # run inside _find_endpoint_handler for every candidate match.
    handler_by_file_name: dict[tuple[str, str], str] = {}
    for node in graph.iter_nodes():
        if node.start_line > 0 and node.file_path:
            file_node_index.setdefault(node.file_path, []).append(
                (node.start_line, node.end_line, node.id)
            )
        key = (node.file_path, node.name)
        if key not in handler_by_file_name:
            handler_by_file_name[key] = node.id

    # Bucket endpoints by normalised URL once, for O(1)-ish candidate
    # lookup per call instead of scanning every endpoint (see the
    # _candidate_endpoints derivation comment above).
    by_norm_url, sorted_norm_urls = _build_endpoint_index(all_endpoints)

    # Match HTTP calls to endpoints.
    edges_created = 0
    seen_edges: set[str] = set()

    for hc, call_file in all_http_calls:
        for ep, ep_file in _candidate_endpoints(hc.url, by_norm_url, sorted_norm_urls):
            # HTTP method must match. A "get" call is treated as ambiguous
            # (fetch() defaults to GET even when options specify another
            # method, which the regex does not capture) and may match any
            # endpoint — but without the method bonus.
            if hc.http_method != ep.http_method:
                if hc.http_method != "get":
                    continue
                method_bonus = 0.0
            else:
                method_bonus = 0.1

            url_conf = _match_confidence(ep.url_pattern, hc.url)
            if url_conf <= 0:
                continue

            confidence = min(url_conf + method_bonus, 1.0)

            # Find the source symbol (caller) containing the HTTP call.
            source_id = _find_symbol_at_line(hc.line, call_file, file_node_index)
            # Find the target symbol (endpoint handler).
            target_id = _find_endpoint_handler(ep, ep_file, file_node_index, handler_by_file_name)

            if not source_id or not target_id:
                continue
            if source_id == target_id:
                continue

            edge_key = f"{source_id}->{target_id}"
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            graph.add_relationship(
                GraphRelationship(
                    id=f"rest_calls:{source_id}->{target_id}",
                    type=RelType.CALLS,
                    source=source_id,
                    target=target_id,
                    properties={"confidence": confidence, "rest_link": True},
                )
            )
            edges_created += 1

    logger.info(
        "REST linking: %d endpoints, %d HTTP calls, %d edges created",
        len(all_endpoints),
        len(all_http_calls),
        edges_created,
    )
    return edges_created


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _detect_language(file_path: str) -> str | None:
    """Detect language from file extension."""
    if file_path.endswith(".py"):
        return "python"
    if file_path.endswith((".ts", ".tsx")):
        return "typescript"
    if file_path.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return "javascript"
    if file_path.endswith((".rb", ".rake", ".ru", ".gemspec", ".rbi")):
        return "ruby"
    if file_path.endswith(".go"):
        return "go"
    return None


def _find_symbol_at_line(
    line: int,
    file_path: str,
    file_node_index: dict[str, list[tuple[int, int, str]]],
) -> str | None:
    """Find the symbol node whose line range contains *line*."""
    for start, end, node_id in file_node_index.get(file_path, []):
        if start <= line <= end:
            return node_id
    return None


def _find_endpoint_handler(
    ep: EndpointInfo,
    file_path: str,
    file_node_index: dict[str, list[tuple[int, int, str]]],
    handler_by_file_name: dict[tuple[str, str], str],
) -> str | None:
    """Find the function/method node that handles an endpoint.

    Endpoints are typically defined by decorating a function, so the
    handler is the first symbol defined on or immediately after the
    decorator line.
    """
    # If we have a function name from the parser, use it directly. Looked
    # up in the pre-built (file_path, name) -> node_id index instead of
    # scanning graph.iter_nodes() per call — same first-occurrence-wins
    # result, since the index was built in the same iteration order.
    if ep.function_name:
        node_id = handler_by_file_name.get((file_path, ep.function_name))
        if node_id is not None:
            return node_id

    # Otherwise find the symbol at or just after the decorator line.
    candidates = []
    for start, _end, node_id in file_node_index.get(file_path, []):
        if ep.line <= start <= ep.line + 5:
            candidates.append((start, node_id))

    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None
