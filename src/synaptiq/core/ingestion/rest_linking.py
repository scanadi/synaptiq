"""REST endpoint linking phase — cross-service call detection.

Detects REST API endpoint definitions (Flask, FastAPI, Express decorators)
and HTTP client call sites (requests, fetch, axios), then creates CALLS
edges between them with URL-pattern-based confidence scoring.

Phase 5b in the ingestion pipeline, runs after call tracing (Phase 5).
"""

from __future__ import annotations

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


def _match_confidence(endpoint_url: str, call_url: str) -> float:
    """Compute matching confidence between an endpoint URL and a call URL.

    Returns:
        1.0 for exact match, 0.9 for normalised match, 0.5 for prefix match,
        0.0 for no match.
    """
    ep = endpoint_url.rstrip("/")
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
    cu = re.sub(r"\{[^}]*\}", "{param}", cu)

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
    for node in graph.iter_nodes():
        if node.start_line > 0 and node.file_path:
            file_node_index.setdefault(node.file_path, []).append(
                (node.start_line, node.end_line, node.id)
            )

    # Match HTTP calls to endpoints.
    edges_created = 0
    seen_edges: set[str] = set()

    for hc, call_file in all_http_calls:
        for ep, ep_file in all_endpoints:
            # Method must match (or be generic).
            if hc.http_method != ep.http_method and hc.http_method not in ("get",):
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
            target_id = _find_endpoint_handler(ep, ep_file, file_node_index, graph)

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
    if file_path.endswith((".js", ".jsx")):
        return "javascript"
    if file_path.endswith((".rb", ".rake", ".ru", ".gemspec", ".rbi")):
        return "ruby"
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
    graph: KnowledgeGraph,
) -> str | None:
    """Find the function/method node that handles an endpoint.

    Endpoints are typically defined by decorating a function, so the
    handler is the first symbol defined on or immediately after the
    decorator line.
    """
    # If we have a function name from the parser, use it directly.
    if ep.function_name:
        for node in graph.iter_nodes():
            if node.file_path == file_path and node.name == ep.function_name:
                return node.id

    # Otherwise find the symbol at or just after the decorator line.
    candidates = []
    for start, _end, node_id in file_node_index.get(file_path, []):
        if ep.line <= start <= ep.line + 5:
            candidates.append((start, node_id))

    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None
