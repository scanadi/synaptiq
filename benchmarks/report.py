"""Benchmark report generator.

Parses pytest output and generates a markdown summary table.
Run with: ``uv run python benchmarks/report.py < pytest-output.txt``
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """A single benchmark measurement."""

    name: str
    metrics: dict[str, str] = field(default_factory=dict)


def parse_output(text: str) -> list[BenchmarkResult]:
    """Parse pytest -s output for benchmark results."""
    results: list[BenchmarkResult] = []
    current: BenchmarkResult | None = None

    for line in text.split("\n"):
        # Detect section headers like [hybrid_search] or [query vs raw].
        header_match = re.match(r"^\[(.+?)\]$", line.strip())
        if header_match:
            current = BenchmarkResult(name=header_match.group(1))
            results.append(current)
            continue

        # Detect key: value lines.
        kv_match = re.match(r"^\s+(\w[\w\s/]*?):\s*(.+)$", line)
        if kv_match and current is not None:
            current.metrics[kv_match.group(1).strip()] = kv_match.group(2).strip()

    return results


def generate_markdown(results: list[BenchmarkResult]) -> str:
    """Generate a markdown report from benchmark results."""
    lines = ["# Synaptiq Benchmark Report", ""]

    # Group by category.
    categories: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        if "latency" in r.name.lower() or "search" in r.name.lower():
            cat = "Query Latency"
        elif "vs" in r.name or "reduction" in r.name.lower() or "token" in r.name.lower():
            cat = "Token Reduction"
        elif "dead" in r.name.lower() or "accuracy" in r.name.lower():
            cat = "Dead Code Accuracy"
        elif "speed" in r.name.lower() or "files" in r.name.lower():
            cat = "Indexing Speed"
        else:
            cat = "Other"
        categories.setdefault(cat, []).append(r)

    for cat, cat_results in categories.items():
        lines.append(f"## {cat}")
        lines.append("")

        for r in cat_results:
            lines.append(f"### {r.name}")
            lines.append("")
            if r.metrics:
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for k, v in r.metrics.items():
                    lines.append(f"| {k} | {v} |")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Read pytest output from stdin and write markdown report."""
    text = sys.stdin.read()
    results = parse_output(text)
    if not results:
        print("No benchmark results found in input.", file=sys.stderr)
        sys.exit(1)
    print(generate_markdown(results))


if __name__ == "__main__":
    main()
