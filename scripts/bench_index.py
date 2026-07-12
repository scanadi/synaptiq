#!/usr/bin/env python3
"""Benchmark harness for the Synaptiq ingestion pipeline (W0.1).

Runs :func:`synaptiq.core.ingestion.pipeline.run_pipeline` against a target
repository ``--runs`` times (default 3), each time against a fresh, throwaway
KuzuDB in a temporary directory so that per-run numbers are not skewed by
incremental-embedding reuse or warm FTS/HNSW state.  Storage is real (not
``None``) so the "Loading to storage" phase (COPY + FTS) is included, per the
W0.1 spec.

Embeddings are skipped by default (they are the most expensive and most
environment-dependent phase — first run downloads/loads the ONNX model);
pass ``--embeddings`` to include them.

Usage::

    uv run python scripts/bench_index.py <path> [--runs N] [--json] [--embeddings]

Examples::

    uv run python scripts/bench_index.py . --runs 3 --json
    uv run python scripts/bench_index.py . --runs 1 --embeddings
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from synaptiq.core.ingestion.pipeline import run_pipeline
from synaptiq.core.storage.kuzu_backend import KuzuBackend


def run_once(repo_path: Path, *, skip_embeddings: bool) -> dict[str, Any]:
    """Run the pipeline once against a fresh, temporary storage backend.

    Returns a sample dict with ``phase_timings``, ``duration_seconds``, and
    a few headline counts (files/symbols/relationships) for context.
    """
    with tempfile.TemporaryDirectory(prefix="synaptiq-bench-") as tmp_dir:
        storage = KuzuBackend()
        storage.initialize(Path(tmp_dir) / "kuzu")
        try:
            _, result = run_pipeline(
                repo_path,
                storage,
                full=True,
                skip_embeddings=skip_embeddings,
            )
        finally:
            storage.close()

    return {
        "phase_timings": dict(result.phase_timings),
        "duration_seconds": result.duration_seconds,
        "files": result.files,
        "symbols": result.symbols,
        "relationships": result.relationships,
        "embeddings": result.embeddings,
    }


def compute_medians(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Median per phase (independently) plus median total duration."""
    phases = sorted({phase for sample in samples for phase in sample["phase_timings"]})
    phase_medians = {
        phase: statistics.median(sample["phase_timings"].get(phase, 0.0) for sample in samples)
        for phase in phases
    }
    return {
        "phase_timings": phase_medians,
        "duration_seconds": statistics.median(sample["duration_seconds"] for sample in samples),
        "files": statistics.median(sample["files"] for sample in samples),
        "symbols": statistics.median(sample["symbols"] for sample in samples),
        "relationships": statistics.median(sample["relationships"] for sample in samples),
    }


def print_table(repo_path: Path, samples: list[dict[str, Any]], medians: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    runs = len(samples)
    console.print(
        f"[bold]Benchmark:[/bold] {repo_path}  "
        f"({int(medians['files'])} files, {int(medians['symbols'])} symbols, "
        f"{int(medians['relationships'])} relationships)"
    )

    total = medians["duration_seconds"]
    table = Table(title=f"Median phase timings ({runs} run{'s' if runs != 1 else ''})")
    table.add_column("Phase")
    table.add_column("Seconds", justify="right")
    table.add_column("% of total", justify="right")
    for phase, seconds in medians["phase_timings"].items():
        pct = (seconds / total * 100) if total else 0.0
        table.add_row(phase, f"{seconds:.3f}", f"{pct:.1f}%")
    table.add_row("Total", f"{total:.3f}", "100.0%", style="bold")
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the Synaptiq ingestion pipeline (median per-phase timing)."
    )
    parser.add_argument("path", type=Path, help="Path to the repository to index.")
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of pipeline runs to median over (default: 3)."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of a table."
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Include embedding generation (skipped by default — it is the "
        "most expensive and most environment-dependent phase).",
    )
    args = parser.parse_args(argv)

    repo_path = args.path.resolve()
    if not repo_path.is_dir():
        print(f"error: {repo_path} is not a directory", file=sys.stderr)
        return 1

    runs = max(1, args.runs)
    samples: list[dict[str, Any]] = []
    for i in range(runs):
        print(f"run {i + 1}/{runs}...", file=sys.stderr)
        t0 = time.monotonic()
        samples.append(run_once(repo_path, skip_embeddings=not args.embeddings))
        print(f"  done in {time.monotonic() - t0:.2f}s", file=sys.stderr)

    medians = compute_medians(samples)

    if args.json:
        payload = {
            "path": str(repo_path),
            "runs": runs,
            "embeddings": args.embeddings,
            "samples": samples,
            "medians": medians,
        }
        print(json.dumps(payload, indent=2))
    else:
        print_table(repo_path, samples, medians)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
