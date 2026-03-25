"""Shared fixtures for Synaptiq benchmarks.

Generates synthetic Python/TypeScript repos of varying sizes for
reproducible performance measurement.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from synaptiq.core.ingestion.pipeline import run_pipeline
from synaptiq.core.storage.kuzu_backend import KuzuBackend

REPO_SIZES = {
    "tiny": 10,
    "small": 50,
    "medium": 100,
    "large": 250,
}


def _generate_python_file(index: int, total: int) -> str:
    """Generate a synthetic Python file with known call chains."""
    imports = ""
    calls = ""
    if index > 0:
        prev_mod = f"mod_{index - 1}"
        imports = f"from .{prev_mod} import process_{index - 1}\n"
        calls = f"    result = process_{index - 1}(data)\n"

    dead_func = ""
    if index % 5 == 0:
        dead_func = textwrap.dedent(f"""\

        def _unused_helper_{index}(x: int) -> int:
            \"\"\"This function is intentionally dead code.\"\"\"
            return x * 2
        """)

    return textwrap.dedent(f"""\
    \"\"\"Module {index} — synthetic benchmark file.\"\"\"

    {imports}

    class Service_{index}:
        \"\"\"Service class for module {index}.\"\"\"

        def __init__(self, name: str = "service_{index}"):
            self.name = name

        def handle(self, data: dict) -> dict:
            \"\"\"Handle incoming data.\"\"\"
    {calls}        return {{"module": {index}, "data": data}}


    def process_{index}(data: dict) -> dict:
        \"\"\"Process data in module {index}.\"\"\"
        svc = Service_{index}()
        return svc.handle(data)
    {dead_func}""")


def _generate_typescript_file(index: int) -> str:
    """Generate a synthetic TypeScript file."""
    return textwrap.dedent(f"""\
    // Module {index} — synthetic benchmark file.

    export interface Config_{index} {{
        name: string;
        value: number;
    }}

    export class Handler_{index} {{
        config: Config_{index};

        constructor(config: Config_{index}) {{
            this.config = config;
        }}

        process(input: string): string {{
            return `${{this.config.name}}: ${{input}}`;
        }}
    }}

    export function createHandler_{index}(name: string): Handler_{index} {{
        return new Handler_{index}({{ name, value: {index} }});
    }}
    """)


def generate_repo(tmp_path: Path, size: int) -> Path:
    """Generate a synthetic repo with *size* files (80% Python, 20% TypeScript)."""
    repo_dir = tmp_path / "bench_repo"
    repo_dir.mkdir(exist_ok=True)

    py_dir = repo_dir / "src" / "app"
    py_dir.mkdir(parents=True, exist_ok=True)
    (py_dir / "__init__.py").write_text("")

    ts_dir = repo_dir / "src" / "frontend"
    ts_dir.mkdir(parents=True, exist_ok=True)

    py_count = int(size * 0.8)
    ts_count = size - py_count

    for i in range(py_count):
        (py_dir / f"mod_{i}.py").write_text(_generate_python_file(i, py_count))

    for i in range(ts_count):
        (ts_dir / f"component_{i}.ts").write_text(_generate_typescript_file(i))

    # Add a .gitignore so the walker works.
    (repo_dir / ".gitignore").write_text("__pycache__/\n.synaptiq/\nnode_modules/\n")

    return repo_dir


@pytest.fixture(params=list(REPO_SIZES.keys()))
def benchmark_repo(request, tmp_path):
    """Parametrised fixture that generates repos of different sizes."""
    size_name = request.param
    size = REPO_SIZES[size_name]
    repo_dir = generate_repo(tmp_path, size)
    return size_name, size, repo_dir


@pytest.fixture
def small_repo(tmp_path):
    """A small (50-file) repo for quick latency benchmarks."""
    return generate_repo(tmp_path, 50)


@pytest.fixture
def indexed_small_repo(small_repo):
    """A pre-indexed small repo with storage backend."""
    storage = KuzuBackend()
    db_path = small_repo / ".synaptiq" / "kuzu"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage.initialize(db_path)
    graph, result = run_pipeline(small_repo, storage=storage, skip_embeddings=True)
    yield small_repo, storage, graph, result
    storage.close()
    shutil.rmtree(small_repo / ".synaptiq", ignore_errors=True)
