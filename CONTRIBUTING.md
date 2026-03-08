# Contributing to Synaptiq

Thanks for your interest in contributing to Synaptiq! This document covers how to get set up, run tests, and submit changes.

## Getting Started

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager)
- **Git**

### Setup

```bash
git clone https://github.com/scanadi/synaptiq.git
cd synaptiq
uv sync --all-extras
```

This installs all dependencies including dev tools (pytest, ruff).

### Verify your setup

```bash
uv run synaptiq --help
uv run pytest -x -q
```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=synaptiq

# Run a specific test file
uv run pytest tests/core/test_graph.py

# Run a specific test
uv run pytest tests/core/test_graph.py::test_add_node -v

# Run only fast unit tests (skip e2e)
uv run pytest tests/core/ tests/cli/ tests/mcp/
```

### Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
# Check for issues
uv run ruff check src/ tests/

# Auto-fix what can be fixed
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/
```

**Ruff rules enabled:** `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `N` (naming), `W` (warnings). Line length is 100 characters. Target is Python 3.11.

### Running from Source

```bash
# Run any CLI command via uv
uv run synaptiq analyze .
uv run synaptiq query "my search"
uv run synaptiq serve --watch
```

## Project Structure

```
src/synaptiq/
├── cli/              # Typer CLI commands
├── config/           # Language mappings, .gitignore handling
├── core/
│   ├── parsers/      # tree-sitter parsers (Python, JS, TS)
│   ├── ingestion/    # 11-phase analysis pipeline
│   ├── graph/        # In-memory knowledge graph
│   ├── storage/      # KuzuDB backend (+ optional Neo4j)
│   ├── search/       # Hybrid search (BM25 + vector + RRF)
│   ├── embeddings/   # ONNX-based embeddings
│   └── daemon/       # Multi-instance concurrency (primary/proxy)
├── mcp/              # MCP server, tools, resources
tests/
├── core/             # Unit tests for core modules
├── cli/              # CLI command tests
├── mcp/              # MCP server tests
└── e2e/              # End-to-end pipeline tests
```

## Submitting Changes

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Run tests: `uv run pytest -x -q`
5. Run linting: `uv run ruff check src/ tests/`
6. Commit with a clear message describing the change
7. Push and open a pull request against `main`

### Commit Messages

Use clear, descriptive commit messages:

- `feat: add Go language parser support`
- `fix: resolve false positives in dead code detection for dataclass fields`
- `refactor: extract symbol resolution into shared utility`
- `test: add coverage for multi-instance proxy mode`
- `docs: update MCP setup instructions for Cursor`

### What Makes a Good PR

- **Focused** — one logical change per PR
- **Tested** — add or update tests for your changes
- **Linted** — passes `ruff check` without warnings
- **Documented** — update README if you're adding user-facing features

## Adding a New Language Parser

Synaptiq uses tree-sitter for parsing. To add support for a new language:

1. Add the tree-sitter grammar dependency to `pyproject.toml`
2. Create a parser in `src/synaptiq/core/parsers/` extending `BaseParser`
3. Register the language and extensions in `src/synaptiq/config/languages.py`
4. Add tests in `tests/core/test_parser_<language>.py`

See `src/synaptiq/core/parsers/python_lang.py` for a reference implementation.

## Reporting Issues

Use [GitHub Issues](https://github.com/scanadi/synaptiq/issues) to report bugs or request features. Include:

- Steps to reproduce (for bugs)
- Python version and OS
- Relevant error output or logs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
