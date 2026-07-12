"""Enable ``python -m synaptiq``.

This is how ``analyze --embeddings lazy`` spawns its detached background
embedding worker (see :mod:`synaptiq.core.embeddings.lazy_worker`): a fresh
``python -m synaptiq _embed-worker <repo>`` process, independent of how the
parent CLI itself was launched (console script, ``uv run``, module, ...).
"""

from __future__ import annotations

from synaptiq.cli.main import app

if __name__ == "__main__":
    app()
