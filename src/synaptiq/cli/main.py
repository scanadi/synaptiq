"""Synaptiq CLI — Graph-powered code intelligence engine."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from synaptiq import __version__

console = Console()
_stderr_console = Console(stderr=True)


def _write_meta(data_dir: Path, repo_path: Path, result: object) -> None:
    """Write meta.json with index stats (shared implementation in pipeline)."""
    from synaptiq.core.ingestion.pipeline import write_meta

    write_meta(data_dir, repo_path, result)


def _print_phase_timing_table(phase_timings: dict, total_seconds: float) -> None:
    """Print a rich table of per-phase wall time (``analyze --profile``)."""
    table = Table(title="Phase timings")
    table.add_column("Phase")
    table.add_column("Seconds", justify="right")
    table.add_column("% of total", justify="right")
    for phase, seconds in phase_timings.items():
        pct = (seconds / total_seconds * 100) if total_seconds else 0.0
        table.add_row(phase, f"{seconds:.2f}", f"{pct:.1f}%")
    table.add_row("Total", f"{total_seconds:.2f}", "100.0%", style="bold")
    console.print(table)


def _load_storage(repo_path: Path | None = None) -> "LadybugBackend":  # noqa: F821
    """Load the LadybugDB backend for the given or current repo."""
    from synaptiq.core.storage.ladybug_backend import LadybugBackend

    target = (repo_path or Path.cwd()).resolve()
    # The on-disk index path is kept as ``.synaptiq/kuzu`` deliberately: an
    # index written by the former KuzuDB backend lives there, and reusing the
    # path lets `synaptiq analyze` (via open_with_recovery) detect the old
    # format on open and rebuild it in place.
    db_path = target / ".synaptiq" / "kuzu"
    if not db_path.exists():
        console.print(
            f"[red]Error:[/red] No index found at {target}. Run 'synaptiq analyze' first."
        )
        raise typer.Exit(code=1)

    storage = LadybugBackend()
    try:
        storage.initialize(db_path, read_only=True)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "lock on file" in msg or "lock is held" in msg:
            console.print(
                "[red]Error:[/red] The database is locked by a running "
                "`synaptiq serve` instance and it could not be reached "
                "over its socket. Stop the server or retry."
            )
            raise typer.Exit(code=1) from exc
        # A stale index from the former KuzuDB backend (single-file LadybugDB
        # rejects the old directory format) or a corrupt file can't be read
        # read-only — point the user at a rebuild instead of dumping a traceback.
        console.print(
            "[red]Error:[/red] The index at "
            f"{target} could not be opened (it may be from an older synaptiq "
            "version or corrupt). Run 'synaptiq analyze' to rebuild it."
        )
        raise typer.Exit(code=1) from exc
    return storage


def _healthy_server_socket(data_dir: Path) -> str | None:
    """Return the socket path of a healthy running server, or ``None``."""
    from synaptiq.core.daemon.lock import LockManager

    existing = LockManager(data_dir).read_existing()
    if existing is not None and not existing.is_stale():
        return existing.socket
    return None


def _call_tool_via_server(socket_path: str, tool: str, arguments: dict) -> str:
    """Forward a tool call to a running server over its Unix socket."""
    import asyncio

    from synaptiq.core.daemon.socket_client import SocketClient

    async def _do() -> str:
        client = SocketClient(Path(socket_path))
        await client.connect()
        try:
            return await client.call_tool(tool, arguments)
        finally:
            await client.close()

    return asyncio.run(_do())


def _run_read_tool(tool: str, arguments: dict) -> str:
    """Run a read-only tool locally, or via a running server's socket.

    While ``synaptiq serve --watch`` holds the database read-write, LadybugDB
    refuses read-only opens from other processes — so CLI reads must go
    through the server instead of opening the database directly.

    Error routing matters here: only ConnectionError means the server is
    actually unreachable (fall back to direct access).  A RuntimeError is
    the server *responding* with an error, and a TimeoutError means it is
    up but slow — in both cases falling back would just hit LadybugDB's file
    lock and mask the real cause.
    """
    from synaptiq.mcp.token_budget import strip_metadata

    data_dir = Path.cwd().resolve() / ".synaptiq"
    socket_path = _healthy_server_socket(data_dir)
    if socket_path is not None:
        try:
            result = _call_tool_via_server(socket_path, tool, arguments)
            return strip_metadata(result)
        except ConnectionError as exc:
            _stderr_console.print(
                f"[yellow]Warning:[/yellow] could not reach running server "
                f"({exc}); falling back to direct database access."
            )
        except TimeoutError as exc:
            console.print(
                f"[red]Error:[/red] the running synaptiq server did not "
                f"respond in time ({exc or 'timeout'})."
            )
            raise typer.Exit(code=1) from exc
        except RuntimeError as exc:
            console.print(f"[red]Error from server:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    from synaptiq.mcp.server import dispatch_tool

    storage = _load_storage()
    try:
        result = dispatch_tool(tool, arguments, storage)
        return strip_metadata(result)
    finally:
        storage.close()


app = typer.Typer(
    name="synaptiq",
    help="Synaptiq — Graph-powered code intelligence engine.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print the version and exit."""
    if value:
        console.print(f"Synaptiq v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(  # noqa: N803
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Synaptiq — Graph-powered code intelligence engine."""
    # Update check — skip for MCP transport commands (stdout is the protocol).
    if len(sys.argv) < 2 or sys.argv[1] not in ("serve", "mcp"):
        from synaptiq.cli.update_check import check_for_update_message, trigger_background_check

        msg = check_for_update_message()
        if msg:
            _stderr_console.print(f"[dim]{msg}[/dim]")
        trigger_background_check()


@app.command()
def analyze(
    path: Path = typer.Argument(Path("."), help="Path to the repository to index."),
    full: bool = typer.Option(False, "--full", help="Perform a full re-index."),
    no_embeddings: bool = typer.Option(
        False, "--no-embeddings", help="Skip embedding generation for faster indexing."
    ),
    profile: bool = typer.Option(
        False, "--profile", help="Print a per-phase timing breakdown after indexing."
    ),
    jobs: Optional[int] = typer.Option(
        None,
        "--jobs",
        "-j",
        help=(
            "Cap LadybugDB threads, ONNX embedding threads, and the walk/parse "
            "worker pools to N. N=0 means explicit all-cores (restores uncapped "
            "engine/embedding threads, overriding SYNAPTIQ_* env vars too). Omitted "
            "(default): SYNAPTIQ_DB_THREADS / SYNAPTIQ_DB_MEMORY_MB / "
            "SYNAPTIQ_EMBED_THREADS apply if set, else embedding threads default "
            "to a polite max(2, cores - 2) so a foreground index leaves the "
            "machine usable (engine threads and worker pools keep their library "
            "defaults). Precedence: --jobs > env vars > profile defaults."
        ),
    ),
) -> None:
    """Index a repository into a knowledge graph."""
    from synaptiq.core.daemon.lock import LockManager
    from synaptiq.core.ingestion.pipeline import PipelineResult, run_pipeline
    from synaptiq.core.resources import set_jobs
    from synaptiq.core.storage.ladybug_backend import open_with_recovery

    repo_path = path.resolve()
    if not repo_path.is_dir():
        _stderr_console.print(f"[red]Error:[/red] {repo_path} is not a directory.")
        raise typer.Exit(code=1)

    if jobs is not None and jobs < 0:
        _stderr_console.print("[red]Error:[/red] --jobs must be >= 0.")
        raise typer.Exit(code=1)
    # Must run before any storage backend or embedding model is created —
    # both read current_limits() at creation time (see core/resources.py).
    set_jobs(jobs)

    data_dir = repo_path / ".synaptiq"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Acquire lock to prevent concurrent access to the database.
    lock_mgr = LockManager(data_dir)
    lock_info = lock_mgr.try_acquire()
    if lock_info is None:
        existing = lock_mgr.read_existing()
        if existing is not None and not existing.is_stale():
            # Server is running — delegate reindex via its Unix socket.
            console.print(
                f"[bold]Server running (PID {existing.pid}), requesting reindex...[/bold]"
            )
            _reindex_via_server(
                existing.socket, full=full, skip_embeddings=no_embeddings, profile=profile
            )
            return
        # Stale lock — clean up and retry.
        lock_mgr.force_cleanup()
        lock_info = lock_mgr.try_acquire()
        if lock_info is None:
            console.print(
                "[red]Error:[/red] Could not acquire lock. Another process may be starting."
            )
            raise typer.Exit(code=1)

    try:
        console.print(f"[bold]Indexing[/bold] {repo_path}")

        # Path kept as ``kuzu`` so an index from the former KuzuDB backend is
        # detected on open and rebuilt in place (see open_with_recovery).
        db_path = data_dir / "kuzu"
        # build_fts_indexes=False: bulk_load builds FTS over the populated
        # tables and swaps the fresh database in, so building empty FTS indexes
        # on this initial open would be pure waste (~2s on LadybugDB).
        storage = open_with_recovery(
            db_path, data_dir / "meta.json", build_fts_indexes=False
        )

        result: PipelineResult | None = None
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Starting...", total=None)

            def on_progress(phase: str, pct: float) -> None:
                progress.update(task, description=f"{phase} ({pct:.0%})")

            _, result = run_pipeline(
                repo_path=repo_path,
                storage=storage,
                full=full,
                progress_callback=on_progress,
                skip_embeddings=no_embeddings,
            )

        _write_meta(data_dir, repo_path, result)

        console.print()
        console.print("[bold green]Indexing complete.[/bold green]")
        console.print(f"  Files:          {result.files}")
        console.print(f"  Symbols:        {result.symbols}")
        console.print(f"  Relationships:  {result.relationships}")
        if result.clusters > 0:
            console.print(f"  Clusters:       {result.clusters}")
        if result.processes > 0:
            console.print(f"  Flows:          {result.processes}")
        if result.dead_code > 0:
            console.print(f"  Dead code:      {result.dead_code}")
        if result.coupled_pairs > 0:
            console.print(f"  Coupled pairs:  {result.coupled_pairs}")
        if result.embeddings > 0:
            console.print(f"  Embeddings:     {result.embeddings}")
        console.print(f"  Duration:       {result.duration_seconds:.2f}s")

        if profile and result.phase_timings:
            console.print()
            _print_phase_timing_table(result.phase_timings, result.duration_seconds)

        storage.close()
    finally:
        lock_mgr.release()


@app.command()
def status() -> None:
    """Show index status for current repository."""
    repo_path = Path.cwd().resolve()
    meta_path = repo_path / ".synaptiq" / "meta.json"

    if not meta_path.exists():
        console.print(
            f"[red]Error:[/red] No index found at {repo_path}. Run 'synaptiq analyze' first."
        )
        raise typer.Exit(code=1)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = meta.get("stats", {})

    console.print(f"[bold]Index status for[/bold] {repo_path}")
    console.print(f"  Version:        {meta.get('version', '?')}")
    console.print(f"  Last indexed:   {meta.get('last_indexed_at', '?')}")
    console.print(f"  Files:          {stats.get('files', '?')}")
    console.print(f"  Symbols:        {stats.get('symbols', '?')}")
    console.print(f"  Relationships:  {stats.get('relationships', '?')}")

    if stats.get("clusters", 0) > 0:
        console.print(f"  Clusters:       {stats['clusters']}")
    if stats.get("flows", 0) > 0:
        console.print(f"  Flows:          {stats['flows']}")
    if stats.get("dead_code", 0) > 0:
        console.print(f"  Dead code:      {stats['dead_code']}")
    if stats.get("coupled_pairs", 0) > 0:
        console.print(f"  Coupled pairs:  {stats['coupled_pairs']}")


@app.command(name="list")
def list_repos() -> None:
    """List all indexed repositories."""
    from synaptiq.mcp.tools import handle_list_repos

    result = handle_list_repos()
    console.print(result)


@app.command()
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
) -> None:
    """Delete index for current repository."""
    repo_path = Path.cwd().resolve()
    data_dir = repo_path / ".synaptiq"

    if not data_dir.exists():
        console.print(f"[red]Error:[/red] No index found at {repo_path}. Nothing to clean.")
        raise typer.Exit(code=1)

    socket_path = _healthy_server_socket(data_dir)
    if socket_path is not None:
        console.print(
            "[red]Error:[/red] A synaptiq server is running against this index. "
            "Stop it before cleaning — deleting a live database can corrupt it."
        )
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm(f"Delete index at {data_dir}?")
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit()

    shutil.rmtree(data_dir)
    console.print(f"[green]Deleted[/green] {data_dir}")


@app.command()
def query(
    q: str = typer.Argument(..., help="Search query for the knowledge graph."),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results."),
) -> None:
    """Search the knowledge graph."""
    result = _run_read_tool("synaptiq_query", {"query": q, "limit": limit})
    console.print(result)


@app.command()
def context(
    name: str = typer.Argument(..., help="Symbol name to inspect."),
) -> None:
    """Show 360-degree view of a symbol."""
    result = _run_read_tool("synaptiq_context", {"symbol": name})
    console.print(result)


@app.command()
def impact(
    target: str = typer.Argument(..., help="Symbol to analyze blast radius for."),
    depth: int = typer.Option(3, "--depth", "-d", help="Traversal depth."),
) -> None:
    """Show blast radius of changing a symbol."""
    result = _run_read_tool("synaptiq_impact", {"symbol": target, "depth": depth})
    console.print(result)


@app.command(name="dead-code")
def dead_code() -> None:
    """List all detected dead code."""
    result = _run_read_tool("synaptiq_dead_code", {})
    console.print(result)


@app.command()
def cypher(
    query: str = typer.Argument(..., help="Raw Cypher query to execute."),
) -> None:
    """Execute raw Cypher against the knowledge graph."""
    result = _run_read_tool("synaptiq_cypher", {"query": query})
    console.print(result)


@app.command()
def setup(
    claude: bool = typer.Option(False, "--claude", help="Configure MCP for Claude Code."),
    cursor: bool = typer.Option(False, "--cursor", help="Configure MCP for Cursor."),
    http: bool = typer.Option(False, "--http", help="Show HTTP transport config instead of stdio."),
    port: int = typer.Option(8080, "--port", help="Port for HTTP transport config."),
) -> None:
    """Configure MCP for Claude Code / Cursor."""
    if http:
        mcp_config = {
            "url": f"http://127.0.0.1:{port}/mcp",
        }
    else:
        mcp_config = {
            "command": "synaptiq",
            "args": ["serve", "--watch"],
        }

    if claude or (not claude and not cursor):
        console.print("[bold]Add to your Claude Code MCP config:[/bold]")
        console.print(json.dumps({"synaptiq": mcp_config}, indent=2))
        if http:
            console.print(
                "\n[dim]Start the server with: synaptiq serve --watch "
                f"--transport http --port {port}[/dim]"
            )

    if cursor or (not claude and not cursor):
        console.print("[bold]Add to your Cursor MCP config:[/bold]")
        console.print(json.dumps({"synaptiq": mcp_config}, indent=2))
        if http:
            console.print(
                "\n[dim]Start the server with: synaptiq serve --watch "
                f"--transport http --port {port}[/dim]"
            )


@app.command()
def watch() -> None:
    """Watch mode — re-index on file changes."""
    import asyncio

    from synaptiq.core.daemon.lock import LockManager
    from synaptiq.core.ingestion.watcher import watch_repo
    from synaptiq.core.resources import set_profile

    # Background daemon — rebuilds and re-embeds must stay polite.
    set_profile("server")

    repo_path = Path.cwd().resolve()
    data_dir = repo_path / ".synaptiq"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Acquire lock to prevent concurrent access to the database.
    lock_mgr = LockManager(data_dir)
    lock_info = lock_mgr.try_acquire()
    if lock_info is None:
        existing = lock_mgr.read_existing()
        if existing is not None and not existing.is_stale():
            console.print(
                f"[red]Error:[/red] synaptiq is running (PID {existing.pid}). Stop it first."
            )
        else:
            console.print(
                "[red]Error:[/red] Another synaptiq process is running. Wait for it to finish."
            )
        raise typer.Exit(code=1)

    try:
        storage = _init_storage_with_index(repo_path, data_dir)
        console.print(f"[bold]Watching[/bold] {repo_path} for changes (Ctrl+C to stop)")

        try:
            asyncio.run(watch_repo(repo_path, storage))
        except KeyboardInterrupt:
            console.print("\n[bold]Watch stopped.[/bold]")
        finally:
            storage.close()
    finally:
        lock_mgr.release()


@app.command()
def diff(
    branch_range: str = typer.Argument(
        ..., help="Branch range for comparison (e.g. main..feature)."
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help=(
            "Build the complete graph for both sides (exhaustive cross-file "
            "relationships, cost scales with repo size). Default mode parses "
            "only changed files."
        ),
    ),
) -> None:
    """Structural branch comparison (parses only changed files by default)."""
    from synaptiq.core.diff import diff_branches, format_diff

    repo_path = Path.cwd().resolve()
    try:
        result = diff_branches(repo_path, branch_range, full=full)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(format_diff(result))
    if not full:
        console.print(
            "[dim]Scoped diff: relationships limited to changed files. "
            "Use --full for exhaustive cross-file comparison.[/dim]"
        )


@app.command()
def mcp() -> None:
    """Start MCP server (stdio transport)."""
    import asyncio

    from synaptiq.core.resources import set_profile
    from synaptiq.mcp.server import main as mcp_main

    set_profile("server")
    asyncio.run(mcp_main())


@app.command()
def serve(
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Enable file watching with auto-reindex."
    ),
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Project directory to index (defaults to cwd)."
    ),
    transport: str = typer.Option(
        "stdio", "--transport", "-t", help="Transport protocol: stdio or http."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host for HTTP transport."),
    port: int = typer.Option(8080, "--port", help="Port for HTTP transport."),
) -> None:
    """Start MCP server, optionally with live file watching."""
    import asyncio
    import os

    from synaptiq.core.resources import set_profile
    from synaptiq.mcp.server import main as mcp_main

    # Long-running daemon beside the user's real work — cap engine
    # threads, buffer pool, and embedding threads (covers primary,
    # proxy, and proxy-promoted-to-primary paths).
    set_profile("server")

    if transport not in ("stdio", "http"):
        _stderr_console.print(
            f"[red]Error:[/red] Unknown transport '{transport}'. Use 'stdio' or 'http'."
        )
        raise typer.Exit(code=1)

    # When --path is given, change cwd so all downstream code resolves correctly.
    if path is not None:
        resolved = path.resolve()
        if not resolved.is_dir():
            _stderr_console.print(f"[red]Error:[/red] {resolved} is not a directory.")
            raise typer.Exit(code=1)
        os.chdir(resolved)

    if not watch:
        if transport == "http":
            _serve_http_standalone(host, port)
        else:
            asyncio.run(mcp_main())
        return

    from synaptiq.core.daemon.lock import LockManager

    repo_path = Path.cwd().resolve()
    data_dir = repo_path / ".synaptiq"
    data_dir.mkdir(parents=True, exist_ok=True)

    lock_mgr = LockManager(data_dir)
    lock_info = lock_mgr.try_acquire()

    if lock_info is None:
        # Another instance holds the lock — check if healthy or stale.
        existing = lock_mgr.read_existing()
        if existing is not None and existing.is_stale():
            lock_mgr.force_cleanup()
            lock_info = lock_mgr.try_acquire()

    if lock_info is not None:
        _serve_primary(repo_path, data_dir, lock_mgr, transport=transport, host=host, port=port)
    else:
        existing = existing or lock_mgr.read_existing()
        if existing is None:
            print("Error: cannot read lock info from primary", file=sys.stderr)
            raise typer.Exit(code=1)
        _serve_proxy(existing.socket, repo_path, data_dir)


def _init_storage_with_index(repo_path: Path, data_dir: Path, *, output=console):
    """Initialise LadybugDB and run the first index if no meta.json exists."""
    from synaptiq.core.ingestion.pipeline import run_pipeline
    from synaptiq.core.storage.ladybug_backend import open_with_recovery

    # Path kept as ``kuzu`` for back-compat (see open_with_recovery). Unlike the
    # analyze path this keeps the default FTS build: the server may open an
    # already-indexed database here and must serve full-text queries immediately.
    db_path = data_dir / "kuzu"
    storage = open_with_recovery(db_path, data_dir / "meta.json")

    if not (data_dir / "meta.json").exists():
        import time as _time

        output.print(f"[bold]Indexing[/bold] {repo_path}")

        _last_phase = [None]
        _phase_start = [_time.monotonic()]

        def _on_progress(phase: str, pct: float) -> None:
            now = _time.monotonic()
            if phase != _last_phase[0]:
                if _last_phase[0] is not None:
                    elapsed = now - _phase_start[0]
                    output.print(f"  {_last_phase[0]} [dim]({elapsed:.1f}s)[/dim]")
                _last_phase[0] = phase
                _phase_start[0] = now

        _, result = run_pipeline(repo_path, storage, full=True, progress_callback=_on_progress)

        if _last_phase[0] is not None:
            elapsed = _time.monotonic() - _phase_start[0]
            output.print(f"  {_last_phase[0]} [dim]({elapsed:.1f}s)[/dim]")

        output.print(
            f"[bold green]Done[/bold green] — {result.files} files, "
            f"{result.symbols} symbols in {result.duration_seconds:.1f}s"
        )
        _write_meta(data_dir, repo_path, result)

    return storage


def _reindex_via_server(
    socket_path: str, *, full: bool = True, skip_embeddings: bool = False, profile: bool = False
) -> None:
    """Send a reindex request to a running synaptiq server via its Unix socket."""
    import asyncio

    from synaptiq.core.daemon.socket_client import SocketClient

    async def _do() -> str:
        client = SocketClient(Path(socket_path))
        await client.connect()
        try:
            return await client.reindex(full=full, skip_embeddings=skip_embeddings)
        finally:
            await client.close()

    console.print("[dim]Waiting for reindex to complete...[/dim]")
    try:
        result_str = asyncio.run(_do())
    except (ConnectionError, RuntimeError, TimeoutError) as exc:
        console.print(f"[red]Error:[/red] {exc or 'reindex request timed out'}")
        raise typer.Exit(code=1) from exc

    result = json.loads(result_str)
    stats = result.get("stats", {})

    console.print()
    console.print("[bold green]Reindex complete (via server).[/bold green]")
    for key in (
        "files",
        "symbols",
        "relationships",
        "clusters",
        "flows",
        "dead_code",
        "coupled_pairs",
    ):
        val = stats.get(key, 0)
        if val > 0:
            label = key.replace("_", " ").title()
            console.print(f"  {label + ':':<16}{val}")
    if "duration" in result:
        console.print(f"  {'Duration:':<16}{result['duration']:.2f}s")

    if profile:
        phase_timings = result.get("phase_timings")
        if phase_timings:
            console.print()
            _print_phase_timing_table(phase_timings, result.get("duration", 0.0))


def _reindex_stats_json(result, duration: float) -> str:
    return json.dumps(
        {
            "stats": {
                "files": result.files,
                "symbols": result.symbols,
                "relationships": result.relationships,
                "clusters": result.clusters,
                "flows": result.processes,
                "dead_code": result.dead_code,
                "coupled_pairs": result.coupled_pairs,
            },
            "duration": round(duration, 2),
            "phase_timings": result.phase_timings,
        }
    )


def _serve_http_standalone(host: str, port: int) -> None:
    """Run the MCP server over HTTP transport without file watching."""
    import uvicorn

    from synaptiq.mcp.http_transport import create_starlette_app
    from synaptiq.mcp.server import server as mcp_server

    _stderr_console.print(f"[bold]Synaptiq MCP server (HTTP)[/bold] http://{host}:{port}/mcp")

    app, _session_mgr = create_starlette_app(mcp_server)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _report_watch_death(task) -> None:
    """A dead watcher silently serves an ever-staler index — say so."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        import logging

        logging.getLogger(__name__).error("File watcher crashed", exc_info=exc)
        _stderr_console.print(
            f"[red]File watcher crashed:[/red] {exc} — "
            "the index will no longer update. Restart the server."
        )


# The MCP SDK's server.run() can outlive its session: it neither returns on
# stdin EOF nor honours task cancellation, so a serve process would otherwise
# wedge forever while holding the database lock.  Two defenses:
#   - a stdin sentinel that turns pipe hangup into the stop event, and
#   - a watchdog that force-exits after a bounded grace once stop fires,
#     running the cleanup callback (lock release) first.

_SHUTDOWN_GRACE_SECONDS = 20.0


def _watch_stdin_hup(loop, stop_event) -> None:
    """Set *stop_event* when stdin's write end closes (pipe hangup)."""
    import select
    import threading

    try:
        fd = sys.stdin.fileno()
    except (ValueError, OSError):
        return

    def _poll_for_hup() -> None:
        poller = select.poll()
        # events=0: POLLHUP/POLLERR/POLLNVAL are always reported, and we
        # never consume data the MCP stdio reader owns.
        poller.register(fd, 0)
        while True:
            # poll() returns as soon as a registered event fires, so a HUP
            # is still caught immediately — this timeout only bounds how
            # often the loop wakes for nothing while stdin stays open.
            # 10s keeps that idle wakeup rare without delaying shutdown.
            for _fd, event in poller.poll(10000):
                if event & (select.POLLHUP | select.POLLERR | select.POLLNVAL):
                    loop.call_soon_threadsafe(stop_event.set)
                    return

    threading.Thread(target=_poll_for_hup, name="stdin-hup-watch", daemon=True).start()


def _arm_shutdown_watchdog(stop_event, cleanup=None, *, grace: float = _SHUTDOWN_GRACE_SECONDS):
    """Return a task that force-exits *grace* seconds after stop fires."""
    import asyncio
    import os
    import threading

    def _force_exit() -> None:
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                pass
        os._exit(0)

    async def _watchdog() -> None:
        await stop_event.wait()
        timer = threading.Timer(grace, _force_exit)
        timer.daemon = True
        timer.start()

    return asyncio.create_task(_watchdog())


class _PrimaryRuntime:
    """Storage + socket server + watcher bundle for a primary instance.

    Shared by ``_serve_primary`` (primary from startup) and the proxy
    promotion path (proxy takes over after the primary dies), so both
    wire up the exact same machinery.
    """

    def __init__(self, repo_path: Path, data_dir: Path, lock_mgr) -> None:
        self._repo_path = repo_path
        self._data_dir = data_dir
        self._lock_mgr = lock_mgr
        self.storage = None
        self.socket_server = None
        self.watch_task = None

    async def start(self, stop_event) -> None:
        """Initialise storage, start the socket server and file watcher."""
        import asyncio

        from synaptiq.core.daemon.rwlock import AsyncRWLock
        from synaptiq.core.daemon.socket_server import SocketServer
        from synaptiq.core.ingestion.watcher import RebuildCoordinator, watch_repo
        from synaptiq.mcp.server import (
            dispatch_resource,
            dispatch_tool,
            set_rwlock,
            set_storage,
        )

        repo_path = self._repo_path
        data_dir = self._data_dir

        storage = _init_storage_with_index(repo_path, data_dir, output=_stderr_console)

        rwlock = AsyncRWLock()
        set_storage(storage)
        set_rwlock(rwlock)

        # Single-flight guard shared with the watcher's global phase so a
        # socket reindex and the watcher can never run two full CPU builds
        # concurrently in this process (G10).
        rebuild_coordinator = RebuildCoordinator()

        def dispatch(method: str, params: dict) -> str:
            if method == "ping":
                return "pong"
            if method == "tool":
                return dispatch_tool(params.get("name", ""), params.get("arguments", {}), storage)
            if method == "resource":
                return dispatch_resource(params.get("uri", ""), storage)
            return f"Unknown method: {method}"

        async def _reindex_async(params: dict) -> str:
            """Reindex with minimal lock hold: build lock-free, commit locked.

            Holding the write lock across the whole pipeline (plus embedding
            generation) would block every agent query for its full duration.

            The commit is shielded from cancellation: if the dispatch timeout
            fires mid-commit, the commit must run to completion while still
            holding the writer lock — a cancelled `async with rwlock.writer()`
            would release the lock while the commit thread keeps resetting the
            database under live readers.

            The whole build+commit runs through the shared
            ``rebuild_coordinator`` so it can never overlap the watcher's
            global phase (or another reindex) — single-flight, per G10.
            """
            import time as _time

            from synaptiq.core.ingestion.pipeline import (
                build_full_index,
                commit_full_index,
                load_previous_embeddings,
            )

            async def _build_and_commit() -> str:
                start = _time.monotonic()
                skip_embeddings = params.get("skip_embeddings", False)
                previous = (
                    {}
                    if skip_embeddings
                    else await asyncio.to_thread(load_previous_embeddings, storage)
                )
                graph, embeddings, result = await asyncio.to_thread(
                    build_full_index,
                    repo_path,
                    full=params.get("full", True),
                    skip_embeddings=skip_embeddings,
                    previous_embeddings=previous,
                )

                async def _locked_commit() -> None:
                    # Generous acquisition timeout: a single read dispatch may
                    # legitimately hold the reader lock for up to the server's
                    # 120s budget — the default 60s would discard the whole build.
                    async with rwlock.writer(timeout=300.0):
                        await asyncio.to_thread(commit_full_index, storage, graph, embeddings)

                await asyncio.shield(_locked_commit())
                _write_meta(data_dir, repo_path, result)
                return _reindex_stats_json(result, _time.monotonic() - start)

            return await rebuild_coordinator.run(_build_and_commit)

        socket_server = SocketServer(
            self._lock_mgr.socket_path,
            dispatch,
            rwlock=rwlock,
            async_handlers={"reindex": _reindex_async},
        )
        await socket_server.start()

        watch_task = asyncio.create_task(
            watch_repo(
                repo_path,
                storage,
                stop_event=stop_event,
                rwlock=rwlock,
                rebuild_coordinator=rebuild_coordinator,
            )
        )
        watch_task.add_done_callback(_report_watch_death)

        self.storage = storage
        self.socket_server = socket_server
        self.watch_task = watch_task

    async def stop(self) -> None:
        """Stop the watcher and socket server, close storage."""
        import asyncio
        import contextlib

        if self.watch_task is not None:
            self.watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.watch_task
            self.watch_task = None
        if self.socket_server is not None:
            await self.socket_server.stop()
            self.socket_server = None
        if self.storage is not None:
            self.storage.close()
            self.storage = None


def _serve_primary(
    repo_path: Path,
    data_dir: Path,
    lock_mgr,
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run as primary: DB + watcher + MCP + socket server."""
    import asyncio

    from synaptiq.mcp.server import server as mcp_server

    runtime = _PrimaryRuntime(repo_path, data_dir, lock_mgr)

    async def _run() -> None:
        import signal

        stop = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        watchdog = _arm_shutdown_watchdog(stop, lock_mgr.release)
        if transport == "stdio":
            _watch_stdin_hup(loop, stop)

        await runtime.start(stop)
        watch_task = runtime.watch_task
        try:
            if transport == "http":
                import uvicorn

                from synaptiq.mcp.http_transport import create_starlette_app

                _stderr_console.print(
                    f"[bold]Synaptiq MCP server (HTTP)[/bold] http://{host}:{port}/mcp"
                )
                app, _session_mgr = create_starlette_app(mcp_server)
                config = uvicorn.Config(app, host=host, port=port, log_level="warning")
                uv_server = uvicorn.Server(config)

                mcp_task = asyncio.create_task(uv_server.serve())

                async def _wait_stop():
                    await stop.wait()
                    uv_server.should_exit = True
                    mcp_task.cancel()
                    watch_task.cancel()

                mcp_task.add_done_callback(lambda _: stop.set())
                await asyncio.gather(
                    mcp_task,
                    watch_task,
                    _wait_stop(),
                    return_exceptions=True,
                )
            else:
                from mcp.server.stdio import stdio_server

                async with stdio_server() as (read, write):
                    mcp_task = asyncio.create_task(
                        mcp_server.run(read, write, mcp_server.create_initialization_options())
                    )

                    async def _wait_stop():
                        await stop.wait()
                        mcp_task.cancel()
                        watch_task.cancel()

                    # MCP exit → stop everything; signal → stop event → cancel both.
                    mcp_task.add_done_callback(lambda _: stop.set())
                    await asyncio.gather(
                        mcp_task,
                        watch_task,
                        _wait_stop(),
                        return_exceptions=True,
                    )
        finally:
            stop.set()
            await runtime.stop()
            watchdog.cancel()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        lock_mgr.release()


def _serve_proxy(socket_path: str, repo_path: Path, data_dir: Path) -> None:
    """Run as proxy: MCP over stdio, forwarding to primary via socket.

    When the primary dies for good (stale or missing lock after reconnect
    failures), the proxy takes over: it acquires the lock, starts the full
    primary runtime in-process, and dispatches locally from then on.
    """
    import asyncio

    from synaptiq.core.daemon.lock import LockManager
    from synaptiq.core.daemon.socket_client import SocketClient
    from synaptiq.mcp.server import set_proxy_client

    lock_mgr = LockManager(data_dir)
    runtime = _PrimaryRuntime(repo_path, data_dir, lock_mgr)
    stop = asyncio.Event()
    promoted = False

    async def _on_primary_lost() -> bool:
        nonlocal promoted
        if promoted:
            return True
        if lock_mgr.try_acquire() is None:
            # Another proxy won the takeover race — its lock file points
            # at the new socket; the next reconnect follows it.
            return False
        try:
            await runtime.start(stop)
        except Exception:
            lock_mgr.release()
            raise
        set_proxy_client(None)
        promoted = True
        _stderr_console.print(
            "[yellow]Primary daemon lost — this instance promoted itself to primary.[/yellow]"
        )
        return True

    client = SocketClient(
        Path(socket_path),
        lock_reader=lock_mgr.read_existing,
        on_primary_lost=_on_primary_lost,
    )

    async def _run() -> None:
        from mcp.server.stdio import stdio_server

        from synaptiq.mcp.server import server as mcp_server

        def _watchdog_cleanup() -> None:
            # A plain proxy owns nothing; releasing would unlink the live
            # primary's lock and socket files.
            if promoted:
                lock_mgr.release()

        watchdog = _arm_shutdown_watchdog(stop, _watchdog_cleanup)
        _watch_stdin_hup(asyncio.get_running_loop(), stop)

        # Retry connection — the primary may still be starting its socket server.
        for attempt in range(5):
            try:
                await client.connect()
                break
            except ConnectionError:
                if attempt == 4:
                    print("Error: could not connect to primary socket", file=sys.stderr)
                    raise
                await asyncio.sleep(0.2 * (attempt + 1))

        set_proxy_client(client)
        try:
            async with stdio_server() as (read, write):
                await mcp_server.run(read, write, mcp_server.create_initialization_options())
        finally:
            stop.set()
            await client.close()
            if promoted:
                await runtime.stop()
                lock_mgr.release()
            watchdog.cancel()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
