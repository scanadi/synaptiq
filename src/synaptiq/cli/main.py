"""Synaptiq CLI — Graph-powered code intelligence engine."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from synaptiq import __version__

console = Console()
_stderr_console = Console(stderr=True)


def _write_meta(data_dir: Path, repo_path: Path, result: object) -> None:
    """Write meta.json with index stats."""
    meta = {
        "version": __version__,
        "name": repo_path.name,
        "path": str(repo_path),
        "stats": {
            "files": result.files,
            "symbols": result.symbols,
            "relationships": result.relationships,
            "clusters": result.clusters,
            "flows": result.processes,
            "dead_code": result.dead_code,
            "coupled_pairs": result.coupled_pairs,
            "embeddings": result.embeddings,
        },
        "last_indexed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _load_storage(repo_path: Path | None = None) -> "KuzuBackend":  # noqa: F821
    """Load the KuzuDB backend for the given or current repo."""
    from synaptiq.core.storage.kuzu_backend import KuzuBackend

    target = (repo_path or Path.cwd()).resolve()
    db_path = target / ".synaptiq" / "kuzu"
    if not db_path.exists():
        console.print(
            f"[red]Error:[/red] No index found at {target}. Run 'synaptiq analyze' first."
        )
        raise typer.Exit(code=1)

    storage = KuzuBackend()
    storage.initialize(db_path, read_only=True)
    return storage


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
) -> None:
    """Index a repository into a knowledge graph."""
    from synaptiq.core.daemon.lock import LockManager
    from synaptiq.core.ingestion.pipeline import PipelineResult, run_pipeline
    from synaptiq.core.storage.kuzu_backend import KuzuBackend

    repo_path = path.resolve()
    if not repo_path.is_dir():
        _stderr_console.print(f"[red]Error:[/red] {repo_path} is not a directory.")
        raise typer.Exit(code=1)

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
            _reindex_via_server(existing.socket, full=full)
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

        db_path = data_dir / "kuzu"

        storage = KuzuBackend()
        try:
            storage.initialize(db_path)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "primary key" in msg or "corrupt" in msg:
                _stderr_console.print("[yellow]Corrupted index detected, rebuilding...[/yellow]")
                storage.close()
                if db_path.is_dir():
                    shutil.rmtree(db_path, ignore_errors=True)
                else:
                    db_path.unlink(missing_ok=True)
                    for suffix in (".wal", ".shadow"):
                        db_path.with_suffix(suffix).unlink(missing_ok=True)
                (data_dir / "meta.json").unlink(missing_ok=True)
                storage = KuzuBackend()
                storage.initialize(db_path)
            else:
                raise

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
    from synaptiq.mcp.tools import handle_query

    storage = _load_storage()
    result = handle_query(storage, q, limit=limit)
    console.print(result)
    storage.close()


@app.command()
def context(
    name: str = typer.Argument(..., help="Symbol name to inspect."),
) -> None:
    """Show 360-degree view of a symbol."""
    from synaptiq.mcp.tools import handle_context

    storage = _load_storage()
    result = handle_context(storage, name)
    console.print(result)
    storage.close()


@app.command()
def impact(
    target: str = typer.Argument(..., help="Symbol to analyze blast radius for."),
    depth: int = typer.Option(3, "--depth", "-d", help="Traversal depth."),
) -> None:
    """Show blast radius of changing a symbol."""
    from synaptiq.mcp.tools import handle_impact

    storage = _load_storage()
    result = handle_impact(storage, target, depth=depth)
    console.print(result)
    storage.close()


@app.command(name="dead-code")
def dead_code() -> None:
    """List all detected dead code."""
    from synaptiq.mcp.tools import handle_dead_code

    storage = _load_storage()
    result = handle_dead_code(storage)
    console.print(result)
    storage.close()


@app.command()
def cypher(
    query: str = typer.Argument(..., help="Raw Cypher query to execute."),
) -> None:
    """Execute raw Cypher against the knowledge graph."""
    from synaptiq.mcp.tools import handle_cypher

    storage = _load_storage()
    result = handle_cypher(storage, query)
    console.print(result)
    storage.close()


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
) -> None:
    """Structural branch comparison."""
    from synaptiq.core.diff import diff_branches, format_diff

    repo_path = Path.cwd().resolve()
    try:
        result = diff_branches(repo_path, branch_range)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(format_diff(result))


@app.command()
def mcp() -> None:
    """Start MCP server (stdio transport)."""
    import asyncio

    from synaptiq.mcp.server import main as mcp_main

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

    from synaptiq.mcp.server import main as mcp_main

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
        _serve_proxy(existing.socket)


def _init_storage_with_index(repo_path: Path, data_dir: Path, *, output=console):
    """Initialise KuzuDB and run the first index if no meta.json exists."""
    from synaptiq.core.ingestion.pipeline import run_pipeline
    from synaptiq.core.storage.kuzu_backend import KuzuBackend

    kuzu_path = data_dir / "kuzu"
    storage = KuzuBackend()

    try:
        storage.initialize(kuzu_path)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "primary key" in msg or "corrupt" in msg:
            print(
                "Corrupted index detected, rebuilding...",
                file=sys.stderr,
            )
            storage.close()
            if kuzu_path.is_dir():
                shutil.rmtree(kuzu_path, ignore_errors=True)
            else:
                kuzu_path.unlink(missing_ok=True)
                for suffix in (".wal", ".shadow"):
                    kuzu_path.with_suffix(suffix).unlink(missing_ok=True)
            meta_path = data_dir / "meta.json"
            meta_path.unlink(missing_ok=True)
            storage = KuzuBackend()
            storage.initialize(kuzu_path)
        else:
            raise

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


def _reindex_via_server(socket_path: str, *, full: bool = True) -> None:
    """Send a reindex request to a running synaptiq server via its Unix socket."""
    import asyncio

    from synaptiq.core.daemon.socket_client import SocketClient

    async def _do() -> str:
        client = SocketClient(Path(socket_path))
        await client.connect()
        try:
            return await client.reindex(full=full)
        finally:
            await client.close()

    console.print("[dim]Waiting for reindex to complete...[/dim]")
    try:
        result_str = asyncio.run(_do())
    except (ConnectionError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
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


def _handle_reindex(repo_path: Path, data_dir: Path, storage: object, full: bool = True) -> str:
    """Run a full reindex — called from the socket server dispatch."""
    import time

    from synaptiq.core.ingestion.pipeline import run_pipeline

    start = time.monotonic()
    _, result = run_pipeline(repo_path, storage, full=full)
    _write_meta(data_dir, repo_path, result)
    duration = time.monotonic() - start

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

    from synaptiq.core.daemon.rwlock import AsyncRWLock
    from synaptiq.core.daemon.socket_server import SocketServer
    from synaptiq.core.ingestion.watcher import watch_repo
    from synaptiq.mcp.server import dispatch_resource, dispatch_tool, set_rwlock, set_storage
    from synaptiq.mcp.server import server as mcp_server

    storage = _init_storage_with_index(repo_path, data_dir, output=_stderr_console)

    rwlock = AsyncRWLock()
    set_storage(storage)
    set_rwlock(rwlock)

    def dispatch(method: str, params: dict) -> str:
        if method == "ping":
            return "pong"
        if method == "reindex":
            return _handle_reindex(repo_path, data_dir, storage, full=params.get("full", True))
        if method == "tool":
            return dispatch_tool(params.get("name", ""), params.get("arguments", {}), storage)
        if method == "resource":
            return dispatch_resource(params.get("uri", ""), storage)
        return f"Unknown method: {method}"

    socket_server = SocketServer(
        lock_mgr.socket_path, dispatch, rwlock=rwlock, write_methods={"reindex"}
    )

    async def _run() -> None:
        import signal

        stop = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        await socket_server.start()
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
                watch_task = asyncio.create_task(
                    watch_repo(repo_path, storage, stop_event=stop, rwlock=rwlock)
                )

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
                    watch_task = asyncio.create_task(
                        watch_repo(repo_path, storage, stop_event=stop, rwlock=rwlock)
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
            await socket_server.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        storage.close()
        lock_mgr.release()


def _serve_proxy(socket_path: str) -> None:
    """Run as proxy: MCP over stdio, forwarding to primary via socket."""
    import asyncio

    from synaptiq.core.daemon.socket_client import SocketClient
    from synaptiq.mcp.server import set_proxy_client

    client = SocketClient(Path(socket_path))

    async def _run() -> None:
        from mcp.server.stdio import stdio_server

        from synaptiq.mcp.server import server as mcp_server

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
            await client.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
