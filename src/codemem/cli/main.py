"""CLI entry point for CodeMem."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console

from codemem.config import Config

console = Console()


@click.command()
@click.argument("project_dir", default=".", type=click.Path(exists=True))
@click.option("--resume", is_flag=True, help="Resume the last session")
@click.option("--session", "session_id", default=None, help="Resume a specific session")
@click.option("--config", "config_path", default=None, type=click.Path(), help="Path to config file")
@click.option("--memory-only", is_flag=True, help="Interactive memory management mode")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def main(
    project_dir: str,
    resume: bool,
    session_id: str | None,
    config_path: str | None,
    memory_only: bool,
    verbose: bool,
) -> None:
    """CodeMem: CLI coding agent with 4-layer memory system."""
    # Setup logging
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Resolve paths
    project = Path(project_dir).resolve()
    data_dir = project / ".codemem"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg_path = Path(config_path) if config_path else data_dir / "config.json"
    config = Config.load(cfg_path)
    config.data_dir = str(data_dir)

    # Save default config if it doesn't exist
    if not cfg_path.exists():
        config.save(cfg_path)

    # Initialize components
    from codemem.agent.orchestrator import AgentOrchestrator
    from codemem.agent.tools import ToolExecutor
    from codemem.cli.repl import run_repl
    from codemem.memory.manager import MemoryManager

    memory = MemoryManager(config, data_dir)
    memory.init_default_skills()  # Initialize default skill library
    tool_executor = ToolExecutor(memory, project_dir=str(project))
    orchestrator = AgentOrchestrator(config, memory, tool_executor)

    # Handle session
    from codemem.memory.models import Session
    from datetime import datetime

    if resume or session_id:
        sessions = memory.store.get_recent_sessions(limit=1)
        if session_id:
            session = memory.store.get_session(session_id)
        elif sessions:
            session = sessions[0]
        else:
            session = None

        if session:
            orchestrator.set_session(session.id)
            session.last_active = datetime.now()
            memory.store.store_session(session)
            console.print(f"[dim]Resumed session {session.id[:8]}[/dim]")
        else:
            console.print("[yellow]No session found. Starting new session.[/yellow]")
            new_session = Session(project_path=str(project))
            memory.store.store_session(new_session)
            orchestrator.set_session(new_session.id)
    else:
        new_session = Session(project_path=str(project))
        memory.store.store_session(new_session)
        orchestrator.set_session(new_session.id)

    # Run REPL
    history_path = str(data_dir / "history")
    run_repl(orchestrator, memory, history_path)

    # Cleanup
    memory.close()


if __name__ == "__main__":
    main()
