"""REPL (Read-Eval-Print Loop) for the CodeMem agent.

Provides interactive terminal interface with slash commands,
streaming output, and session management.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from codemem.agent.orchestrator import AgentOrchestrator
from codemem.config import Config
from codemem.memory.manager import MemoryManager


console = Console()


def print_welcome() -> None:
    """Print the welcome banner."""
    console.print(Panel(
        "[bold cyan]CodeMem[/bold cyan] - CLI Coding Agent with 4-Layer Memory\n"
        "[dim]Type your message to chat. Use /help for commands.[/dim]",
        border_style="cyan",
    ))


def print_help() -> None:
    """Print available slash commands."""
    table = Table(title="Commands", show_header=True, header_style="bold")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    table.add_row("/memory search <query>", "Search all memory layers")
    table.add_row("/memory list [layer]", "List memories (semantic/episodic/procedural)")
    table.add_row("/memory stats", "Show memory statistics")
    table.add_row("/memory graph", "Show memory link graph (ASCII)")
    table.add_row("/memory evolve", "Trigger memory evolution pass")
    table.add_row("/skill list", "List available skills")
    table.add_row("/skill evolve", "Trigger skill evolution from failures")
    table.add_row("/sessions", "List recent sessions")
    table.add_row("/compact", "Clear conversation context")
    table.add_row("/cost", "Show token usage and costs")
    table.add_row("/config", "Show/edit configuration (set/get/edit/list)")
    table.add_row("/help", "Show this help")
    table.add_row("/quit", "Exit CodeMem")
    console.print(table)


def handle_slash_command(cmd: str, orchestrator: AgentOrchestrator, memory: MemoryManager) -> bool:
    """Handle a slash command. Returns True if handled, False to exit."""
    parts = cmd.strip().split(maxsplit=2)
    command = parts[0].lower()

    if command in ("/quit", "/exit", "/q"):
        return False

    if command == "/help":
        print_help()

    elif command == "/compact":
        orchestrator.clear_conversation()
        console.print("[dim]Conversation cleared. Memory preserved.[/dim]")

    elif command == "/memory":
        if len(parts) < 2:
            console.print("[yellow]Usage: /memory <search|list|stats> [args][/yellow]")
            return True
        subcmd = parts[1].lower()
        if subcmd == "search" and len(parts) >= 3:
            query = parts[2]
            query_emb = memory.embedding.embed(query)
            results = memory.store.search_notes(query_emb, top_k=10)
            if results:
                for note, score in results:
                    console.print(f"  [cyan][{note.layer}][/cyan] {note.content} [dim](score: {score:.3f})[/dim]")
            else:
                console.print("[dim]No memories found.[/dim]")
        elif subcmd == "list":
            layer = parts[2] if len(parts) >= 3 else None
            if layer:
                notes = memory.store.get_notes_by_layer(layer)
            else:
                notes = []
                for l in ("semantic", "episodic", "procedural"):
                    notes.extend(memory.store.get_notes_by_layer(l))
            if notes:
                for note in notes:
                    console.print(f"  [cyan][{note.layer}][/cyan] {note.content[:100]}")
            else:
                console.print("[dim]No memories stored yet.[/dim]")
        elif subcmd == "stats":
            stats = memory.get_stats()
            table = Table(title="Memory Statistics")
            table.add_column("Layer", style="cyan")
            table.add_column("Count", justify="right")
            table.add_row("Semantic Notes", str(stats.get("semantic_notes", 0)))
            table.add_row("Episodic Records", str(stats.get("episodes", 0)))
            table.add_row("Procedural Skills", str(stats.get("skills", 0)))
            table.add_row("Sessions", str(stats.get("sessions", 0)))
            console.print(table)

    elif command == "/sessions":
        sessions = memory.store.get_recent_sessions(limit=10)
        if sessions:
            table = Table(title="Recent Sessions")
            table.add_column("ID", style="dim")
            table.add_column("Started")
            table.add_column("Turns", justify="right")
            table.add_column("Summary")
            for s in sessions:
                table.add_row(
                    s.id[:8],
                    s.started_at.strftime("%Y-%m-%d %H:%M"),
                    str(s.turn_count),
                    s.summary[:50] if s.summary else "-",
                )
            console.print(table)
        else:
            console.print("[dim]No sessions yet.[/dim]")

    elif command == "/cost":
        # Show token usage from orchestrator
        usage = orchestrator.token_usage
        table = Table(title="Token Usage")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Input Tokens", f"{usage.get('input_tokens', 0):,}")
        table.add_row("Output Tokens", f"{usage.get('output_tokens', 0):,}")
        table.add_row("Total Tokens", f"{usage.get('total_tokens', 0):,}")
        # Estimate cost (Claude Sonnet: $3/1M input, $15/1M output)
        input_cost = usage.get('input_tokens', 0) * 3 / 1_000_000
        output_cost = usage.get('output_tokens', 0) * 15 / 1_000_000
        table.add_row("Estimated Cost", f"${input_cost + output_cost:.4f}")
        console.print(table)

    elif command == "/memory":
        if len(parts) < 2:
            console.print("[yellow]Usage: /memory <search|list|stats|graph|evolve> [args][/yellow]")
            return True
        subcmd = parts[1].lower()
        if subcmd == "graph":
            _show_memory_graph(memory)
        elif subcmd == "evolve":
            console.print("[dim]Running memory evolution pass...[/dim]")
            notes = memory.store.get_notes_by_layer("semantic")
            evolved = 0
            for note in notes[:10]:
                memory.evolver.maybe_evolve(note.id, similarity_threshold=0.6)
                evolved += 1
            console.print(f"[green]Evolved {evolved} memories.[/green]")

    elif command == "/skill":
        if len(parts) < 2:
            console.print("[yellow]Usage: /skill <list|evolve> [args][/yellow]")
            return True
        subcmd = parts[1].lower()
        if subcmd == "list":
            skills = memory.store.get_all_skills()
            if skills:
                table = Table(title="Skills")
                table.add_column("Name", style="cyan")
                table.add_column("Purpose")
                table.add_column("Success Rate", justify="right")
                table.add_column("Uses", justify="right")
                for s in skills:
                    table.add_row(
                        s.name,
                        s.purpose[:60],
                        f"{s.success_rate:.0%}" if s.success_rate > 0 else "-",
                        str(s.usage_count),
                    )
                console.print(table)
            else:
                console.print("[dim]No skills stored yet.[/dim]")
        elif subcmd == "evolve":
            console.print("[dim]Skill evolution requires interaction history.[/dim]")

    elif command == "/config":
        if len(parts) < 2:
            _show_config(orchestrator)
        else:
            subcmd = parts[1].lower()
            if subcmd == "set" and len(parts) >= 4:
                key = parts[2]
                value = parts[3]
                if orchestrator.config.set_value(key, value):
                    cfg_path = Path(orchestrator.config.data_dir) / "config.json"
                    orchestrator.config.save(cfg_path)
                    console.print(f"[green]Set {key} = {value}[/green]")
                    # Rebuild client if API settings changed
                    if key.startswith("api."):
                        _rebuild_client(orchestrator)
                else:
                    console.print(f"[red]Unknown config key: {key}[/red]")
            elif subcmd == "get" and len(parts) >= 3:
                key = parts[2]
                val = orchestrator.config.get_value(key)
                if val is not None:
                    console.print(f"  {key} = {val}")
                else:
                    console.print(f"[red]Unknown config key: {key}[/red]")
            elif subcmd == "edit":
                cfg_path = Path(orchestrator.config.data_dir) / "config.json"
                if not cfg_path.exists():
                    orchestrator.config.save(cfg_path)
                import subprocess
                editor = "notepad" if sys.platform == "win32" else "vi"
                subprocess.run([editor, str(cfg_path)])
                # Reload config
                from codemem.config import Config
                orchestrator.config = Config.load(cfg_path)
                _rebuild_client(orchestrator)
                console.print("[green]Config reloaded.[/green]")
            elif subcmd == "list":
                _show_config(orchestrator)
            else:
                console.print("[yellow]Usage: /config [set <key> <value> | get <key> | edit | list][/yellow]")

    else:
        console.print(f"[yellow]Unknown command: {command}. Type /help for available commands.[/yellow]")

    return True


def _show_memory_graph(memory: MemoryManager) -> None:
    """Display an ASCII graph of memory links."""
    notes = []
    for layer in ("semantic", "episodic", "procedural"):
        notes.extend(memory.store.get_notes_by_layer(layer))

    if not notes:
        console.print("[dim]No memories to graph.[/dim]")
        return

    # Build adjacency from links
    nodes = {}  # id -> (label, layer)
    edges = []  # (from_id, to_id, relationship)
    for note in notes:
        label = note.content[:40] + ("..." if len(note.content) > 40 else "")
        nodes[note.id] = (label, note.layer)
        for link in note.links:
            edges.append((note.id, link.target_id, link.relationship))

    # Render as ASCII
    console.print(Panel("[bold]Memory Link Graph[/bold]", border_style="cyan"))

    # Group by layer
    for layer in ("semantic", "episodic", "procedural"):
        layer_notes = [n for n in notes if n.layer == layer]
        if not layer_notes:
            continue
        color = {"semantic": "green", "episodic": "yellow", "procedural": "magenta"}[layer]
        console.print(f"\n[bold {color}]{layer.upper()}[/bold {color}]")
        for note in layer_notes:
            label = note.content[:50] + ("..." if len(note.content) > 50 else "")
            link_count = len(note.links)
            link_str = f" [{link_count} links]" if link_count > 0 else ""
            console.print(f"  [{color}]{note.id[:6]}[/{color}] {label}{link_str}")

    # Show edges
    if edges:
        console.print(f"\n[bold]Links ({len(edges)} total):[/bold]")
        for src, dst, rel in edges[:20]:
            src_label = nodes.get(src, ("???", "??"))[0][:30]
            dst_label = nodes.get(dst, ("???", "??"))[0][:30]
            console.print(f"  {src_label} --[{rel}]--> {dst_label}")
        if len(edges) > 20:
            console.print(f"  [dim]... and {len(edges) - 20} more[/dim]")
    else:
        console.print("\n[dim]No links between memories yet.[/dim]")


def _show_config(orchestrator: AgentOrchestrator) -> None:
    """Display current configuration."""
    cfg = orchestrator.config
    table = Table(title="Configuration", show_header=True, header_style="bold")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("api.api_key", cfg.get_value("api.api_key"))
    table.add_row("api.base_url", cfg.get_value("api.base_url") or "(default)")
    table.add_row("api.model", cfg.api.model)
    table.add_row("api.temperature", str(cfg.api.temperature))
    table.add_row("api.max_tokens", str(cfg.api.max_tokens))
    table.add_row("memory.auto_extract_facts", str(cfg.memory.auto_extract_facts))
    table.add_row("memory.auto_evolve", str(cfg.memory.auto_evolve))
    table.add_row("memory.embedding_model", cfg.memory.embedding_model)
    console.print(table)
    console.print("[dim]Use /config set <key> <value> to change. /config edit to open in editor.[/dim]")


def _rebuild_client(orchestrator: AgentOrchestrator) -> None:
    """Rebuild the Anthropic client after config changes."""
    import anthropic
    client_kwargs = {}
    api_key = orchestrator.config.api.get_effective_api_key()
    if api_key:
        client_kwargs["api_key"] = api_key
    base_url = orchestrator.config.api.get_effective_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url
    orchestrator.client = anthropic.Anthropic(**client_kwargs)


def run_repl(orchestrator: AgentOrchestrator, memory: MemoryManager, history_path: Optional[str] = None) -> None:
    """Run the interactive REPL."""
    print_welcome()

    history = FileHistory(history_path or ".codemem_history")
    session = PromptSession(history=history)

    while True:
        try:
            with patch_stdout():
                user_input = session.prompt("\n> ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            if not handle_slash_command(user_input, orchestrator, memory):
                break
            continue

        # Regular chat
        console.print()
        try:
            response = orchestrator.chat(
                user_input,
                on_token=lambda t: console.print(t, end="", highlight=False),
            )
            console.print()  # Newline after streaming
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
