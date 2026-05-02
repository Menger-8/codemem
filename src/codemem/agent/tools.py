"""Tool definitions and execution for the coding agent.

Provides file operations, shell commands, git integration,
memory management tools, and code search.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

# Tool definitions for Claude API tool_use
MEMORY_TOOLS = [
    {
        "name": "memory_store",
        "description": "Store a new memory (fact, skill, or episode). Use this to remember important information about the project, user preferences, or coding patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "layer": {
                    "type": "string",
                    "enum": ["semantic", "procedural"],
                    "description": "Memory layer: 'semantic' for facts, 'procedural' for skills"
                },
                "content": {"type": "string", "description": "The memory content to store"},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "Key concepts"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Category tags"},
            },
            "required": ["layer", "content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search stored memories by semantic similarity. Use this to recall relevant facts, past episodes, or skills.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "layer": {
                    "type": "string",
                    "enum": ["semantic", "episodic", "procedural"],
                    "description": "Which memory layer to search (optional, searches all if omitted)"
                },
                "top_k": {"type": "integer", "description": "Number of results (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_delete",
        "description": "Delete a stored memory by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory ID to delete"},
            },
            "required": ["memory_id"],
        },
    },
]

CODING_TOOLS = [
    {
        "name": "file_read",
        "description": "Read the contents of a file. Returns the file content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "offset": {"type": "integer", "description": "Line number to start from (0-based)"},
                "limit": {"type": "integer", "description": "Max lines to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_write",
        "description": "Write content to a file. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_edit",
        "description": "Edit a file by replacing exact string matches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_string": {"type": "string", "description": "Exact string to find and replace"},
                "new_string": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "shell_exec",
        "description": "Execute a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "code_search",
        "description": "Search for a pattern in code files using regex.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search in (default: current dir)"},
                "glob": {"type": "string", "description": "File glob pattern (e.g. '*.py')"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "git_status",
        "description": "Show git working tree status.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_diff",
        "description": "Show git diff of changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "staged": {"type": "boolean", "description": "Show staged changes (default false)"},
            },
        },
    },
    {
        "name": "git_log",
        "description": "Show recent git commit history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of commits to show (default 10)"},
            },
        },
    },
    {
        "name": "git_branch",
        "description": "Show current branch and list all branches.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

ALL_TOOLS = CODING_TOOLS + MEMORY_TOOLS


def get_tool_definitions() -> list[dict]:
    """Return all tool definitions for Claude API."""
    return ALL_TOOLS


class ToolExecutor:
    """Executes tool calls from the agent."""

    def __init__(self, memory_manager, project_dir: str = "."):
        self.memory = memory_manager
        self.project_dir = Path(project_dir).resolve()

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result as a string."""
        handlers = {
            "file_read": self._file_read,
            "file_write": self._file_write,
            "file_edit": self._file_edit,
            "shell_exec": self._shell_exec,
            "code_search": self._code_search,
            "git_status": self._git_status,
            "git_diff": self._git_diff,
            "git_log": self._git_log,
            "git_branch": self._git_branch,
            "memory_store": self._memory_store,
            "memory_search": self._memory_search,
            "memory_delete": self._memory_delete,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            return handler(**tool_input)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _file_read(self, path: str, offset: int = 0, limit: int = 0) -> str:
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if offset:
            lines = lines[offset:]
        if limit:
            lines = lines[:limit]
        numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    def _file_write(self, path: str, content: str) -> str:
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"

    def _file_edit(self, path: str, old_string: str, new_string: str) -> str:
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        text = file_path.read_text(encoding="utf-8")
        if old_string not in text:
            return f"Error: old_string not found in {path}"
        count = text.count(old_string)
        if count > 1:
            return f"Error: old_string matches {count} times in {path}. Provide more context to make it unique."
        new_text = text.replace(old_string, new_string, 1)
        file_path.write_text(new_text, encoding="utf-8")
        return f"Edited {path}: replaced 1 occurrence"

    def _shell_exec(self, command: str, timeout: int = 30) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_dir),
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output[:10000]  # Truncate large outputs
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"

    def _code_search(self, pattern: str, path: str = ".", glob: str = "") -> str:
        import re
        search_dir = self._resolve_path(path)
        if not search_dir.exists():
            return f"Error: Path not found: {path}"
        regex = re.compile(pattern)
        results = []
        files = list(search_dir.rglob(glob)) if glob else list(search_dir.rglob("*"))
        for f in files:
            if not f.is_file() or f.stat().st_size > 100000:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        rel = f.relative_to(self.project_dir)
                        results.append(f"{rel}:{i}: {line.strip()}")
                        if len(results) >= 50:
                            return "\n".join(results)
            except Exception:
                continue
        return "\n".join(results) if results else "No matches found."

    def _git_status(self) -> str:
        return self._shell_exec("git status --short")

    def _git_diff(self, staged: bool = False) -> str:
        cmd = "git diff --staged" if staged else "git diff"
        return self._shell_exec(cmd)

    def _git_log(self, count: int = 10) -> str:
        return self._shell_exec(f"git log --oneline -{count}")

    def _git_branch(self) -> str:
        return self._shell_exec("git branch -a")

    def _memory_store(self, layer: str, content: str,
                      keywords: list[str] | None = None,
                      tags: list[str] | None = None) -> str:
        note_id = self.memory.apply_fact_operation(
            __import__("codemem.memory.models", fromlist=["FactOperation"]).FactOperation(
                operation="ADD",
                content=content,
                keywords=keywords or [],
                tags=tags or [],
            )
        )
        return f"Stored memory {note_id} in {layer} layer"

    def _memory_search(self, query: str, layer: str = "", top_k: int = 5) -> str:
        query_emb = self.memory.embedding.embed(query)
        if layer:
            results = self.memory.store.search_notes(query_emb, layer=layer, top_k=top_k)
        else:
            results = self.memory.store.search_notes(query_emb, top_k=top_k)
            # Also search episodes
            ep_results = self.memory.store.search_episodes(query_emb, top_k=top_k)
            for ep, score in ep_results:
                results.append((ep, score))
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

        if not results:
            return "No memories found."
        parts = []
        for item, score in results:
            if isinstance(item, MemoryNote):
                parts.append(f"[{item.layer}] {item.content} (score: {score:.3f})")
            elif isinstance(item, EpisodicRecord):
                parts.append(f"[episode] {item.gist} (score: {score:.3f})")
        return "\n".join(parts)

    def _memory_delete(self, memory_id: str) -> str:
        if self.memory.store.delete_note(memory_id):
            return f"Deleted memory {memory_id}"
        return f"Memory {memory_id} not found"

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.project_dir / p
        return p.resolve()
