"""Tests for the tool executor."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


class TestToolExecutor:
    """Test tool execution."""

    def _make_executor(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        from codemem.config import Config
        from codemem.memory.manager import MemoryManager
        from codemem.agent.tools import ToolExecutor
        config = Config()
        self._td = tempfile.mkdtemp()
        self._project = Path(self._td) / "project"
        self._project.mkdir()
        mm = MemoryManager(config, Path(self._td) / "data")
        return ToolExecutor(mm, str(self._project)), mm

    def test_file_read(self):
        executor, mm = self._make_executor()
        test_file = self._project / "test.txt"
        test_file.write_text("hello\nworld\n")
        result = executor.execute("file_read", {"path": str(test_file)})
        assert "hello" in result
        assert "world" in result
        mm.close()

    def test_file_read_with_offset_limit(self):
        executor, mm = self._make_executor()
        test_file = self._project / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\n")
        result = executor.execute("file_read", {"path": str(test_file), "offset": 1, "limit": 2})
        assert "line2" in result
        assert "line3" in result
        assert "line1" not in result
        mm.close()

    def test_file_write(self):
        executor, mm = self._make_executor()
        test_file = self._project / "new.txt"
        result = executor.execute("file_write", {"path": str(test_file), "content": "new content"})
        assert "Written" in result
        assert test_file.read_text() == "new content"
        mm.close()

    def test_file_edit(self):
        executor, mm = self._make_executor()
        test_file = self._project / "edit.txt"
        test_file.write_text("old text here")
        result = executor.execute("file_edit", {"path": str(test_file), "old_string": "old", "new_string": "new"})
        assert "Edited" in result
        assert test_file.read_text() == "new text here"
        mm.close()

    def test_file_edit_not_found(self):
        executor, mm = self._make_executor()
        result = executor.execute("file_edit", {"path": "/nonexistent", "old_string": "a", "new_string": "b"})
        assert "Error" in result
        mm.close()

    def test_shell_exec(self):
        executor, mm = self._make_executor()
        result = executor.execute("shell_exec", {"command": "echo hello"})
        assert "hello" in result
        mm.close()

    def test_shell_exec_timeout(self):
        executor, mm = self._make_executor()
        result = executor.execute("shell_exec", {"command": "sleep 10", "timeout": 1})
        assert "timed out" in result or "Error" in result
        mm.close()

    def test_code_search(self):
        executor, mm = self._make_executor()
        (self._project / "main.py").write_text("def hello():\n    print('world')\n")
        result = executor.execute("code_search", {"pattern": "def hello", "path": str(self._project)})
        assert "hello" in result
        mm.close()

    def test_git_status(self):
        executor, mm = self._make_executor()
        result = executor.execute("git_status", {})
        assert isinstance(result, str)
        mm.close()

    def test_memory_store(self):
        executor, mm = self._make_executor()
        result = executor.execute("memory_store", {
            "layer": "semantic",
            "content": "Test fact",
            "keywords": ["test"],
        })
        assert "Stored" in result
        mm.close()

    def test_memory_search(self):
        executor, mm = self._make_executor()
        executor.execute("memory_store", {
            "layer": "semantic",
            "content": "Python FastAPI backend",
        })
        result = executor.execute("memory_search", {"query": "python web framework"})
        assert isinstance(result, str)
        mm.close()

    def test_memory_delete(self):
        executor, mm = self._make_executor()
        store_result = executor.execute("memory_store", {
            "layer": "semantic",
            "content": "To delete",
        })
        # Extract memory ID from result
        mem_id = store_result.split("Stored memory ")[1].split(" ")[0]
        result = executor.execute("memory_delete", {"memory_id": mem_id})
        assert "Deleted" in result
        mm.close()

    def test_unknown_tool(self):
        executor, mm = self._make_executor()
        result = executor.execute("nonexistent_tool", {})
        assert "Error" in result
        mm.close()

    def test_tool_definitions(self):
        from codemem.agent.tools import get_tool_definitions
        tools = get_tool_definitions()
        names = {t["name"] for t in tools}
        assert "file_read" in names
        assert "memory_store" in names
        assert "git_status" in names
        assert "git_log" in names
        assert "git_branch" in names
