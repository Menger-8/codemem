"""Tests for the memory system (models, stores, manager, compressor, evolution)."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest


# --- Model Tests ---

class TestMemoryModels:
    """Test Pydantic data models."""

    def test_memory_note_creation(self):
        from codemem.memory.models import MemoryNote
        note = MemoryNote(layer="semantic", content="Test fact", keywords=["test"])
        assert note.id  # auto-generated
        assert note.layer == "semantic"
        assert note.content == "Test fact"
        assert note.keywords == ["test"]
        assert note.tags == []
        assert note.access_count == 0

    def test_memory_note_custom_id(self):
        from codemem.memory.models import MemoryNote
        note = MemoryNote(id="custom-id", layer="procedural", content="A skill")
        assert note.id == "custom-id"

    def test_memory_link(self):
        from codemem.memory.models import MemoryLink, MemoryNote
        link = MemoryLink(target_id="abc-123", relationship="related_to")
        note = MemoryNote(
            layer="semantic", content="Test",
            links=[link]
        )
        assert len(note.links) == 1
        assert note.links[0].target_id == "abc-123"

    def test_episodic_record(self):
        from codemem.memory.models import EpisodicFact, EpisodicRecord
        record = EpisodicRecord(
            session_id="sess-1",
            gist="Fixed authentication bug",
            facts=[EpisodicFact(subject="auth", predicate="fixed", object="login bug")],
            files_involved=["auth.py"],
            action_type="debug",
            outcome="success",
        )
        assert record.gist == "Fixed authentication bug"
        assert len(record.facts) == 1
        assert record.facts[0].subject == "auth"

    def test_memory_skill(self):
        from codemem.memory.models import MemorySkill
        skill = MemorySkill(
            name="test_skill",
            purpose="Test purpose",
            when_to_use="When testing",
            how_to_apply="Apply like this",
        )
        assert skill.name == "test_skill"
        assert skill.success_rate == 0.0
        assert skill.usage_count == 0

    def test_working_memory_compress(self):
        from codemem.memory.models import WorkingMemory
        wm = WorkingMemory(
            task_summary="Build a REST API",
            active_files=["api.py", "models.py"],
            user_preferences={"lang": "python"},
        )
        compressed = wm.compress()
        assert "Build a REST API" in compressed
        assert "api.py" in compressed

    def test_fact_operation(self):
        from codemem.memory.models import FactOperation
        op = FactOperation(operation="ADD", content="New fact", keywords=["key"])
        assert op.operation == "ADD"
        assert op.content == "New fact"

    def test_session(self):
        from codemem.memory.models import Session
        s = Session(project_path="/test/project")
        assert s.id  # auto-generated
        assert s.turn_count == 0
        assert s.summary == ""


# --- Store Tests ---

class TestMemoryStore:
    """Test SQLite-backed storage."""

    def _make_store(self):
        from codemem.memory.stores import MemoryStore
        self._td = tempfile.mkdtemp()
        return MemoryStore(Path(self._td) / "test.db")

    def test_store_and_retrieve_note(self):
        from codemem.memory.models import MemoryNote
        store = self._make_store()
        note = MemoryNote(layer="semantic", content="Python 3.12", keywords=["python"])
        store.store_note(note)
        loaded = store.get_note(note.id)
        assert loaded is not None
        assert loaded.content == "Python 3.12"
        store.close()

    def test_delete_note(self):
        from codemem.memory.models import MemoryNote
        store = self._make_store()
        note = MemoryNote(layer="semantic", content="To delete")
        store.store_note(note)
        assert store.delete_note(note.id)
        assert store.get_note(note.id) is None
        store.close()

    def test_search_notes(self):
        from codemem.memory.models import MemoryNote
        store = self._make_store()
        note1 = MemoryNote(layer="semantic", content="FastAPI backend", embedding=[0.1] * 384)
        note2 = MemoryNote(layer="semantic", content="React frontend", embedding=[0.9] * 384)
        store.store_note(note1)
        store.store_note(note2)
        results = store.search_notes([0.15] * 384, layer="semantic", top_k=2)
        assert len(results) == 2
        # First result should be more similar to query
        assert results[0][0].content == "FastAPI backend"
        store.close()

    def test_get_notes_by_layer(self):
        from codemem.memory.models import MemoryNote
        store = self._make_store()
        store.store_note(MemoryNote(layer="semantic", content="Fact"))
        store.store_note(MemoryNote(layer="procedural", content="Skill"))
        sem_notes = store.get_notes_by_layer("semantic")
        assert len(sem_notes) == 1
        assert sem_notes[0].content == "Fact"
        store.close()

    def test_store_and_search_episodes(self):
        from codemem.memory.models import EpisodicFact, EpisodicRecord
        store = self._make_store()
        ep = EpisodicRecord(
            session_id="s1",
            gist="Fixed auth bug",
            facts=[EpisodicFact(subject="auth", predicate="fixed", object="bug")],
            files_involved=["auth.py"],
            action_type="debug",
            outcome="success",
            embedding=[0.5] * 384,
        )
        store.store_episode(ep)
        results = store.search_episodes([0.5] * 384, top_k=1)
        assert len(results) == 1
        assert results[0][0].gist == "Fixed auth bug"
        store.close()

    def test_store_and_search_skills(self):
        from codemem.memory.models import MemorySkill
        store = self._make_store()
        skill = MemorySkill(
            name="test_skill", purpose="Test", when_to_use="When testing",
            how_to_apply="Apply it", embedding=[0.3] * 384,
        )
        store.store_skill(skill)
        results = store.search_skills([0.3] * 384, top_k=1)
        assert len(results) == 1
        assert results[0][0].name == "test_skill"
        store.close()

    def test_sessions(self):
        from codemem.memory.models import Session
        store = self._make_store()
        s = Session(project_path="/test")
        store.store_session(s)
        loaded = store.get_session(s.id)
        assert loaded is not None
        assert loaded.project_path == "/test"
        recent = store.get_recent_sessions()
        assert len(recent) == 1
        store.close()

    def test_working_memory(self):
        from codemem.memory.models import WorkingMemory
        store = self._make_store()
        wm = WorkingMemory(task_summary="Test task")
        store.save_working_memory(wm)
        loaded = store.load_working_memory()
        assert loaded.task_summary == "Test task"
        store.close()

    def test_stats(self):
        from codemem.memory.models import MemoryNote, MemorySkill
        store = self._make_store()
        store.store_note(MemoryNote(layer="semantic", content="Fact"))
        store.store_skill(MemorySkill(
            name="sk", purpose="P", when_to_use="W", how_to_apply="H",
        ))
        stats = store.get_stats()
        assert stats["semantic_notes"] == 1
        assert stats["skills"] == 1
        store.close()


# --- Compressor Tests ---

class TestCompressor:
    """Test the pre-compression pipeline."""

    def test_compress_interaction_removes_boilerplate(self):
        from codemem.memory.compressor import compress_interaction
        raw = "import os\nimport sys\n\nActual content here\n\n\n\n"
        result = compress_interaction(raw)
        assert "import os" not in result
        assert "Actual content" in result

    def test_compress_diff(self):
        from codemem.memory.compressor import compress_diff
        diff = """diff --git a/f.py b/f.py
--- a/f.py
+++ b/f.py
@@ -1,3 +1,3 @@
 context line
-old
+new
"""
        result = compress_diff(diff)
        assert "-old" in result
        assert "+new" in result

    def test_compress_error_output(self):
        from codemem.memory.compressor import compress_error_output
        error = "Traceback:\n  File 'main.py'\nZeroDivisionError: division by zero"
        result = compress_error_output(error)
        assert "ZeroDivisionError" in result

    def test_extract_topic_boundaries(self):
        from codemem.memory.compressor import extract_topic_boundaries
        text = "Hello\nHow are you\n\nError:\nTraceback\nTypeError"
        boundaries = extract_topic_boundaries(text)
        assert len(boundaries) >= 1


# --- Embedding Tests ---

class TestEmbedding:
    """Test embedding engine (with fallback)."""

    def test_fallback_embedding(self):
        from codemem.memory.manager import EmbeddingEngine
        engine = EmbeddingEngine.__new__(EmbeddingEngine)
        engine._model = None
        engine._model_name = "test"
        engine._dim = 384
        engine._use_fallback = True
        vec = engine.embed("test text")
        assert len(vec) == 384
        # Should be normalized
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_fallback_different_texts_different_vectors(self):
        from codemem.memory.manager import EmbeddingEngine
        engine = EmbeddingEngine.__new__(EmbeddingEngine)
        engine._model = None
        engine._model_name = "test"
        engine._dim = 384
        engine._use_fallback = True
        vec1 = engine.embed("python programming")
        vec2 = engine.embed("javascript web development")
        # Vectors should be different
        assert vec1 != vec2


# --- Manager Tests ---

class TestMemoryManager:
    """Test the central memory manager."""

    def _make_manager(self):
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"  # Force fallback
        from codemem.config import Config
        from codemem.memory.manager import MemoryManager
        config = Config()
        self._td = tempfile.mkdtemp()
        return MemoryManager(config, Path(self._td))

    def test_init_default_skills(self):
        mm = self._make_manager()
        count = mm.init_default_skills()
        assert count == 6
        skills = mm.store.get_all_skills()
        assert len(skills) == 6
        names = {s.name for s in skills}
        assert "debug_error" in names
        assert "write_tests" in names
        mm.close()

    def test_init_default_skills_idempotent(self):
        mm = self._make_manager()
        mm.init_default_skills()
        count = mm.init_default_skills()
        assert count == 0  # Already initialized
        mm.close()

    def test_apply_fact_operation_add(self):
        from codemem.memory.models import FactOperation
        mm = self._make_manager()
        op = FactOperation(operation="ADD", content="Test fact", keywords=["test"])
        note_id = mm.apply_fact_operation(op)
        assert note_id is not None
        note = mm.store.get_note(note_id)
        assert note.content == "Test fact"
        mm.close()

    def test_apply_fact_operation_delete(self):
        from codemem.memory.models import FactOperation
        mm = self._make_manager()
        op = FactOperation(operation="ADD", content="To delete")
        note_id = mm.apply_fact_operation(op)
        del_op = FactOperation(operation="DELETE", target_id=note_id)
        mm.apply_fact_operation(del_op)
        assert mm.store.get_note(note_id) is None
        mm.close()

    def test_store_episode(self):
        mm = self._make_manager()
        ep_id = mm.store_episode(
            session_id="s1",
            gist="Fixed bug",
            facts=[{"subject": "auth", "predicate": "fixed", "object": "bug"}],
            files=["auth.py"],
            action_type="debug",
            outcome="success",
        )
        assert ep_id is not None
        mm.close()

    def test_assemble_context(self):
        from codemem.memory.models import FactOperation
        mm = self._make_manager()
        # Store some facts
        mm.apply_fact_operation(FactOperation(
            operation="ADD", content="Project uses Python", keywords=["python"],
        ))
        context = mm.assemble_context("How do I add a route?")
        assert isinstance(context, str)
        mm.close()

    def test_update_working_memory(self):
        mm = self._make_manager()
        mm.update_working_memory(task_summary="New task", active_files=["main.py"])
        assert mm.working_memory.task_summary == "New task"
        assert "main.py" in mm.working_memory.active_files
        mm.close()

    def test_stats(self):
        mm = self._make_manager()
        stats = mm.get_stats()
        assert "semantic_notes" in stats
        assert "skills" in stats
        mm.close()
