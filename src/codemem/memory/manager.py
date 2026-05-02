"""Central memory manager orchestrating all 4 layers.

Implements fact extraction (Mem0-style), memory evolution (A-Mem-style),
and context assembly for the agent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from codemem.config import Config
from codemem.memory.models import (
    EpisodicFact,
    EpisodicRecord,
    FactOperation,
    MemoryLink,
    MemoryNote,
    MemorySkill,
    WorkingMemory,
)
from codemem.memory.stores import MemoryStore

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Compute text embeddings using sentence-transformers.

    Falls back to a simple hash-based embedding if the model can't be loaded
    (e.g., no internet access to download from HuggingFace).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self._model = None
        self._model_name = model_name
        self._dim = dim
        self._use_fallback = False

    def _ensure_model(self):
        """Lazy-load the model, setting fallback if it fails."""
        if self._model is not None or self._use_fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except Exception as e:
            logger.warning(f"Failed to load embedding model '{self._model_name}': {e}")
            logger.warning("Falling back to hash-based embeddings (reduced quality)")
            self._use_fallback = True

    def embed(self, text: str) -> list[float]:
        """Compute embedding for a single text."""
        self._ensure_model()
        if self._use_fallback:
            return self._fallback_embed(text)
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for multiple texts."""
        self._ensure_model()
        if self._use_fallback:
            return [self._fallback_embed(t) for t in texts]
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def _fallback_embed(self, text: str) -> list[float]:
        """Simple hash-based embedding fallback.

        Not as good as real embeddings but allows the system to function.
        Uses multiple hash functions to create a sparse-ish vector.
        """
        import hashlib
        vec = [0.0] * self._dim
        words = text.lower().split()
        for i, word in enumerate(words):
            for seed in range(3):
                h = hashlib.md5(f"{seed}:{word}".encode()).hexdigest()
                idx = int(h[:8], 16) % self._dim
                vec[idx] += 1.0 / (i + 1)
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class MemoryManager:
    """Orchestrates the 4-layer memory system.

    Manages storage, retrieval, fact extraction, memory evolution,
    and context assembly.
    """

    def __init__(self, config: Config, data_dir: Path):
        self.config = config
        self.store = MemoryStore(data_dir / "memory.db")
        self.embedding = EmbeddingEngine(config.memory.embedding_model)
        self.working_memory = self.store.load_working_memory()
        self._evolver = None
        self._link_builder = None
        self._extractor = None

    def close(self) -> None:
        self.store.close()

    @property
    def evolver(self):
        """Lazy-init memory evolver."""
        if self._evolver is None:
            from codemem.memory.evolution import MemoryEvolver
            self._evolver = MemoryEvolver(self)
        return self._evolver

    @property
    def link_builder(self):
        """Lazy-init link builder."""
        if self._link_builder is None:
            from codemem.memory.evolution import LinkBuilder
            self._link_builder = LinkBuilder(self)
        return self._link_builder

    @property
    def extractor(self):
        """Lazy-init fact extractor."""
        if self._extractor is None:
            from codemem.memory.extractor import FactExtractor
            self._extractor = FactExtractor(model=self.config.api.model)
        return self._extractor

    # --- Context Assembly (called before each LLM turn) ---

    def assemble_context(self, query: str) -> str:
        """Assemble memory context for injection into LLM prompt.

        Combines working memory + retrieved semantic/episodic/procedural memories.
        Budget: ~11K tokens total pre-conversation.
        """
        parts = []

        # Layer 1: Working Memory (always included, ~4K tokens)
        wm_text = self.working_memory.compress()
        if wm_text:
            parts.append(f"## Working Memory\n{wm_text}")

        # Layer 2: Semantic Memory (top-5 relevant facts, ~2K tokens)
        query_emb = self.embedding.embed(query)
        semantic_results = self.store.search_notes(
            query_emb, layer="semantic", top_k=self.config.memory.semantic_memory_top_k
        )
        if semantic_results:
            facts = []
            for note, score in semantic_results:
                note.access_count += 1
                self.store.store_note(note)
                facts.append(f"- {note.content}")
            parts.append("## Known Facts\n" + "\n".join(facts))

        # Layer 3: Episodic Memory (recent sessions + relevant episodes, ~2K tokens)
        recent_eps = self.store.get_recent_episodes(limit=3)
        if recent_eps:
            episodes = []
            for ep in recent_eps:
                episodes.append(f"- [{ep.action_type}] {ep.gist}")
            parts.append("## Recent Activity\n" + "\n".join(episodes))

        # Layer 4: Procedural Memory (top-2 relevant skills, ~1K tokens)
        skill_results = self.store.search_skills(
            query_emb, top_k=self.config.memory.procedural_memory_top_k
        )
        if skill_results:
            skills = []
            for skill, score in skill_results:
                skills.append(f"- **{skill.name}**: {skill.when_to_use}")
            parts.append("## Relevant Skills\n" + "\n".join(skills))

        return "\n\n".join(parts) if parts else ""

    # --- Fact Extraction (Mem0-style) ---

    def extract_facts(self, user_message: str, assistant_response: str) -> list[FactOperation]:
        """Extract facts from an interaction using LLM tool calls.

        This returns the tool schema for fact extraction - the actual LLM call
        is made by the orchestrator.
        """
        return []  # Actual extraction happens via tool calls in orchestrator

    def apply_fact_operation(self, op: FactOperation) -> Optional[str]:
        """Apply a fact operation (ADD/UPDATE/DELETE/NOOP)."""
        if op.operation == "NOOP":
            return None

        if op.operation == "ADD":
            note = MemoryNote(
                layer="semantic",
                content=op.content,
                keywords=op.keywords,
                tags=op.tags,
                embedding=self.embedding.embed(op.content),
            )
            self.store.store_note(note)
            self._maybe_evolve(note)
            return note.id

        if op.operation == "UPDATE" and op.target_id:
            existing = self.store.get_note(op.target_id)
            if existing:
                existing.content = op.content
                if op.keywords:
                    existing.keywords = op.keywords
                if op.tags:
                    existing.tags = op.tags
                existing.embedding = self.embedding.embed(op.content)
                existing.updated_at = __import__("datetime").datetime.now()
                self.store.store_note(existing)
                return existing.id

        if op.operation == "DELETE" and op.target_id:
            self.store.delete_note(op.target_id)
            return op.target_id

        return None

    # --- Memory Evolution (A-Mem-style) ---

    def _maybe_evolve(self, new_note: MemoryNote) -> None:
        """When a new memory is added, check if related memories need updating.

        Uses the evolver (LLM-based) and link_builder for automatic linking.
        """
        if not self.config.memory.auto_evolve:
            return

        # Build links with related memories
        self.link_builder.build_links(new_note.id, top_k=3, threshold=0.5)

        # Evolve related memories (LLM-based, more expensive)
        self.evolver.maybe_evolve(new_note.id, similarity_threshold=0.65)

    # --- Episodic Memory ---

    def store_episode(
        self,
        session_id: str,
        gist: str,
        facts: list[dict],
        files: list[str],
        action_type: str,
        outcome: str,
    ) -> str:
        """Store an episodic record."""
        episode = EpisodicRecord(
            session_id=session_id,
            gist=gist,
            facts=[EpisodicFact(**f) for f in facts],
            files_involved=files,
            action_type=action_type,
            outcome=outcome,
            embedding=self.embedding.embed(gist),
        )
        self.store.store_episode(episode)
        return episode.id

    # --- Working Memory ---

    def update_working_memory(self, **kwargs) -> None:
        """Update working memory fields."""
        for key, value in kwargs.items():
            if hasattr(self.working_memory, key):
                setattr(self.working_memory, key, value)
        self.store.save_working_memory(self.working_memory)

    def compress_working_memory(self, context_text: str) -> str:
        """Compress working memory when context gets too large.

        Returns a prompt for the LLM to generate a compressed summary.
        """
        return (
            "Please compress the following conversation context into a concise "
            "internal state summary (max 500 words). Preserve: key decisions made, "
            "files modified, current task status, user preferences discovered.\n\n"
            f"Context to compress:\n{context_text}"
        )

    # --- Procedural Memory (Skills) ---

    def store_skill(self, name: str, purpose: str, when_to_use: str,
                    how_to_apply: str, constraints: str = "") -> str:
        """Store a new skill."""
        skill = MemorySkill(
            name=name,
            purpose=purpose,
            when_to_use=when_to_use,
            how_to_apply=how_to_apply,
            constraints=constraints,
            embedding=self.embedding.embed(f"{purpose} {when_to_use}"),
        )
        self.store.store_skill(skill)
        return skill.id

    def init_default_skills(self) -> int:
        """Initialize the default skill library. Returns number of skills added."""
        existing = self.store.get_all_skills()
        if existing:
            return 0  # Already initialized

        defaults = [
            {
                "name": "debug_error",
                "purpose": "Diagnose and fix runtime errors, exceptions, and tracebacks",
                "when_to_use": "When the user encounters an error, exception, or traceback",
                "how_to_apply": "1. Read the full error message and traceback\n2. Identify the root cause (not just the symptom)\n3. Check if it's a dependency, syntax, or logic error\n4. Propose a minimal fix\n5. Verify the fix doesn't break other things",
                "constraints": "Focus on the root cause, not workarounds",
            },
            {
                "name": "refactor_code",
                "purpose": "Improve code structure without changing behavior",
                "when_to_use": "When code is messy, duplicated, or hard to maintain",
                "how_to_apply": "1. Understand current behavior (read tests if available)\n2. Identify code smells (duplication, long functions, deep nesting)\n3. Apply refactoring incrementally\n4. Run tests after each change\n5. Verify no behavior change",
                "constraints": "Never change behavior during refactoring. Keep changes small and testable.",
            },
            {
                "name": "write_tests",
                "purpose": "Write comprehensive tests for code",
                "when_to_use": "When the user asks to add tests or test coverage is low",
                "how_to_apply": "1. Read the code to understand behavior\n2. Identify edge cases and boundary conditions\n3. Write tests for happy path first, then edge cases\n4. Use descriptive test names that explain the scenario\n5. Follow existing test patterns in the project",
                "constraints": "Match existing test framework and style. Test behavior, not implementation.",
            },
            {
                "name": "explain_code",
                "purpose": "Explain how code works in clear, accessible language",
                "when_to_use": "When the user asks 'what does this do' or 'how does this work'",
                "how_to_apply": "1. Read the code and its dependencies\n2. Identify the high-level purpose\n3. Explain the flow step by step\n4. Highlight non-obvious decisions or patterns\n5. Relate to the user's knowledge level",
                "constraints": "Adapt explanation depth to the user's question. Don't over-explain simple code.",
            },
            {
                "name": "git_commit",
                "purpose": "Create well-structured git commits",
                "when_to_use": "When the user wants to commit changes",
                "how_to_apply": "1. Run git status and git diff to understand changes\n2. Group related changes logically\n3. Write a clear commit message (what and why, not how)\n4. Use conventional commit format if project uses it\n5. Stage specific files, not everything",
                "constraints": "Never commit secrets, .env files, or large binaries. Always let user review before pushing.",
            },
            {
                "name": "fix_import_error",
                "purpose": "Diagnose and fix Python/JS/TS import errors",
                "when_to_use": "When ModuleNotFoundError, ImportError, or 'cannot find module' appears",
                "how_to_apply": "1. Check the exact import path\n2. Verify the package is installed (pip list / npm list)\n3. Check virtual environment is activated\n4. Look for circular imports\n5. Check sys.path / NODE_PATH configuration\n6. Verify __init__.py exists for Python packages",
                "constraints": "Different ecosystems (Python/Node/Go) have different import mechanisms. Identify the language first.",
            },
        ]

        count = 0
        for skill_data in defaults:
            self.store_skill(**skill_data)
            count += 1

        logger.info(f"Initialized {count} default skills")
        return count

    def create_skill_from_interaction(
        self, user_msg: str, assistant_msg: str, success: bool
    ) -> Optional[str]:
        """Auto-create a skill from a successful interaction (MemSkill-style).

        If the interaction solved a recurring problem, extract it as a skill.
        """
        if not success:
            return None

        # Ask LLM if this interaction warrants a new skill
        try:
            response = self.extractor.client.messages.create(
                model=self.config.api.model,
                max_tokens=512,
                temperature=0.0,
                system="You are a skill extraction system. Determine if a coding interaction solved a generalizable problem that should become a reusable skill.",
                messages=[{
                    "role": "user",
                    "content": f"User: {user_msg[:1000]}\n\nAssistant: {assistant_msg[:1000]}\n\nShould this become a reusable skill? If yes, provide the skill details."
                }],
                tools=[{
                    "name": "create_skill",
                    "description": "Create a new skill if the interaction is generalizable",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "should_create": {"type": "boolean"},
                            "name": {"type": "string"},
                            "purpose": {"type": "string"},
                            "when_to_use": {"type": "string"},
                            "how_to_apply": {"type": "string"},
                        },
                        "required": ["should_create"],
                    },
                }],
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "create_skill":
                    if block.input.get("should_create"):
                        return self.store_skill(
                            name=block.input["name"],
                            purpose=block.input["purpose"],
                            when_to_use=block.input["when_to_use"],
                            how_to_apply=block.input["how_to_apply"],
                        )
        except Exception as e:
            logger.warning(f"Skill creation failed: {e}")

        return None

    # --- Statistics ---

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return self.store.get_stats()
