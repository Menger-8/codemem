"""Memory evolution mechanism inspired by A-Mem.

When new memories are added, related old memories get their context
updated — simulating how humans re-understand old knowledge when
learning new things.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codemem.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class MemoryEvolver:
    """Handles memory evolution: new memories update related old ones.

    Inspired by A-Mem's memory evolution mechanism — when a new note
    arrives, we find its top-k neighbors and ask the LLM whether
    the new note provides additional context for them.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.client = None  # Lazy init

    @property
    def llm(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()
        return self.client

    def maybe_evolve(self, new_note_id: str, similarity_threshold: float = 0.65) -> list[str]:
        """Check if a new note should trigger evolution of related memories.

        Returns IDs of evolved (updated) memories.
        """
        new_note = self.memory.store.get_note(new_note_id)
        if not new_note or not new_note.embedding:
            return []

        # Find similar notes in the same layer
        related = self.memory.store.search_notes(
            new_note.embedding,
            layer=new_note.layer,
            top_k=5,
        )

        evolved_ids = []
        candidates = [(note, score) for note, score in related
                      if note.id != new_note_id and score > similarity_threshold]

        if not candidates:
            return []

        # Ask LLM which candidates should be evolved
        evolved = self._ask_llm_to_evolve(new_note, candidates)
        for note_id, new_context in evolved:
            note = self.memory.store.get_note(note_id)
            if note:
                old_context = note.context
                note.context = new_context
                note.updated_at = datetime.now()
                self.memory.store.store_note(note)
                evolved_ids.append(note_id)
                logger.info(f"Evolved memory {note_id[:8]}: context updated")

                # Also establish links
                from codemem.memory.models import MemoryLink
                link = MemoryLink(target_id=new_note_id, relationship="evolved_by")
                if link not in note.links:
                    note.links.append(link)
                    self.memory.store.store_note(note)

                reverse = MemoryLink(target_id=note_id, relationship="evolves")
                if reverse not in new_note.links:
                    new_note.links.append(reverse)
                    self.memory.store.store_note(new_note)

        return evolved_ids

    def _ask_llm_to_evolve(
        self,
        new_note,
        candidates: list,
    ) -> list[tuple[str, str]]:
        """Ask LLM which candidate memories should have their context updated."""
        candidates_text = "\n".join(
            f"- ID: {n.id}\n  Content: {n.content}\n  Context: {n.context}\n  Similarity: {s:.3f}"
            for n, s in candidates
        )

        prompt = f"""You are analyzing whether new information should update the context of existing memories.

NEW MEMORY:
- Content: {new_note.content}
- Keywords: {new_note.keywords}
- Tags: {new_note.tags}

EXISTING RELATED MEMORIES:
{candidates_text}

For each existing memory that would benefit from updated context (because the new memory
provides additional, complementary, or corrective information), output an evolution.

Only evolve memories where the new information genuinely enriches understanding.
Do NOT evolve if the memories are merely about similar topics without adding new context."""

        try:
            response = self.llm.messages.create(
                model=self.memory.config.api.model,
                max_tokens=1024,
                temperature=0.0,
                system="You are a memory evolution analyzer. Output JSON only.",
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "name": "evolve_memories",
                    "description": "Select memories to evolve with updated context",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "evolutions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "memory_id": {"type": "string"},
                                        "new_context": {"type": "string", "description": "Updated context description"},
                                    },
                                    "required": ["memory_id", "new_context"],
                                },
                            }
                        },
                        "required": ["evolutions"],
                    },
                }],
            )

            results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "evolve_memories":
                    for evo in block.input.get("evolutions", []):
                        results.append((evo["memory_id"], evo["new_context"]))
            return results
        except Exception as e:
            logger.warning(f"Evolution LLM call failed: {e}")
            return []


class LinkBuilder:
    """Builds links between related memories.

    Inspired by A-Mem's link generation: when a new note arrives,
    find top-k similar notes and let LLM decide meaningful links.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    def build_links(self, note_id: str, top_k: int = 5, threshold: float = 0.5) -> list[str]:
        """Find and create links for a memory note.

        Returns IDs of linked notes.
        """
        note = self.memory.store.get_note(note_id)
        if not note or not note.embedding:
            return []

        candidates = self.memory.store.search_notes(
            note.embedding,
            layer=note.layer,
            top_k=top_k + 1,  # +1 to exclude self
        )

        linked_ids = []
        for candidate, score in candidates:
            if candidate.id == note_id or score < threshold:
                continue

            # Determine relationship type
            relationship = self._classify_relationship(note, candidate, score)

            from codemem.memory.models import MemoryLink
            # Add forward link
            link = MemoryLink(target_id=candidate.id, relationship=relationship)
            if link not in note.links:
                note.links.append(link)
                self.memory.store.store_note(note)

            # Add reverse link
            reverse_rel = self._reverse_relationship(relationship)
            reverse_link = MemoryLink(target_id=note_id, relationship=reverse_rel)
            if reverse_link not in candidate.links:
                candidate.links.append(reverse_link)
                self.memory.store.store_note(candidate)

            linked_ids.append(candidate.id)

        return linked_ids

    @staticmethod
    def _classify_relationship(note_a, note_b, score: float) -> str:
        """Classify the relationship between two notes based on content overlap."""
        a_words = set(note_a.content.lower().split())
        b_words = set(note_b.content.lower().split())
        overlap = len(a_words & b_words) / max(len(a_words | b_words), 1)

        if overlap > 0.5:
            return "related_to"
        elif score > 0.8:
            return "similar_to"
        elif any(t in note_b.tags for t in note_a.tags):
            return "same_topic"
        else:
            return "related_to"

    @staticmethod
    def _reverse_relationship(rel: str) -> str:
        """Get the reverse of a relationship type."""
        reverse_map = {
            "related_to": "related_to",
            "similar_to": "similar_to",
            "same_topic": "same_topic",
            "supersedes": "superseded_by",
            "superseded_by": "supersedes",
            "evolves": "evolved_by",
            "evolved_by": "evolves",
        }
        return reverse_map.get(rel, "related_to")
