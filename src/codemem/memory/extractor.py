"""Fact extraction pipeline inspired by Mem0.

After each interaction, the LLM extracts key facts and decides
ADD/UPDATE/DELETE/NOOP for each fact.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic

from codemem.memory.models import FactOperation, MemoryNote

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system. Analyze conversations and extract facts worth remembering for a coding assistant.

Focus on extracting:
1. **User preferences**: coding style, language preferences, frameworks, naming conventions
2. **Project architecture**: tech stack, directory structure, design patterns, dependencies
3. **Coding patterns**: how the user likes tests written, error handling preferences, commit style
4. **Domain knowledge**: business logic, API contracts, database schema
5. **Decisions made**: architectural choices, trade-offs discussed, agreed-upon approaches

For each fact, decide:
- ADD: New fact not currently in memory
- UPDATE: Existing fact needs revision (provide the fact being updated)
- DELETE: A stored fact is now wrong or irrelevant
- NOOP: Nothing worth remembering from this interaction

Be selective. Only extract facts that will genuinely help in future coding sessions.
Do NOT extract: transient errors, one-time commands, obvious/universal facts."""

EXTRACTION_TOOL = {
    "name": "extract_facts",
    "description": "Extract and manage facts from the conversation",
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["ADD", "UPDATE", "DELETE", "NOOP"],
                            "description": "What to do with this fact"
                        },
                        "content": {
                            "type": "string",
                            "description": "The fact content (for ADD/UPDATE)"
                        },
                        "target_id": {
                            "type": "string",
                            "description": "ID of existing fact to update/delete (for UPDATE/DELETE)"
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key concepts in this fact"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Category tags (e.g., 'tech-stack', 'preference', 'architecture')"
                        },
                    },
                    "required": ["operation"],
                },
            }
        },
        "required": ["facts"],
    },
}


class FactExtractor:
    """Extracts facts from interactions using LLM tool calls.

    Implements Mem0's extraction-then-update pipeline:
    1. After each interaction, send recent conversation to LLM
    2. LLM decides ADD/UPDATE/DELETE/NOOP for each extracted fact
    3. Apply the operations to the memory store
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = anthropic.Anthropic()

    def extract(
        self,
        user_message: str,
        assistant_response: str,
        existing_facts: Optional[list[MemoryNote]] = None,
    ) -> list[FactOperation]:
        """Extract facts from an interaction.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            existing_facts: Currently stored facts (for UPDATE/DELETE decisions)

        Returns:
            List of fact operations to apply
        """
        # Build context with existing facts
        existing_text = ""
        if existing_facts:
            existing_text = "\n\nEXISTING FACTS IN MEMORY:\n"
            for f in existing_facts[:20]:  # Limit to avoid token overflow
                existing_text += f"- ID: {f.id} | {f.content}\n"

        conversation = f"""CONVERSATION:
User: {user_message[:2000]}
Assistant: {assistant_response[:2000]}{existing_text}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": conversation}],
                tools=[EXTRACTION_TOOL],
            )

            operations = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_facts":
                    for fact_data in block.input.get("facts", []):
                        try:
                            op = FactOperation(**fact_data)
                            if op.operation != "NOOP":
                                operations.append(op)
                        except Exception as e:
                            logger.warning(f"Failed to parse fact operation: {e}")

            return operations

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return []

    def extract_episode_facts(
        self,
        user_message: str,
        assistant_response: str,
    ) -> list[dict]:
        """Extract structured facts for episodic memory (subject-predicate-object).

        Returns list of dicts with 'subject', 'predicate', 'object' keys.
        """
        prompt = f"""Extract structured facts from this coding interaction as (subject, predicate, object) triples.

User: {user_message[:1500]}
Assistant: {assistant_response[:1500]}

Examples:
- ("auth module", "uses", "JWT refresh tokens")
- ("User", "prefers", "TypeScript over JavaScript")
- ("Database", "is", "PostgreSQL with Prisma ORM")
- ("API", "follows", "REST conventions")"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                system="Extract structured facts as JSON triples.",
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "name": "extract_triples",
                    "description": "Extract subject-predicate-object triples",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "triples": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subject": {"type": "string"},
                                        "predicate": {"type": "string"},
                                        "object": {"type": "string"},
                                    },
                                    "required": ["subject", "predicate", "object"],
                                },
                            }
                        },
                        "required": ["triples"],
                    },
                }],
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_triples":
                    return block.input.get("triples", [])

        except Exception as e:
            logger.warning(f"Episode fact extraction failed: {e}")

        return []
