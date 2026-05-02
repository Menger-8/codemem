"""Agent orchestrator: the main conversation loop.

Manages context assembly, Claude API calls, tool dispatch,
and post-interaction memory extraction.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import anthropic

from codemem.agent.tools import ToolExecutor, get_tool_definitions
from codemem.config import Config
from codemem.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

FACT_EXTRACTION_PROMPT = """After each user interaction, extract key facts that should be remembered for future coding sessions.

For each fact, decide one of:
- ADD: New fact not currently stored
- UPDATE: Existing fact needs revision (provide target_id if known)
- DELETE: A stored fact is now wrong (provide target_id if known)
- NOOP: Nothing worth remembering

Focus on: user preferences, project architecture, tech stack, coding patterns, decisions made.
Do NOT store: transient errors, one-time commands, obvious facts."""

SYSTEM_PROMPT = """You are CodeMem, a coding assistant with a persistent memory system.

You have access to:
1. **Coding tools**: file read/write/edit, shell commands, git, code search
2. **Memory tools**: store and search memories across 4 layers

Your memory system has 4 layers:
- **Working Memory**: Current task context (auto-managed)
- **Semantic Memory**: Facts about the project, user preferences, tech stack
- **Episodic Memory**: Records of past interactions and what happened
- **Procedural Memory**: Reusable coding skills and patterns

When you learn something important (user preference, project architecture, coding pattern),
use memory_store to save it. When you need to recall something, use memory_search.

Be concise and direct. Focus on writing correct, secure code."""


class AgentOrchestrator:
    """Main agent loop: assemble context → call LLM → dispatch tools → update memory."""

    def __init__(self, config: Config, memory: MemoryManager, tool_executor: ToolExecutor):
        self.config = config
        self.memory = memory
        self.tool_executor = tool_executor
        # Build client kwargs from config
        client_kwargs = {}
        api_key = config.api.get_effective_api_key()
        if api_key:
            client_kwargs["api_key"] = api_key
        base_url = config.api.get_effective_base_url()
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        self.conversation: list[dict] = []
        self.session_id: Optional[str] = None
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def set_session(self, session_id: str) -> None:
        self.session_id = session_id

    def chat(self, user_message: str, on_token: Any = None) -> str:
        """Process a user message and return the assistant's response.

        Args:
            user_message: The user's input text
            on_token: Optional callback for streaming tokens

        Returns:
            The assistant's final text response
        """
        # 1. Assemble memory context
        memory_context = self.memory.assemble_context(user_message)

        # 2. Build the user message with memory context
        if memory_context:
            full_message = f"<memory_context>\n{memory_context}\n</memory_context>\n\n{user_message}"
        else:
            full_message = user_message

        self.conversation.append({"role": "user", "content": full_message})

        # 3. Call Claude API with tools
        response_text = self._call_llm(on_token)

        # 4. Post-interaction: extract facts (if enabled)
        if self.config.memory.auto_extract_facts:
            self._extract_and_store_facts(user_message, response_text)

        # 5. Update working memory
        self._update_working_memory(user_message, response_text)

        # 6. Store episodic record
        if self.session_id:
            self._store_episode(user_message, response_text)

        return response_text

    def _call_llm(self, on_token: Any = None) -> str:
        """Call Claude API with tool use support, handling tool call loops."""
        messages = list(self.conversation)
        tools = get_tool_definitions()

        while True:
            if self.config.api.stream and on_token:
                return self._stream_call(messages, tools, on_token)

            response = self.client.messages.create(
                model=self.config.api.model,
                max_tokens=self.config.api.max_tokens,
                temperature=self.config.api.temperature,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )

            # Track token usage
            self.token_usage["input_tokens"] += response.usage.input_tokens
            self.token_usage["output_tokens"] += response.usage.output_tokens
            self.token_usage["total_tokens"] += response.usage.input_tokens + response.usage.output_tokens

            # Process response
            tool_calls = []
            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(block)

            # If no tool calls, we're done
            if not tool_calls:
                final_text = "\n".join(text_parts)
                self.conversation.append({"role": "assistant", "content": response.content})
                return final_text

            # Execute tool calls and continue
            self.conversation.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tc in tool_calls:
                logger.info(f"Tool call: {tc.name}({json.dumps(tc.input, ensure_ascii=False)[:200]})")
                result = self.tool_executor.execute(tc.name, tc.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            self.conversation.append({"role": "user", "content": tool_results})

    def _stream_call(self, messages: list, tools: list, on_token: Any) -> str:
        """Stream the LLM response, handling tool calls."""
        text_parts = []
        tool_calls = []
        current_tool = None

        with self.client.messages.stream(
            model=self.config.api.model,
            max_tokens=self.config.api.max_tokens,
            temperature=self.config.api.temperature,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool = {"id": event.content_block.id, "name": event.content_block.name, "input": ""}
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        text_parts.append(event.delta.text)
                        if on_token:
                            on_token(event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        if current_tool:
                            current_tool["input"] += event.delta.partial_json
                elif event.type == "content_block_stop":
                    if current_tool:
                        try:
                            current_tool["input"] = json.loads(current_tool["input"])
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                        tool_calls.append(current_tool)
                        current_tool = None

            # Track token usage from stream
            try:
                final_msg = stream.get_final_message()
                if final_msg and final_msg.usage:
                    self.token_usage["input_tokens"] += final_msg.usage.input_tokens
                    self.token_usage["output_tokens"] += final_msg.usage.output_tokens
                    self.token_usage["total_tokens"] += final_msg.usage.input_tokens + final_msg.usage.output_tokens
            except Exception:
                pass  # Usage tracking is best-effort

        # If there were tool calls, execute them and loop
        if tool_calls:
            # Add assistant message with tool use blocks
            assistant_content = []
            for tc in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            self.conversation.append({"role": "assistant", "content": assistant_content})

            # Execute tools
            tool_results = []
            for tc in tool_calls:
                logger.info(f"Tool call: {tc['name']}")
                result = self.tool_executor.execute(tc["name"], tc["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                })

            self.conversation.append({"role": "user", "content": tool_results})

            # Continue conversation (recurse)
            return self._call_llm(on_token)

        # No tool calls - finalize
        final_text = "".join(text_parts)
        self.conversation.append({"role": "assistant", "content": final_text})
        return final_text

    def _extract_and_store_facts(self, user_msg: str, assistant_msg: str) -> None:
        """Use FactExtractor to extract and apply facts from the interaction."""
        try:
            # Get existing facts for UPDATE/DELETE context
            existing = self.memory.store.get_notes_by_layer("semantic")

            # Extract facts using the dedicated extractor
            operations = self.memory.extractor.extract(
                user_msg, assistant_msg, existing_facts=existing
            )

            for op in operations:
                result = self.memory.apply_fact_operation(op)
                if result:
                    logger.info(f"Extracted fact: {op.operation} - {op.content[:80]}")

            # Also try to create a skill if the interaction was successful
            if self._was_successful(assistant_msg):
                self.memory.create_skill_from_interaction(user_msg, assistant_msg, True)

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")

    @staticmethod
    def _was_successful(response: str) -> bool:
        """Heuristic: did the interaction seem successful?"""
        failure_indicators = ["error", "failed", "cannot", "unable", "sorry", "apologize"]
        response_lower = response.lower()
        return not any(ind in response_lower for ind in failure_indicators)

    def _update_working_memory(self, user_msg: str, assistant_msg: str) -> None:
        """Update working memory with latest interaction."""
        wm = self.memory.working_memory
        # Keep last 5 tool outputs in recent_outputs
        if len(wm.recent_outputs) > 5:
            wm.recent_outputs = wm.recent_outputs[-5:]
        # Add a summary of the latest exchange
        exchange_summary = f"User: {user_msg[:100]}... | Assistant: {assistant_msg[:100]}..."
        wm.recent_outputs.append(exchange_summary)
        self.memory.update_working_memory()

    def _store_episode(self, user_msg: str, assistant_msg: str) -> None:
        """Store an episodic record of the interaction."""
        # Simple action type detection
        action_type = "other"
        msg_lower = user_msg.lower()
        if any(w in msg_lower for w in ["fix", "bug", "error", "debug", "traceback"]):
            action_type = "debug"
        elif any(w in msg_lower for w in ["refactor", "clean", "restructure"]):
            action_type = "refactor"
        elif any(w in msg_lower for w in ["test", "spec", "assert"]):
            action_type = "test"
        elif any(w in msg_lower for w in ["add", "create", "implement", "feature", "new"]):
            action_type = "feature"
        elif any(w in msg_lower for w in ["explain", "what", "how", "why"]):
            action_type = "explain"

        # Extract files mentioned
        import re
        files = re.findall(r'[\w/\\]+\.\w+', user_msg)

        gist = user_msg[:100] + ("..." if len(user_msg) > 100 else "")

        self.memory.store_episode(
            session_id=self.session_id,
            gist=gist,
            facts=[],
            files=files[:10],
            action_type=action_type,
            outcome="unknown",
        )

    def clear_conversation(self) -> None:
        """Clear conversation history (but keep memory)."""
        self.conversation = []
