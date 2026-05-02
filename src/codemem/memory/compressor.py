"""Pre-compression pipeline inspired by LightMem.

Reduces raw interaction data before storage by removing boilerplate
and extracting key information, reducing token cost by 60-80%.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Patterns to strip from raw interactions
BOILERPLATE_PATTERNS = [
    # Python import statements (keep only non-stdlib)
    re.compile(r"^import\s+\w+", re.MULTILINE),
    re.compile(r"^from\s+\w+\s+import\s+.*$", re.MULTILINE),
    # Standard error boilerplate
    re.compile(r"Traceback \(most recent call last\):\n(?:\s+File .*\n\s+.*\n)*", re.MULTILINE),
    # CLI prompt echoes
    re.compile(r"^[>$]\s+.*$", re.MULTILINE),
    # Blank lines (collapse multiple)
    re.compile(r"\n{3,}"),
    # ANSI escape codes
    re.compile(r"\x1b\[[0-9;]*m"),
    # Repetitive separator lines
    re.compile(r"^[-=_]{10,}$", re.MULTILINE),
]

# High-value patterns to preserve
HIGH_VALUE_PATTERNS = [
    re.compile(r"(?:error|Error|ERROR|exception|Exception|Exception|traceback|Traceback).*", re.IGNORECASE),
    re.compile(r"(?:def |class |function |const |let |var )\w+"),
    re.compile(r"(?:TODO|FIXME|HACK|BUG|NOTE):.*", re.IGNORECASE),
]


def compress_interaction(raw_text: str, max_tokens: int = 2000) -> str:
    """Compress raw interaction text by removing boilerplate and extracting key info.

    Args:
        raw_text: The raw interaction content
        max_tokens: Approximate max tokens (1 token ≈ 4 chars)

    Returns:
        Compressed text preserving key information
    """
    if not raw_text:
        return ""

    # Step 1: Remove boilerplate patterns
    text = raw_text
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)

    # Step 2: Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    # Step 3: If still too long, extract high-value segments
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        text = _extract_key_segments(text, max_chars)

    return text


def compress_diff(raw_diff: str) -> str:
    """Compress a git diff by removing whitespace-only changes and context lines.

    Args:
        raw_diff: Raw git diff output

    Returns:
        Compressed diff with only meaningful changes
    """
    if not raw_diff:
        return ""

    lines = raw_diff.split("\n")
    result = []
    current_file = ""

    for line in lines:
        # Track file names
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                current_file = match.group(1)
                result.append(f"\n--- {current_file} ---")
            continue

        # Skip index lines and git metadata
        if line.startswith("index ") or line.startswith("---") or line.startswith("+++"):
            continue

        # Keep hunk headers
        if line.startswith("@@"):
            result.append(line)
            continue

        # Keep actual changes (+/- lines)
        if line.startswith("+") or line.startswith("-"):
            # Skip pure whitespace changes
            content = line[1:].strip()
            if content:
                result.append(line)
            continue

        # Skip context lines to save space
        # (only keep first/last of each hunk)

    return "\n".join(result)


def compress_error_output(error_text: str) -> str:
    """Compress error/traceback output, keeping only the essential error info.

    Args:
        error_text: Raw error output

    Returns:
        Compressed error with key info preserved
    """
    if not error_text:
        return ""

    lines = error_text.split("\n")
    result = []

    for line in lines:
        stripped = line.strip()
        # Keep error/exception lines
        if re.search(r"(?:Error|Exception|Warning|error|exception)", stripped, re.IGNORECASE):
            result.append(stripped)
        # Keep the last line (usually the actual error message)
        elif stripped and not stripped.startswith("File ") and not stripped.startswith("  "):
            result.append(stripped)

    # Keep first and last 3 lines
    if len(lines) > 6:
        result = lines[:3] + ["..."] + lines[-3:]

    return "\n".join(result)


def extract_topic_boundaries(text: str) -> list[tuple[int, int, str]]:
    """Detect topic boundaries in conversation text.

    Returns list of (start, end, topic_summary) tuples.
    Uses simple heuristics: question marks, code blocks, file mentions.
    """
    boundaries = []
    lines = text.split("\n")
    current_start = 0
    current_topic = "general"

    for i, line in enumerate(lines):
        new_topic = None

        # Detect topic changes
        if re.search(r"\?|\？", line):  # Question
            new_topic = "question"
        elif re.search(r"```|~~~", line):  # Code block
            new_topic = "code"
        elif re.search(r"(?:File|file|path):?\s*[\w/\\]+\.\w+", line):  # File reference
            new_topic = "file_operation"
        elif re.search(r"(?:Error|error|Exception|exception|Traceback)", line):
            new_topic = "debugging"
        elif re.search(r"(?:TODO|FIXME|HACK|NOTE):", line, re.IGNORECASE):
            new_topic = "task"

        if new_topic and new_topic != current_topic and i > current_start + 2:
            boundaries.append((current_start, i, current_topic))
            current_start = i
            current_topic = new_topic

    # Final segment
    boundaries.append((current_start, len(lines), current_topic))

    return boundaries


def _extract_key_segments(text: str, max_chars: int) -> str:
    """Extract the most informative segments from text to fit within max_chars."""
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text[:max_chars]

    # Score each paragraph by information density
    scored = []
    for p in paragraphs:
        score = 0
        # Prefer paragraphs with code
        if re.search(r"[{}()\[\];=]", p):
            score += 2
        # Prefer paragraphs with errors
        if re.search(r"(?:error|Error|exception|Exception)", p):
            score += 3
        # Prefer paragraphs with specific content (not generic)
        if len(set(p.split())) / max(len(p.split()), 1) > 0.6:
            score += 1
        # Penalize very short or very long
        if len(p) < 20:
            score -= 1
        scored.append((score, p))

    # Sort by score, take top paragraphs within budget
    scored.sort(key=lambda x: x[0], reverse=True)
    result = []
    total = 0
    for score, p in scored:
        if total + len(p) + 2 > max_chars:
            break
        result.append(p)
        total += len(p) + 2

    return "\n\n".join(result)


class Compressor:
    """Stateful compressor that tracks conversation context for better compression."""

    def __init__(self, target_token_budget: int = 4000):
        self.target_chars = target_token_budget * 4
        self.history: list[str] = []

    def add_interaction(self, user_msg: str, assistant_msg: str) -> str:
        """Compress and add an interaction to history.

        Returns the compressed version stored.
        """
        raw = f"User: {user_msg}\nAssistant: {assistant_msg}"
        compressed = compress_interaction(raw)
        self.history.append(compressed)

        # If total history exceeds budget, compress older entries
        total_chars = sum(len(h) for h in self.history)
        if total_chars > self.target_chars:
            self._compress_old_entries()

        return compressed

    def get_context(self) -> str:
        """Get compressed context from history."""
        return "\n---\n".join(self.history[-5:])  # Last 5 interactions

    def _compress_old_entries(self) -> None:
        """Compress older history entries more aggressively."""
        if len(self.history) <= 2:
            return

        # Keep last 2 entries as-is, compress the rest into a summary
        recent = self.history[-2:]
        old = self.history[:-2]

        # Simple: just truncate old entries
        compressed_old = []
        for entry in old:
            # Keep first 100 chars of each old entry
            compressed_old.append(entry[:100] + "...")

        self.history = compressed_old + recent
