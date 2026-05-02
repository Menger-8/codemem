"""Configuration management for CodeMem."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """API connection configuration."""

    api_key: str = ""  # Anthropic API key
    base_url: str = ""  # Custom base URL (for proxies/compatible APIs)
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 50
    temperature: float = 0.0
    max_tokens: int = 4096
    stream: bool = True

    def get_effective_api_key(self) -> str:
        """Get API key from config, falling back to environment variable."""
        return self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def get_effective_base_url(self) -> str:
        """Get base URL from config, falling back to environment variable."""
        return self.base_url or os.environ.get("ANTHROPIC_BASE_URL", "")


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    working_memory_budget: int = 4000  # tokens
    semantic_memory_top_k: int = 5
    episodic_memory_top_k: int = 3
    procedural_memory_top_k: int = 2
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    embedding_dim: int = 384
    auto_extract_facts: bool = True
    auto_evolve: bool = True
    compress_threshold: int = 8000  # Trigger compression at this token count


class Config(BaseModel):
    """Root configuration."""

    api: APIConfig = Field(default_factory=APIConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    data_dir: str = ""  # Set to project-local .codemem/ by default
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Config:
        """Load config from file, falling back to defaults."""
        if path and path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self, path: Path) -> None:
        """Save config to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    def set_value(self, key: str, value: str) -> bool:
        """Set a config value by dotted key path (e.g. 'api.api_key').

        Returns True if successful, False if key is invalid.
        """
        parts = key.split(".")
        if len(parts) == 2:
            section, field = parts
            section_obj = getattr(self, section, None)
            if section_obj and hasattr(section_obj, field):
                # Type coerce
                field_type = section_obj.__annotations__.get(field, str)
                if field_type is bool:
                    value = value.lower() in ("true", "1", "yes")
                elif field_type is int:
                    value = int(value)
                elif field_type is float:
                    value = float(value)
                setattr(section_obj, field, value)
                return True
        elif len(parts) == 1:
            if hasattr(self, parts[0]):
                setattr(self, parts[0], value)
                return True
        return False

    def get_value(self, key: str) -> Optional[str]:
        """Get a config value by dotted key path."""
        parts = key.split(".")
        if len(parts) == 2:
            section, field = parts
            section_obj = getattr(self, section, None)
            if section_obj and hasattr(section_obj, field):
                val = getattr(section_obj, field)
                # Mask sensitive values
                if field in ("api_key",):
                    if val:
                        return val[:4] + "****" + val[-4:] if len(val) > 8 else "****"
                    return "(not set)"
                return str(val)
        elif len(parts) == 1:
            val = getattr(self, parts[0], None)
            if val is not None:
                return str(val)
        return None
