"""Configuration, read once from the environment and passed to whoever needs it."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    embedding_model: str | None = None  # None: skip the embedding-based aspect merge
    log_level: str = "INFO"  # DEBUG shows tool arguments and results in full
    log_dir: str | None = "logs"  # where the rotating log file goes; None: stdout only

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> Settings:
        """Build settings from environment variables.

        Loads a .env file first: the one at `dotenv_path` if given, otherwise the nearest .env found
        by walking up from the current directory. Variables already set in the environment win.
        """
        load_dotenv(dotenv_path)
        return cls(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ["LLM_MODEL"],
            embedding_model=os.environ.get("EMBEDDING_MODEL") or None,
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            log_dir=os.environ.get("LOG_DIR", "logs") or None,
        )
