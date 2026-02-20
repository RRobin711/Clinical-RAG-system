from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env(env_path: Optional[Path] = None) -> None:
    """
    Loads environment variables from a .env file (if present).
    Safe to call multiple times.
    """
    if env_path is None:
        load_dotenv()
    else:
        load_dotenv(dotenv_path=env_path)


@dataclass(frozen=True)
class Settings:
    """
    Centralized config, largely driven by environment variables so the repo remains public-safe.
    """
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-20250514"
    # Anthropic base_url can be set if needed (e.g., proxies)
    anthropic_base_url: Optional[str] = None
    # Increase if you routinely hit timeouts on slow connections
    anthropic_timeout_s: float = 60.0

    @staticmethod
    def from_env() -> "Settings":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is missing. Create a .env file or export ANTHROPIC_API_KEY in your shell."
            )
        
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514").strip()
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        timeout_s = float(os.environ.get("ANTHROPIC_TIMEOUT_S", "60.0"))
        
        return Settings(
            anthropic_api_key=api_key,
            anthropic_model=model,
            anthropic_base_url=base_url,
            anthropic_timeout_s=timeout_s,
        )