from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


# Default model names (can be overridden via env vars)
GEMINI_CHAT_MODEL: str = os.getenv(
    "SMARTSTRESS_GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite"
)
GEMINI_EMBED_MODEL: str = os.getenv(
    "SMARTSTRESS_GEMINI_EMBED_MODEL", "gemini-embedding-001"
)

# Path to the local API key file (relative to project root or this package)
API_KEY_FILENAME = ".API_KEY"


def _find_api_key_file() -> Optional[Path]:
    """
    Try to locate the .API_KEY file.

    Search order:
    1. SMARTSTRESS_API_KEY_FILE env var
    2. Agents_LangGraph directory (one level above this package)
    3. Current working directory
    """
    env_path = os.getenv("SMARTSTRESS_API_KEY_FILE")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p

    # Package root: smartstress_langgraph/.. -> Agents_LangGraph
    pkg_root = Path(__file__).resolve().parent.parent
    candidate = pkg_root / API_KEY_FILENAME
    if candidate.is_file():
        return candidate

    # Fallback to CWD
    cwd_candidate = Path.cwd() / API_KEY_FILENAME
    if cwd_candidate.is_file():
        return cwd_candidate

    return None


@lru_cache(maxsize=1)
def load_google_api_key() -> str:
    """
    Load GOOGLE_API_KEY from .API_KEY file or environment.

    .API_KEY format:
        GOOGLE_API_KEY=xxxx...
    or just:
        xxxx...

    Environment override:
        If GOOGLE_API_KEY env var is set, it is used directly.
    """
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    api_key_path = _find_api_key_file()
    if not api_key_path:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Please set the GOOGLE_API_KEY "
            "environment variable or create a .API_KEY file in Agents_LangGraph "
            "containing the key."
        )

    content = api_key_path.read_text(encoding="utf-8").strip()
    if not content:
        raise RuntimeError(f".API_KEY file at {api_key_path} is empty.")

    # Support both raw key and KEY=... format
    if "=" in content:
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "GOOGLE_API_KEY":
                    v = v.strip()
                    if not v:
                        raise RuntimeError(
                            f".API_KEY at {api_key_path} contains an empty GOOGLE_API_KEY entry."
                        )
                    return v
        raise RuntimeError(
            f".API_KEY at {api_key_path} does not contain a valid GOOGLE_API_KEY entry."
        )

    return content


def get_default_generation_config() -> dict:
    """Default hyper-parameters for Gemini chat models."""
    return {
        "temperature": float(os.getenv("SMARTSTRESS_TEMPERATURE", "0.3")),
        "top_p": float(os.getenv("SMARTSTRESS_TOP_P", "0.9")),
        "top_k": int(os.getenv("SMARTSTRESS_TOP_K", "40")),
        "max_output_tokens": int(os.getenv("SMARTSTRESS_MAX_TOKENS", "1024")),
    }



