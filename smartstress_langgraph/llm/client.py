from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import google.generativeai as genai
from google.generativeai import GenerativeModel

from ..config import (
    GEMINI_CHAT_MODEL,
    GEMINI_EMBED_MODEL,
    get_default_generation_config,
    load_google_api_key,
)


_configured = False


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return
    api_key = load_google_api_key()
    genai.configure(api_key=api_key)
    _configured = True


def get_chat_client(
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> GenerativeModel:
    """
    Return a configured GenerativeModel for chat completion.
    """
    _ensure_configured()
    model_name = model or GEMINI_CHAT_MODEL
    kwargs: Dict[str, Any] = {}
    if system_prompt:
        kwargs["system_instruction"] = system_prompt
    return genai.GenerativeModel(model_name, **kwargs)


def generate_chat(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Simple wrapper that takes a list of {role, content} messages and returns text.

    This is intentionally minimal; higher-level tooling (e.g., LangChain) can
    be integrated later if needed.
    """
    client = get_chat_client(system_prompt=system_prompt)
    cfg = get_default_generation_config()
    if generation_config:
        cfg.update(generation_config)

    # google-generativeai expects a list of content blocks; we map roles.
    history = []

    def _map_role(role: str) -> str:
        lowered = role.lower()
        if lowered in {"assistant", "model"}:
            return "model"
        # Treat system/other roles as user instructions for Gemini
        return "user"

    for m in messages:
        history.append(
            {"role": _map_role(m.get("role", "user")), "parts": [{"text": m.get("content", "")}]}
        )

    response = client.generate_content(history, generation_config=cfg)
    return response.text or ""


def embed_documents(texts: Iterable[str]) -> List[List[float]]:
    """
    Compute embeddings for a list of texts using Gemini embeddings.
    """
    _ensure_configured()
    embeddings: List[List[float]] = []
    for t in texts:
        if not t:
            embeddings.append([])
            continue
        res = genai.embed_content(model=GEMINI_EMBED_MODEL, content=t)
        data = _extract_embedding_payload(res)
        embeddings.append(_coerce_embedding(data))
    return embeddings


def _extract_embedding_payload(response: Any) -> Any:
    if isinstance(response, dict) and "embedding" in response:
        return response["embedding"]
    return response


def _coerce_embedding(data: Any) -> List[float]:
    if isinstance(data, dict):
        data = data.get("values") or data.get("value") or []

    if isinstance(data, list):
        return [float(v) for v in data if isinstance(v, (int, float))]

    try:
        return [float(data)]
    except Exception:  # noqa: BLE001
        return []





