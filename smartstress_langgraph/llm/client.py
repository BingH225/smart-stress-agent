from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.api_core import client_options
from google import genai as new_genai  # New SDK for proxy support
from google.genai import types

from ..config import (
    GEMINI_CHAT_MODEL,
    GEMINI_EMBED_MODEL,
    get_default_generation_config,
    load_google_api_key,
)


_configured = False
_new_client = None


def _ensure_configured() -> None:
    global _configured, _new_client
    if _configured:
        return
    api_key = load_google_api_key()

    # --- NEW SDK PROXY PROTOCOL (Active) ---
    _new_client = new_genai.Client(
        api_key=api_key,
        vertexai=False,  # Try standard protocol
        http_options={
            "base_url": "https://api.openai-proxy.org/google"
        },
    )
    
    # --- LEGACY PROTOCOL (Preserved in comments) ---
    # opts = client_options.ClientOptions(api_endpoint="https://api.openai-proxy.org/google")
    # genai.configure(api_key=api_key, transport='rest', client_options=opts)
    # genai.configure(api_key=api_key)
    
    _configured = True


def get_chat_client(
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> GenerativeModel:
    """
    DEPRECATED: Returns a legacy SDK model. Use generate_chat directly.
    """
    _ensure_configured()
    # This remains for backward compatibility but might not work with proxy
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
    Uses the new SDK with proxy support.
    """
    _ensure_configured()
    model_name = GEMINI_CHAT_MODEL
    
    # Map messages to new SDK format
    contents = []
    for m in messages:
        role = m.get("role", "user")
        if role.lower() == "assistant":
            role = "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m.get("content", ""))]))

    # Merge default config with overrides
    cfg_defaults = get_default_generation_config()
    if generation_config:
        cfg_defaults.update(generation_config)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=cfg_defaults.get("temperature"),
        top_p=cfg_defaults.get("top_p"),
        top_k=cfg_defaults.get("top_k"),
        max_output_tokens=cfg_defaults.get("max_output_tokens"),
    )

    # Legacy code using old SDK (commented out)
    # client = get_chat_client(system_prompt=system_prompt)
    # response = client.generate_content(history, generation_config=cfg)
    # return response.text or ""

    response = _new_client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config
    )
    return response.text or ""


def embed_documents(texts: Iterable[str]) -> List[List[float]]:
    """
    Compute embeddings for a list of texts using Gemini embeddings.
    """
    _ensure_configured()
    embeddings: List[List[float]] = []
    
    # The new SDK supports batch embedding
    text_list = list(texts)
    if not text_list:
        return []

    try:
        res = _new_client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text_list
        )
        if res.embeddings:
            for emb in res.embeddings:
                embeddings.append([float(v) for v in emb.values])
        else:
            embeddings.extend([[] for _ in text_list])
    except Exception:
        # Fallback to empty embeddings if error
        embeddings.extend([[] for _ in text_list])

    # Legacy code using old SDK (commented out)
    # for t in texts:
    #     res = genai.embed_content(model=GEMINI_EMBED_MODEL, content=t)
    #     data = _extract_embedding_payload(res)
    #     embeddings.append(_coerce_embedding(data))
    
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






