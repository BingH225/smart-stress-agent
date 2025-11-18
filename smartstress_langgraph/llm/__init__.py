"""LLM client and prompt templates for SmartStress (Gemini-based)."""

from .client import get_chat_client, embed_documents
from . import prompts

__all__ = ["get_chat_client", "embed_documents", "prompts"]


