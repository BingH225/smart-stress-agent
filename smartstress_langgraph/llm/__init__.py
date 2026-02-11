"""LLM client and prompt templates for SmartStress (Gemini-based)."""

from .client import get_chat_client, embed_documents, generate_chat
from . import prompts

__all__ = ["get_chat_client", "embed_documents", "generate_chat", "prompts"]



