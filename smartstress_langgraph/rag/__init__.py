"""RAG components: document schemas, vector store, ingestion, retrieval."""

from .schemas import RagDocument
from .vector_store import VectorStore, get_default_vector_store
from .ingestion import load_documents_from_folder, build_or_update_index
from .retrieval import retrieve_context

__all__ = [
    "RagDocument",
    "VectorStore",
    "get_default_vector_store",
    "load_documents_from_folder",
    "build_or_update_index",
    "retrieve_context",
]


