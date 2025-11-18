from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import chromadb
from chromadb import Collection

from ..llm import embed_documents
from .schemas import RagDocument


class VectorStore:
    """
    Thin wrapper around a local Chroma collection.

    This is intentionally minimal and filesystem-based for easy experimentation.
    """

    def __init__(self, persist_dir: Path, collection_name: str = "smartstress_rag"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, docs: Sequence[RagDocument]) -> None:
        texts = [d.content for d in docs]
        embeddings = embed_documents(texts)
        ids = [d.id or f"doc-{i}" for i, d in enumerate(docs)]
        metadatas = [
            {
                "source": d.source,
                "section": d.section,
                "created_at": d.created_at,
                "tags": d.tags,
            }
            for d in docs
        ]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def similarity_search(
        self, query: str, k: int = 5
    ) -> List[Tuple[RagDocument, float]]:
        res = self.collection.query(query_texts=[query], n_results=k)
        results: List[Tuple[RagDocument, float]] = []
        for i in range(len(res.get("ids", [[]])[0])):
            doc = RagDocument(
                id=res["ids"][0][i],
                content=res["documents"][0][i],
                source=res["metadatas"][0][i].get("source"),
                section=res["metadatas"][0][i].get("section"),
                created_at=res["metadatas"][0][i].get("created_at"),
                tags=res["metadatas"][0][i].get("tags", []),
            )
            score = res["distances"][0][i] if "distances" in res else 0.0
            results.append((doc, score))
        return results


def get_default_vector_store() -> VectorStore:
    """
    Return a default VectorStore under Agents_LangGraph/.rag_store.
    """
    root = Path(__file__).resolve().parents[2]  # -> Agents_LangGraph
    persist_dir = root / ".rag_store"
    persist_dir.mkdir(parents=True, exist_ok=True)
    return VectorStore(persist_dir=persist_dir)



