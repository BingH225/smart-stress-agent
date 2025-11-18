from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, List

from .schemas import RagDocument
from .vector_store import VectorStore, get_default_vector_store


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents_from_folder(folder_path: str) -> List[RagDocument]:
    """
    Load Markdown/TXT files from a folder as RagDocuments.

    PDF and other formats can be added later; for now, we focus on text-like
    sources to keep the implementation dependency-light.
    """
    root = Path(folder_path).expanduser()
    docs: List[RagDocument] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        content = _read_text_file(path)
        if not content.strip():
            continue
        docs.append(
            RagDocument(
                id=str(uuid.uuid4()),
                content=content,
                source=str(path),
                section=None,
                tags=[],
            )
        )
    return docs


def build_or_update_index(
    docs: Iterable[RagDocument],
    store: VectorStore | None = None,
) -> int:
    """
    Add documents to the vector store and return the number of ingested docs.
    """
    materialized = list(docs)
    vs = store or get_default_vector_store()
    if materialized:
        vs.add_documents(materialized)
    return len(materialized)



