from __future__ import annotations

from typing import List

from .vector_store import get_default_vector_store


def retrieve_context(query: str, k: int = 5) -> List[str]:
    """
    Retrieve top-k text snippets relevant to the query.
    """
    store = get_default_vector_store()
    results = store.similarity_search(query=query, k=k)
    return [f"{doc.content}\n\n[source: {doc.source or 'unknown'}]" for doc, _ in results]



