from __future__ import annotations

from typing import List

from .tidb_vector_store import get_tidb_vector_store


def retrieve_context(query: str, k: int = 5) -> List[str]:
    """
    Retrieve top-k text snippets relevant to the query.
    """
    store = get_tidb_vector_store()
    results = store.similarity_search(query=query, k=k)
    store.close()  # Close connection after use
    return [f"{doc.content}\n\n[source: {doc.source or 'unknown'}]" for doc, _ in results]




