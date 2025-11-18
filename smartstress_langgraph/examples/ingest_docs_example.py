from __future__ import annotations

"""
Example for ingesting documents into the local RAG store.

Usage (from project root):
    python -m Agents_LangGraph.smartstress_langgraph.examples.ingest_docs_example \
        path/to/docs
"""

import sys

from ..api import ingest_documents


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: ingest_docs_example.py <folder_path>")
        raise SystemExit(1)
    folder = sys.argv[1]
    stats = ingest_documents(folder)
    print("Ingestion stats:", stats)


if __name__ == "__main__":
    main()



