## SmartStress RAG Guide (TiDB Vector Store)

This guide explains how to ingest local documents into the TiDB-backed vector
store and use retrieval in the MindCare dialogue flow.

### 1. Prepare Documents

- Put Markdown/TXT sources under a folder such as `rag_docs/` at project root.
- Current ingestion supports `.md` and `.txt` files.

### 2. Configure Environment Variables

Create a `.env` file in project root with at least:

```text
GOOGLE_API_KEY=your_google_api_key
DB_HOST=your_tidb_host
DB_PORT=4000
DB_USERNAME=your_tidb_user
DB_PASSWORD=your_tidb_password
DB_DATABASE=your_tidb_database
```

Notes:
- `GOOGLE_API_KEY` is used for chat + embeddings.
- TiDB credentials are read by `smartstress_langgraph/rag/tidb_vector_store.py`.
- Legacy `.API_KEY` is still supported for Google API key only.

### 3. Install Dependencies

From project root:

```bash
pip install -r requirements.txt
```

Key packages for RAG path:
- `mysql-connector-python` (TiDB/MySQL protocol)
- `python-dotenv` (load `.env`)
- `google-generativeai` and `google-genai` (LLM + embedding clients)

### 4. Ingest Documents

CLI example:

```bash
python -m smartstress_langgraph.examples.ingest_docs_example rag_docs
```

Programmatic example:

```python
from smartstress_langgraph import ingest_documents

stats = ingest_documents("rag_docs", tags=["psychoeducation"])
print("Ingested docs:", stats)
```

Ingestion flow:
- `load_documents_from_folder()` reads `.md/.txt`.
- `embed_documents()` generates embeddings.
- `TiDBVectorStore.add_documents()` writes to:
  - `rag_documents`
  - `rag_embeddings`

### 5. Retrieval in Runtime

- `mind_care_node` calls `retrieve_context(query, k=3)` when needed.
- Retrieval returns top-k snippets with source annotations.
- Retrieved snippets are appended to prompt context and stored in state
  field `rag_context`.

### 6. Reset / Rebuild TiDB Index

To fully rebuild:
1. Backup important data.
2. Clear `rag_documents` and `rag_embeddings` tables in TiDB.
3. Re-run ingestion script.

Example SQL:

```sql
DELETE FROM rag_embeddings;
DELETE FROM rag_documents;
```

Then run ingestion again.


