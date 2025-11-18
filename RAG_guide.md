## SmartStress RAG Guide (Local Vector Store)

This guide explains how to ingest existing documents into the local vector
database and surface them inside the MindCare dialogue flow.

### 1. Prepare Documents

- Put all Markdown/TXT sources you want to index under a folder such as
  `Agents_LangGraph/rag_docs/`.
- The reference implementation currently supports `.md` and `.txt`. Extend the
  ingestion script if you need PDF or other formats.

### 2. Configure `GOOGLE_API_KEY`

1. Store your Google API key in `Agents_LangGraph/.API_KEY`, for example:

   ```text
   GOOGLE_API_KEY=your_real_key_here
   ```

2. Alternatively set an environment variable:

   ```bash
   set GOOGLE_API_KEY=your_real_key_here
   ```

   The runtime prefers the environment variable and falls back to the
   `.API_KEY` file.

### 3. Install Dependencies

From the project root (the directory containing `Agents_LangGraph`) run:

```bash
pip install -r Agents_LangGraph/requirements.txt
```

Key packages:

- `google-generativeai` (Gemini client)
- `langgraph`
- `langchain-core`
- `pydantic`
- `chromadb`

### 4. Ingest Documents

From the project root:

```bash
python -m Agents_LangGraph.smartstress_langgraph.examples.ingest_docs_example Agents_LangGraph/rag_docs
```

The script will:

- Use `load_documents_from_folder()` to read `.md`/`.txt` files.
- Create embeddings with `gemini-embedding-001`.
- Store vectors in the local Chroma database at `Agents_LangGraph/.rag_store`.

You can also ingest programmatically:

```python
from Agents_LangGraph.smartstress_langgraph import ingest_documents

stats = ingest_documents("Agents_LangGraph/rag_docs", tags=["psychoeducation"])
print("Ingested docs:", stats)
```

### 5. Use RAG During Dialogue

- The MindCare node calls `retrieve_context()` when it needs psychoeducation or
  scheduling tips. The top-k snippets are appended to the LLM system prompt and
  saved under `rag_context` so the frontend can display evidence chips with the
  original sources.

### 6. Reset the Index

- The vector store persists under `Agents_LangGraph/.rag_store`.
- Delete that folder if you need to rebuild the index from scratch:

```bash
rm -rf Agents_LangGraph/.rag_store
```

Then re-run the ingestion script.


