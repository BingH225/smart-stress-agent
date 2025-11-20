## SmartStress LangGraph Backend (SDK)

This folder contains a LangGraph-based multi-agent backend for SmartStress,
implemented as a pure Python SDK that can be imported by the existing FastAPI
service and React frontend.

- Agents: PhysioSense (detection), MindCare (dialogue + HITL), TaskRelief
  (planning + execution via mock tools, ready for real APIs).
- LLM: Google Gemini (`gemini-2.5-flash-lite` for chat,
  `gemini-embedding-001` for RAG embeddings) accessed through `GOOGLE_API_KEY`.
- Safety: explicit HITL interrupt, audit trail, error logging, and RAG evidence
  snapshots stored in the shared state.

See `RAG_guide.md` for vector-store ingestion instructions.

### 1. Setup

1. Install dependencies (from repo root):

   ```bash
   pip install -r Agents_LangGraph/requirements.txt
   ```

2. Provide credentials:

   - Option A: set environment variable `GOOGLE_API_KEY`.
   - Option B: add it to a `.env` file in this directory:

     ```text
     GOOGLE_API_KEY=your_real_key_here
     ```

     (A legacy `.API_KEY` file is still recognised if you prefer that layout.)

### 2. Quick Start (Python)

```python
from Agents_LangGraph.smartstress_langgraph import (
    start_monitoring_session,
    continue_session,
)
from Agents_LangGraph.smartstress_langgraph.io_models import (
    StartSessionRequest, ContinueSessionRequest,
    UserInfo, SensorData, ChatMessage,
)

start_req = StartSessionRequest(
    user=UserInfo(user_id="p001", session_id="sess-001"),
    initial_sensor_data=SensorData(timestamp="2024-01-01T10:00:00Z", values={"hr": 95}),
)
handle, state = start_monitoring_session(start_req)

follow_up = ContinueSessionRequest(
    session_handle=handle,
    user_message=ChatMessage(
        role="user",
        content="The 3pm cross-team review is stressing me out and I still need slides.",
    ),
)
handle, updated_state = continue_session(follow_up)
```

`state` / `updated_state` are `SmartStressStateView` objects that can be sent
to the existing backend/React application.

### 3. Feature Highlights

- `smartstress_langgraph/state.py`: Typed state with audit trail/logging helpers.
- `smartstress_langgraph/nodes/`: PhysioSense, MindCare, TaskRelief logic.
- `smartstress_langgraph/rag/`: Local Chroma vector store with ingest/retrieve APIs.
- `smartstress_langgraph/examples/`: CLI demos for session flow and RAG ingestion.



