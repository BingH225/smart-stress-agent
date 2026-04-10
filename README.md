# SmartStress Agent (LangGraph Edition)

SmartStress is a **multi-agent** stress monitoring and intervention system built on **LangGraph**. It combines **physiological data analysis** (PhysioSense) with **Large Language Models** (LLM) to provide a closed-loop service ranging from stress detection and psychological support to schedule intervention.

## 🌟 Core Features

* **State Machine Architecture**: Powered by a LangGraph directed cyclic graph, supporting a non-linear "Monitor -> Detect -> Intervene -> Monitor" workflow.
* **Multi-Modal Sensing**: Integrates physiological sensor data (HR, HRV) with natural language dialogue for precise stress assessment.
* **Data Persistence**: Built-in **SQLite** backend ensuring cross-session memory, long-term conversation history retention, and state recovery after system restarts.
* **Human-in-the-Loop (Safety)**: Critical operations (like schedule modifications) feature an interrupt mechanism requiring explicit human confirmation to ensure clinical and operational safety.
* **RAG Knowledge Base**: Integrated vector retrieval provides the AI with psychology-based advice grounded in local documents.

## 🧠 Agent Architecture

The system consists of three collaborative Agent nodes:

| Agent Node      | Role                  | Responsibility                                                                                                   |
| :-------------- | :-------------------- | :--------------------------------------------------------------------------------------------------------------- |
| **PhysioSense** | Physiological Sensing | Analyzes sensor data (HR, HRV) to calculate real-time stress probability.                                        |
| **MindCare**    | Psychological Support | Engages in empathetic dialogue, identifies specific stressors, and queries the RAG knowledge base when needed.   |
| **TaskRelief**  | Task Intervention     | Generates concrete schedule adjustment plans based on identified stressors and executes them upon user approval. |

## 🛠️ Installation & Configuration

### 1. Prerequisites

Python 3.10+ is required. Using a virtual environment is recommended.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
FRONTEND_PATH=/path/to/smart-stress-ui/dist
DB_HOST=your_tidb_host
DB_PORT=4000
DB_USERNAME=your_tidb_user
DB_PASSWORD=your_tidb_password
DB_DATABASE=your_tidb_database
```

*(Alternatively, you can use a legacy `.API_KEY` file for the API key)*

## 🚀 Quick Start

### Using the SDK

You can directly import the core SDK in Python to start or continue a session:

```python
from smartstress_langgraph.api import start_monitoring_session
from smartstress_langgraph.io_models import StartSessionRequest, UserInfo, SensorData

# 1. Start a session (Automatically creates SQLite persistence record)
handle, state = start_monitoring_session(
    StartSessionRequest(
        user=UserInfo(user_id="user_1", session_id="session_alpha"),
        initial_sensor_data=SensorData(timestamp="...", values={"hr": 95})
    )
)

# 2. Inspect current state
print(f"Current Stress Probability: {state.current_stress_prob}")
```

### Test Scripts

The project includes several ready-to-use scripts for verifying core functionality:

* `python run_api_key_test.py`: Smoke test for the full "Sense -> Chat -> Plan" flow.
* `python verify_persistence.py`: Verifies state recovery from `smartstress.db` after a restart.
* `python test_api_conn.py`: Tests LLM API connectivity.
* `python smoke_test.py`: Quick RAG retrieval smoke test.

### Starting the Server

The project includes a FastAPI server for REST API access and frontend hosting.

```bash
python server.py
```

* **API Docs**: `http://localhost:8000/docs`
* **Frontend**: `http://localhost:8000` (Requires `FRONTEND_PATH` in `.env`)

## 📚 RAG Knowledge Base Management

MindCare can utilize local documents to enhance its responses.

1.  Place `.txt` or `.md` files in the `rag_docs/` directory.
2.  Run the ingestion script:
    ```bash
    python -m smartstress_langgraph.examples.ingest_docs_example rag_docs
    ```
    This writes embeddings and documents into TiDB tables (`rag_documents`, `rag_embeddings`) for runtime retrieval.

## 📂 Project Structure

```text
smart-stress-agent/
├── smartstress_langgraph/    # Core SDK Package
│   ├── nodes/                # Agent Logic (PhysioSense, MindCare, TaskRelief)
│   ├── rag/                  # Vector Store & Retrieval Logic
│   ├── llm/                  # Gemini Client Wrapper
│   ├── graph.py              # LangGraph Definition & SQLite Persistence
│   ├── state.py              # Global State (TypedDict) Definition
│   └── api.py                # High-level Business API
├── server.py                 # FastAPI Backend Entry Point
├── verify_persistence.py     # Persistence Test Script
├── run_api_key_test.py       # Workflow Smoke Test
└── requirements.txt          # Dependencies
```

-----

*Note: Session data is stored in the local `smartstress.db` file by default. Delete this file if you wish to reset the experimental environment.*
