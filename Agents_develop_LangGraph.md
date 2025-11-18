# SmartStress Multi-Agent Architecture (LangGraph Edition)

## 1. Introduction

### 1.1 Purpose

This document is the technical blueprint for the SmartStress backend. It
describes how to implement the production system with **LangGraph**, replacing
the early Dify prototype [5] with a stateful, loop-friendly, and clinically safe
multi-agent architecture.

### 1.2 Core Idea: Agents as a State Machine

Unlike role-based frameworks such as ADK or CrewAI, LangGraph treats the system
as a **directed graph** (i.e., an explicit **state machine**). The three
SmartStress agents-`PhysioSense`, `MindCare`, and `TaskRelief` [5]-are nodes in
this graph rather than separate chatbots. They interact through a shared,
persistent state object, and control flow is governed by conditional edges.

Benefits:

1. **Predictability.** Every transition is defined by the graph, which is vital
   for regulated healthcare settings [4].
2. **Safety (HITL).** LangGraph's native interrupt mechanism lets us pause the
   workflow before TaskRelief performs any write operation (e.g., editing a
   calendar) until a human explicitly approves [2, 3].
3. **Native loops.** The "monitor -> detect -> intervene -> monitor" lifecycle is a
   loop, and LangGraph supports cycles out of the box [2, 4], unlike many linear
   workflow tools.

## 2. Core Architecture: `SmartStressState`

State is a first-class concept in LangGraph [4]. We define a Python `TypedDict`
that stores all runtime information. Every node reads and updates this shared
state.

```python
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class ToolCall(TypedDict):
    """Structure describing a tool invocation proposed by an agent."""

    tool_name: str
    tool_input: Dict[str, Any]


class SmartStressState(TypedDict, total=False):
    """Central shared state that travels through the LangGraph."""

    # === User & session ===
    user_id: str
    session_id: str

    # === PhysioSense (L1) ===
    raw_sensor_input: Optional[Dict[str, Any]]
    current_stress_prob: float
    stress_history: List[float]

    # === MindCare (L2) ===
    conversation_history: List[BaseMessage]
    current_stressor: Optional[str]

    # === TaskRelief (L3) ===
    suggested_action: Optional[ToolCall]
    tool_output: Optional[str]

    # === Safety / HITL ===
    awaiting_human_confirmation: bool
    human_confirmation_response: Optional[str]
```

## 3. Nodes: Implementing Agent Logic

Each LangGraph node is a callable that receives `SmartStressState` and returns a
dict containing the fields it wants to update.

### 3.1 `physio_sense_node` (L1: Detection)

- **Role:** PhysioSense [5]
- **Responsibility:** Run the CNN-LSTM model to estimate physiological stress.
- **Trigger:** New sensor payloads or periodic sampling.

```python
def physio_sense_node(state: SmartStressState) -> dict:
    """Run the stress model and update probabilities."""
    raw_data = state.get("raw_sensor_input")

    # Placeholder: call the real CNN-LSTM from the research codebase.
    stress_prob = 0.92

    history = state.get("stress_history", []) + [stress_prob]
    return {
        "current_stress_prob": stress_prob,
        "stress_history": history,
        "raw_sensor_input": None,
    }
```

### 3.2 `mind_care_node` (L2: Dialogue + Presentation)

- **Role:** MindCare [5]
- **Responsibilities:** (a) converse with the user to identify stressors; (b)
  present TaskRelief's proposals to collect human consent.
- **Note:** Logic branches based on the current state.

```python
def mind_care_node(state: SmartStressState) -> dict:
    """Manage conversations, RAG context, and HITL confirmation."""

    if state.get("awaiting_human_confirmation"):
        user_msg = state["conversation_history"][-1]
        return {
            "awaiting_human_confirmation": False,
            "human_confirmation_response": user_msg.content.lower(),
        }

    if state.get("suggested_action"):
        action = state["suggested_action"]
        prompt = (
            f"I can reduce stress by executing {action['tool_name']}. "
            "May I proceed? Please answer 'yes' or 'no'."
        )
        history = state["conversation_history"] + [prompt]
        return {
            "conversation_history": history,
            "awaiting_human_confirmation": True,
        }

    if state.get("current_stress_prob", 0.0) > 0.9 and not state.get("current_stressor"):
        prompt = "I noticed your stress level is elevated. What happened recently?"
        history = state.get("conversation_history", []) + [prompt]
        return {
            "conversation_history": history,
            "current_stressor": "3pm stakeholder review",
        }

    return {}
```

### 3.3 `task_relief_propose_node` (L3: Planning)

- **Role:** Planning half of TaskRelief [5]
- **Responsibility:** Read `current_stressor`, call read-only tools (e.g.,
  Google Calendar API [6-9]), and produce a low-risk plan.

```python
def task_relief_propose_node(state: SmartStressState) -> dict:
    """Draft an actionable plan for the current stressor."""
    stressor = state["current_stressor"]

    proposed_action = ToolCall(
        tool_name="google_calendar_update_event",
        tool_input={"event_id": "xyz123", "new_start": "tomorrow_10am"},
    )
    return {"suggested_action": proposed_action}
```

### 3.4 `execute_tool_node` (L3: Execution)

- **Role:** Execution half of TaskRelief
- **Responsibility:** Safely run the tool **after** human approval.

```python
def execute_tool_node(state: SmartStressState) -> dict:
    """Perform the proposed action once consent is granted."""
    action = state["suggested_action"]
    result = "Meeting successfully rescheduled."  # replace with the real API call

    return {
        "tool_output": result,
        "suggested_action": None,
        "current_stressor": None,
        "human_confirmation_response": None,
    }
```

## 4. Edges & Routing Logic

`add_conditional_edges` encodes the "brain" of the graph.

```python
def route_after_mind_care(state: SmartStressState) -> str:
    """Decide which node to run after MindCare completes."""
    if state.get("awaiting_human_confirmation"):
        return "wait_for_human_input"
    if state.get("human_confirmation_response") == "yes":
        return "execute_tool"
    if state.get("human_confirmation_response") in ["no", "cancel"]:
        return "monitoring_loop"
    if state.get("current_stressor") and not state.get("suggested_action"):
        return "propose_relief_action"
    return "monitoring_loop"
```

## 5. Assembling the Graph

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(SmartStressState)

workflow.add_node("physio_sense", physio_sense_node)
workflow.add_node("mind_care", mind_care_node)
workflow.add_node("task_relief_propose", task_relief_propose_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("wait_for_human_input", lambda state: state)

workflow.set_entry_point("physio_sense")

workflow.add_edge("physio_sense", "mind_care")
workflow.add_edge("task_relief_propose", "mind_care")
workflow.add_edge("execute_tool", "mind_care")

workflow.add_conditional_edges(
    "mind_care",
    route_after_mind_care,
    {
        "wait_for_human_input": "wait_for_human_input",
        "execute_tool": "execute_tool",
        "propose_relief_action": "task_relief_propose",
        "monitoring_loop": "physio_sense",
        "end": END,
    },
)

app = workflow.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["wait_for_human_input"],
)
```

## 6. Conclusion

LangGraph turns SmartStress into an **auditable, predictable, and interruptible**
state machine:

- **Safety.** `execute_tool_node` (write operations) is guarded by explicit
  interrupts plus routing logic that requires a positive "yes."
- **Extensibility.** Adding tools such as Asana or Trello only requires a new
  tool wrapper and a branch inside `task_relief_propose_node`.
- **Durability.** With the checkpointer, we can pause execution for hours or
  days while waiting for human input, then resume from exactly the same state
  [1, 4].

This level of determinism is essential for SaMD compliance and moves
SmartStress well beyond the prototype phase.