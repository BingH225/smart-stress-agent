from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.graph import StateGraph, END

from .nodes import (
    physio_sense_node,
    mind_care_node,
    task_relief_propose_node,
    execute_tool_node,
)
from .state import SmartStressState


def route_after_mind_care(state: SmartStressState) -> str:
    """
    Core router after MindCare, mirroring the design doc.
    """
    if state.get("awaiting_human_confirmation"):
        return "wait_for_human_input"

    if state.get("human_confirmation_response") == "yes":
        return "execute_tool"

    if state.get("human_confirmation_response") in ["no", "cancel"]:
        return "monitoring_loop"

    if state.get("current_stressor") and not state.get("suggested_action"):
        return "propose_relief_action"

    return "end"


def build_workflow_graph() -> StateGraph:
    workflow = StateGraph(SmartStressState)

    # Nodes
    workflow.add_node("physio_sense", physio_sense_node)
    workflow.add_node("mind_care", mind_care_node)
    workflow.add_node("task_relief_propose", task_relief_propose_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("wait_for_human_input", lambda state: state)

    # Entry / basic edges
    workflow.set_entry_point("physio_sense")
    workflow.add_edge("physio_sense", "mind_care")
    workflow.add_edge("task_relief_propose", "mind_care")
    workflow.add_edge("execute_tool", "mind_care")

    # Conditional routing from MindCare
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

    return workflow


_APP = None


import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver as SqliteSaver

# ...

def build_app():
    """
    Compile (once) and return a LangGraph app with HITL interrupt configuration.
    """
    global _APP
    if _APP is None:
        workflow = build_workflow_graph()
        
        # Create DB connection (check_same_thread=False is needed for FastAPI)
        conn = sqlite3.connect("smartstress.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        _APP = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["wait_for_human_input"],
        )
    return _APP



