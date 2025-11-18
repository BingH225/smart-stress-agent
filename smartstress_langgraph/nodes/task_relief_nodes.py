from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage

from ..llm.client import generate_chat
from ..llm.prompts import TASK_RELIEF_SYSTEM_PROMPT
from ..state import SmartStressState, ToolCall, append_audit_event, append_error


def task_relief_propose_node(state: SmartStressState) -> Dict[str, Any]:
    """Plan a low-risk action to help with the current stressor."""
    stressor = state.get("current_stressor")
    if not stressor:
        return {}

    preferences = state.get("user_preferences", {})
    preference_clause = ""
    if preferences:
        pref_text = ", ".join(f"{k}={v}" for k, v in preferences.items())
        preference_clause = f"User preferences: {pref_text}.\n"

    prompt = (
        f"The user's primary stressor is: {stressor}.\n"
        f"{preference_clause}"
        "Propose one concrete, low-risk task or schedule adjustment (for example, "
        "rescheduling a meeting, inserting a short break, or splitting a task.\n"
        "Answer in a single English sentence that includes the action, the time "
        "window, and any tools or stakeholders involved."
    )

    messages = [{"role": "user", "content": prompt}]
    try:
        plan_text = generate_chat(
            messages=messages,
            system_prompt=TASK_RELIEF_SYSTEM_PROMPT,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        append_error(state, f"TaskRelief planning failure: {exc}")
        return {}

    if not plan_text:
        append_error(state, "TaskRelief returned empty plan.")
        return {}

    proposed_action: ToolCall = {
        "tool_name": "mock_update_calendar_event",
        "tool_input": {"plan": plan_text, "stressor": stressor},
    }

    append_audit_event(
        state,
        node_name="task_relief_propose",
        summary="Proposed relief action",
        details={"plan": plan_text},
    )
    return {"suggested_action": proposed_action}


def execute_tool_node(state: SmartStressState) -> Dict[str, Any]:
    """Execute the proposed action after human confirmation."""
    action = state.get("suggested_action")
    if not action:
        return {}

    response = state.get("human_confirmation_response")
    if response != "yes":
        # No execution; reset relevant fields.
        append_audit_event(
            state,
            node_name="execute_tool",
            summary="Execution skipped (no consent)",
            details={"response": response},
        )
        return {
            "suggested_action": None,
            "human_confirmation_response": None,
        }

    tool_name = action.get("tool_name", "unknown_tool")
    tool_input = action.get("tool_input", {})

    # Mock tool execution: we only log the action instead of touching real APIs.
    result_text = (
        f"[MOCK] Would execute tool '{tool_name}' with input: {tool_input}.\n"
        "In this demo environment we do not modify any real calendars or systems."
    )

    history = list(state.get("conversation_history", []))
    history.append(AIMessage(content=result_text))

    append_audit_event(
        state,
        node_name="execute_tool",
        summary="Executed mock tool",
        details={"tool_name": tool_name},
    )
    return {
        "tool_output": result_text,
        "suggested_action": None,
        "current_stressor": None,
        "human_confirmation_response": None,
        "awaiting_human_confirmation": False,
        "conversation_history": history,
    }



