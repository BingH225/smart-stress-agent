from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, AIMessage

from ..llm.client import generate_chat
from ..llm.prompts import MIND_CARE_SYSTEM_PROMPT
from ..rag.retrieval import retrieve_context
from ..state import SmartStressState, append_audit_event, append_error


def _looks_like_confirmation(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {"yes", "no", "cancel", "y", "n"}


def _extract_stressor_from_text(text: str) -> Optional[str]:
    prompt = (
        "The user described their stress as follows:\n"
        f"{text}\n\n"
        "Summarize the single most likely stressor (event, task, or interaction) "
        "in <= 15 English words. If you cannot infer it, respond with 'unknown stressor'."
    )
    try:
        result = generate_chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=(
                "You are a text classifier. Output only the stressor summary. "
                "Do not add explanations or advice."
            ),
        )
    except Exception:
        return None

    result = (result or "").strip()
    if not result or result.lower() == "unknown stressor":
        return None
    return result


def mind_care_node(state: SmartStressState) -> Dict[str, Any]:
    """
    MindCare node: handles conversation, RAG, and presenting TaskRelief proposals.
    """
    updates: Dict[str, Any] = {}

    # Scenario C: waiting for human confirmation
    if state.get("awaiting_human_confirmation"):
        history = state.get("conversation_history", [])
        if not history:
            append_error(
                state,
                "awaiting_human_confirmation=True but conversation_history is empty.",
            )
            updates["awaiting_human_confirmation"] = False
            return updates

        last_msg = history[-1]
        if isinstance(last_msg, HumanMessage):
            text = last_msg.content.lower()
        else:
            text = str(getattr(last_msg, "content", "")).lower()

        normalized = "cancel"
        lowered = text.strip().lower()
        if any(token in lowered for token in ["yes", "y", "sure", "ok"]):
            normalized = "yes"
        elif any(token in lowered for token in ["no", "n", "nope", "nah"]):
            normalized = "no"

        updates.update(
            {
                "awaiting_human_confirmation": False,
                "human_confirmation_response": normalized,
            }
        )
        append_audit_event(
            state,
            node_name="mind_care",
            summary="Processed human confirmation",
            details={"response": normalized},
        )
        return updates

    # Scenario B: present TaskRelief suggestion
    if state.get("suggested_action"):
        action = state["suggested_action"]
        tool_name = action.get("tool_name", "an action")
        prompt = (
            f"I found a possible way to reduce stress: I can run \"{tool_name}\".\n"
            "This only adjusts your schedule or tasks and is fully reversible.\n"
            "Do you want me to proceed? Please answer yes or no."
        )
        history = list(state.get("conversation_history", []))
        history.append(AIMessage(content=prompt))
        updates.update(
            {
                "conversation_history": history,
                "awaiting_human_confirmation": True,
            }
        )
        append_audit_event(
            state,
            node_name="mind_care",
            summary="Presented TaskRelief suggestion",
            details={"tool_name": tool_name},
        )
        return updates

    # Scenario A-1: new human message that might describe a stressor
    history = state.get("conversation_history", [])
    latest_human: Optional[HumanMessage] = None
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            latest_human = msg
            break

    if (
        latest_human
        and not state.get("current_stressor")
        and not state.get("awaiting_human_confirmation")
        and not state.get("suggested_action")
        and len(latest_human.content.strip()) >= 6
        and not _looks_like_confirmation(latest_human.content)
    ):
        stressor = _extract_stressor_from_text(latest_human.content)
        if stressor:
            updates["current_stressor"] = stressor
            append_audit_event(
                state,
                node_name="mind_care",
                summary="Identified stressor from dialogue",
                details={"stressor": stressor},
            )
            return updates

    # Scenario A: high stress, unknown stressor
    current_stress_prob = float(state.get("current_stress_prob", 0.0))
    if current_stress_prob > 0.9 and not state.get("current_stressor"):
        # Retrieve psychoeducational context (RAG)
        rag_snippets = retrieve_context(
            "short-term stress management and scheduling advice", k=3
        )

        try:
            system_prompt = (
                MIND_CARE_SYSTEM_PROMPT
                + "\n\nSupporting evidence:\n"
                + "\n---\n".join(rag_snippets)
            )
            user_prompt = (
                "Write a short (<=3 sentences) reply that:\n"
                f"- Acknowledges the user's elevated stress probability ({current_stress_prob:.2f}).\n"
                "- Offers one brief tip grounded in the evidence above.\n"
                "- Ends with an open question inviting the user to describe their primary stressor.\n"
            )
            reply = generate_chat(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            append_error(state, f"MindCare LLM failure: {exc}")
            reply = (
                "I can see your stress indicators are higher than usual. "
                "If you feel comfortable, could you share the situation that has felt most stressful lately? "
                "I'll tailor the next steps based on what you share."
            )

        history = list(state.get("conversation_history", []))
        history.append(AIMessage(content=reply))

        # For now, we rely on downstream steps to extract a concrete stressor.
        updates.update(
            {
                "conversation_history": history,
                "rag_context": rag_snippets,
            }
        )
        append_audit_event(
            state,
            node_name="mind_care",
            summary="Initiated stressor exploration",
            details={"stress_prob": current_stress_prob},
        )
        return updates

    # Default: no-op
    append_audit_event(
        state,
        node_name="mind_care",
        summary="No-op (no new dialogue needed)",
    )
    return updates



