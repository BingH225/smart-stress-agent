from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .state import SessionHandle

class UserInfo(BaseModel):
    user_id: str
    session_id: str
    traits: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional user metadata (e.g., age group, role).",
    )


class SensorData(BaseModel):
    """Batch of physiological sensor values."""

    timestamp: str
    values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw sensor payload (HR, HRV, EDA, etc.).",
    )


class StartSessionRequest(BaseModel):
    user: UserInfo
    initial_sensor_data: Optional[SensorData] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class SessionHandleModel(BaseModel):
    user_id: str
    session_id: str
    thread_id: str
    checkpoint_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_handle(cls, handle: SessionHandle) -> "SessionHandleModel":
        return cls(
            user_id=handle.user_id,
            session_id=handle.session_id,
            thread_id=handle.thread_id,
            checkpoint_id=handle.checkpoint_id,
            metadata=handle.metadata,
        )

    def to_handle(self) -> SessionHandle:
        return SessionHandle(
            user_id=self.user_id,
            session_id=self.session_id,
            thread_id=self.thread_id,
            checkpoint_id=self.checkpoint_id,
            metadata=self.metadata,
        )


class ContinueSessionRequest(BaseModel):
    session_handle: SessionHandleModel
    user_message: Optional[ChatMessage] = None
    sensor_data: Optional[SensorData] = None


class RagDocumentMeta(BaseModel):
    id: Optional[str] = None
    source: Optional[str] = None
    section: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class RagIngestionRequest(BaseModel):
    folder_path: str
    tags: List[str] = Field(default_factory=list)


class SmartStressStateView(BaseModel):
    """
    Serializable view of SmartStressState for frontend / API responses.
    """

    user_id: str
    session_id: str
    current_stress_prob: Optional[float] = None
    stress_history: List[float] = Field(default_factory=list)
    stress_timestamps: List[str] = Field(default_factory=list)
    current_stressor: Optional[str] = None
    suggested_action: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    awaiting_human_confirmation: bool = False
    human_confirmation_response: Optional[str] = None
    rag_context: List[str] = Field(default_factory=list)
    error_log: List[str] = Field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)



