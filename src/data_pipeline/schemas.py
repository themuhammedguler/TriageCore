"""Data schemas for medical triage training data.

Defines Pydantic models for chat message structures, raw records,
and validated ChatML/Llama-3 training samples.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Permitted message roles in instruction-tuning formats."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Single turn in a conversational triage interaction."""
    role: MessageRole
    content: str = Field(..., min_length=1, description="Message text content")

    @field_validator("content")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message content cannot be empty or whitespace only.")
        return cleaned


class TriageSample(BaseModel):
    """Complete multi-turn or single-turn dialogue ready for training."""
    messages: List[ChatMessage] = Field(..., min_length=2, description="At least user and assistant turns required")

    @field_validator("messages")
    @classmethod
    def validate_conversation_flow(cls, messages: List[ChatMessage]) -> List[ChatMessage]:
        roles = [m.role for m in messages]
        # Must contain at least one user and one assistant message
        if MessageRole.USER not in roles:
            raise ValueError("Conversation must contain at least one 'user' message.")
        if MessageRole.ASSISTANT not in roles:
            raise ValueError("Conversation must contain at least one 'assistant' message.")
        # Assistant must not be the first message (unless system precedes it)
        if roles[0] == MessageRole.ASSISTANT:
            raise ValueError("Conversation cannot start with an 'assistant' message.")
        return messages


class RawMedicalRecord(BaseModel):
    """Raw record extracted from source datasets before cleaning and formatting."""
    patient_query: str = Field(..., min_length=1)
    doctor_response: str = Field(..., min_length=1)
    source: str = Field(default="unknown")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
