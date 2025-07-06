from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class StructuredCommand(BaseModel):
    """LLM-issued command with machine-readable action flags."""

    request_id: str = Field(..., description="Unique identifier for the request")
    action_type: str = Field(..., description="Flag describing the system action")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Command payload")
    human_readable_summary: Optional[str] = Field(
        default=None,
        description="Optional summary for logging or UI",
    )
