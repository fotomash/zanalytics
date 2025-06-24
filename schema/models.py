from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import List, Optional
from pydantic import Extra

class CommonConfig:
    """Shared Pydantic model config for all schemas."""
    orm_mode = True
    validate_assignment = True
    extra = Extra.ignore

class Intent(BaseModel):
    user_id: str
    business_type: str
    agent: Optional[str] = None
    action: str
    payload: dict = Field(default_factory=dict)
    schema_version: str = Field(default="1.0", description="Intent schema version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config(CommonConfig):
        """Model-specific config inherited from CommonConfig."""
        title = "Intent"
        schema_extra = {
            "example": {
                "user_id": "user123",
                "business_type": "retail",
                "agent": "agent007",
                "action": "purchase",
                "payload": {"item": "apple", "quantity": 3},
                "schema_version": "1.0",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class AgentResult(BaseModel):
    message: str
    updates: Optional[dict] = None
    tags: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    status: Optional[str] = Field(default="success", description="Result status")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional result metadata")
    result_schema_version: str = Field(default="1.0", description="AgentResult schema version")

    class Config(CommonConfig):
        """Model-specific config inherited from CommonConfig."""
        title = "AgentResult"
        schema_extra = {
            "example": {
                "message": "Operation completed successfully.",
                "updates": {"field": "value"},
                "tags": ["tag1", "tag2"],
                "triggers": ["trigger1"],
                "status": "success",
                "metadata": {"info": "additional info"},
                "result_schema_version": "1.0"
            }
        }


# -----------------------------------------------------------------------------
# AGENT-SPECIFIC SCHEMAS
# -----------------------------------------------------------------------------
# To keep the core framework generic, any agent- or domain-specific
# schemas should be defined in each agent's own directory:
#
#   modules/<agent_name>/models.py
#
# For example, to extend the Intent or AgentResult:
#
#   from core.models import Intent, AgentResult
#   from pydantic import BaseModel
#
#   class PurchaseIntent(Intent):
#       order_id: str
#       items: List[Item]
#
#   class PurchaseResult(AgentResult):
#       confirmation_number: str
#
# This pattern ensures plug-and-play flexibility and future-proofing.
