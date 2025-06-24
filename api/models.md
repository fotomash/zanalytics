# Data Models

```python
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class IntentType(str, Enum):
    """
    Core intent types. Extend this enum in your application modules for domain-specific actions.
    """
    GENERIC = "generic"
    CUSTOM = "custom"  # For user-defined or module-specific intents

# Example of extending IntentType in a module:
from models import IntentType as BaseIntentType
class ExtendedIntentType(BaseIntentType):
    NEW_ACTION = "new_action"

class Intent(BaseModel):
    """
    Represents a user's intent, including business context and payload.
    """
    user_id: str = Field(..., description="Unique identifier for the user")
    business_type: str = Field(..., description="Domain or product this intent applies to")
    intent_type: IntentType = Field(..., description="Type of intent being expressed")
    payload: Dict[str, Any] = Field(..., description="Structured data for the intent")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the intent was created")
    request_id: Optional[str] = Field(None, description="Unique request identifier for tracing across services")
    api_version: Optional[str] = Field("v1", description="API version used for this intent")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary metadata for custom hooks or routing")

class AgentResult(BaseModel):
    """
    Result returned by a single agent after processing an intent.
    """
    agent_name: str = Field(..., description="Name of the agent that handled the intent")
    output: Dict[str, Any] = Field(..., description="Structured response or logs from the agent")

class Memory(BaseModel):
    """
    Represents the user's persistent state for the current session or day.
    """
    user_id: str = Field(..., description="Unique identifier for the user")
    date: datetime = Field(default_factory=datetime.utcnow, description="Date for this memory snapshot")
    context: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value store of user state")

class ResponseModel(BaseModel):
    """
    Standard API response including message, status, detailed data, agent outputs, and memory.
    """
    message: str = Field(..., description="Human-readable response for the user")
    status: str = Field(..., description="Response status code or keyword")
    data: Optional[Dict[str, Any]] = Field(None, description="Optional payload data for the client")
    agents: Optional[List[AgentResult]] = Field(None, description="List of individual agent outputs")
    memory: Optional[Memory] = Field(None, description="Updated user memory/state after handling the intent")
    api_version: Optional[str] = Field(None, description="Echoed API version")
```

*Note: Remember to update `models.md` whenever new fields or models are introduced to keep the documentation current.*