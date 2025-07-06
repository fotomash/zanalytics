from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger
from pydantic import BaseModel, Field


class BaseComponent:
    """Lightweight base class providing common component utilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        # Bind the component name for structured log messages
        self.logger = logger.bind(component=self.__class__.__name__)
        self.is_initialized: bool = False

    async def initialize(self) -> bool:
        """Initialize the component."""
        self.is_initialized = True
        return True

    async def process(self, data: Any) -> Any:  # pragma: no cover - abstract
        """Process incoming data. Should be overridden by subclasses."""
        raise NotImplementedError

    async def cleanup(self) -> None:
        """Clean up resources before shutdown."""
        pass


class DataModel(BaseModel):
    """Base Pydantic model for data structures."""
    pass


class AgentResponse(BaseModel):
    """Standard response model returned by agents."""

    success: bool = Field(..., description="Whether the agent completed successfully")
    message: str = Field(..., description="User-facing response text")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured payload returned by the agent")
    triggers: List[str] = Field(default_factory=list, description="Follow-up triggers")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
