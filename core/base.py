from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from loguru import logger as _logger
from pydantic import BaseModel

logger = _logger

class DataModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

class BaseComponent(ABC):
    """Simple base class with lifecycle hooks and logging."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.bind(component=self.__class__.__name__)
        self.is_initialized = False

    async def initialize(self) -> bool:  # pragma: no cover - base implementation
        self.is_initialized = True
        return True

    @abstractmethod
    async def process(self, data: Any) -> Any:
        pass

    async def cleanup(self):  # pragma: no cover - base implementation
        pass

class AgentResponse(DataModel):
    """Generic response returned by agents."""

    success: bool = True
    message: str = ""
    data: Dict[str, Any] = {}
    triggers: List[str] = []
    meta: Dict[str, Any] = {}
