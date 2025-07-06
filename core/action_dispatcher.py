from __future__ import annotations
from typing import Callable, Dict, Any
import inspect

class ActionDispatcher:
    """Simple router for handling structured LLM commands."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register(self, action_type: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        self._handlers[action_type] = handler

    async def dispatch(self, command: Dict[str, Any]) -> None:
        action = command.get("action_type")
        payload = command.get("payload", {})
        handler = self._handlers.get(action)
        if handler:
            if inspect.iscoroutinefunction(handler):
                await handler(payload)
            else:
                handler(payload)
