"""Action Dispatcher module for routing structured commands."""

import json
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Awaitable

from loguru import logger
import redis


class ActionDispatcher:
    """Central command routing system."""

    def __init__(self, redis_client: redis.Redis, config: Optional[Dict[str, Any]] = None) -> None:
        self.redis = redis_client
        self.config = config or {}
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        self._register_default_handlers()

    def register_handler(self, action_type: str, handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> None:
        self.handlers[action_type] = handler
        logger.debug(f"Registered handler for {action_type}")

    async def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        if not self._validate_command(command):
            return {"status": "error", "message": "Invalid command"}
        action_type = command["action_type"]
        request_id = command["request_id"]
        handler = self.handlers.get(action_type)
        if not handler:
            logger.error(f"Unknown action: {action_type}")
            return {"status": "error", "message": f"Unknown action {action_type}"}
        try:
            if "metadata" not in command:
                command["metadata"] = {}
            command["metadata"]["processed_at"] = datetime.utcnow().isoformat()
            result = await handler(command)
            self._publish_event(
                "command_processed",
                {"request_id": request_id, "action_type": action_type, "status": "success"},
            )
            return result
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.error(f"Failed to process {request_id}: {exc}")
            self._publish_event(
                "command_failed",
                {"request_id": request_id, "action_type": action_type, "error": str(exc)},
            )
            return {"status": "error", "message": str(exc)}

    def _validate_command(self, command: Dict[str, Any]) -> bool:
        required_fields = {"request_id", "action_type", "payload"}
        return required_fields.issubset(command)

    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        event = {"type": event_type, "timestamp": datetime.utcnow().isoformat(), "data": data}
        try:
            self.redis.publish("zanalytics:events", json.dumps(event))
        except Exception as exc:  # pragma: no cover - redis failure
            logger.error(f"Publish failed: {exc}")

    # ------------------------------------------------------------------
    # Default handlers used for tests/demonstration
    # ------------------------------------------------------------------

    def _register_default_handlers(self) -> None:
        self.register_handler("LOG_JOURNAL_ENTRY", self._handle_journal_entry)

    async def _handle_journal_entry(self, command: Dict[str, Any]) -> Dict[str, Any]:
        payload = command["payload"]
        entry = {
            "id": command["request_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "content": payload.get("content", ""),
        }
        self.redis.hset("zanalytics:journal", entry["id"], json.dumps(entry))
        self.redis.publish("zanalytics:journal:new", json.dumps(entry))
        return {"status": "success", "entry_id": entry["id"]}


# Singleton helper -------------------------------------------------------------

_dispatcher_instance: Optional[ActionDispatcher] = None


def get_dispatcher(redis_client: redis.Redis, config: Optional[Dict[str, Any]] = None) -> ActionDispatcher:
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = ActionDispatcher(redis_client, config)
    return _dispatcher_instance
