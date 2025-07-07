"""
Action Dispatcher - The central routing system for all commands in Zanalytics.
Processes structured commands and routes them to appropriate handlers.
"""

import json
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import redis
from loguru import logger

class ActionDispatcher:
    """
    Central command routing system that processes structured commands
    from LLMs, agents, and triggers, routing them to appropriate handlers.
    """

    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register the default action handlers."""
        self.register_handler("LOG_JOURNAL_ENTRY", self._handle_journal_entry)
        self.register_handler("UPDATE_DASHBOARD_STATE", self._handle_dashboard_update)
        self.register_handler("TRIGGER_AGENT_ANALYSIS", self._handle_agent_trigger)
        self.register_handler("NOTIFY_USER", self._handle_notification)
        self.register_handler("EXECUTE_TRADE_IDEA", self._handle_trade_idea)
        self.register_handler("UPDATE_MARKET_CONTEXT", self._handle_market_update)
        self.register_handler("SCHEDULE_FOLLOWUP", self._handle_schedule_followup)
        self.register_handler("ARCHIVE_ANALYSIS", self._handle_archive)

    def register_handler(self, action_type: str, handler: Callable):
        """Register a custom handler for a specific action type."""
        self.handlers[action_type] = handler
        logger.info(f"Registered handler for action type: {action_type}")

    async def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a structured command and route it to the appropriate handler.

        Args:
            command: Structured command following the master schema

        Returns:
            Response from the handler
        """
        # Validate command structure
        if not self._validate_command(command):
            return {"status": "error", "message": "Invalid command structure"}

        action_type = command.get("action_type")
        request_id = command.get("request_id")

        logger.info(f"Processing command {request_id} with action: {action_type}")

        # Add metadata if not present
        if "metadata" not in command:
            command["metadata"] = {}
        command["metadata"]["processed_at"] = datetime.utcnow().isoformat()

        # Get handler and process
        handler = self.handlers.get(action_type)
        if not handler:
            logger.error(f"No handler registered for action type: {action_type}")
            return {"status": "error", "message": f"Unknown action type: {action_type}"}

        try:
            # Execute handler
            result = await handler(command)

            # Log success
            logger.success(f"Command {request_id} processed successfully")

            # Publish event for monitoring
            self._publish_event("command_processed", {
                "request_id": request_id,
                "action_type": action_type,
                "status": "success"
            })

            return result

        except Exception as e:
            logger.error(f"Error processing command {request_id}: {str(e)}")
            self._publish_event("command_failed", {
                "request_id": request_id,
                "action_type": action_type,
                "error": str(e)
            })
            return {"status": "error", "message": str(e)}

    def _validate_command(self, command: Dict[str, Any]) -> bool:
        """Validate command against the master schema."""
        required_fields = ["request_id", "action_type", "payload"]
        return all(field in command for field in required_fields)

    def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish an event to Redis for monitoring."""
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        self.redis.publish("zanalytics:events", json.dumps(event))

    # Default Handlers

    async def _handle_journal_entry(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle journal entry logging."""
        payload = command["payload"]

        # Create journal entry
        entry = {
            "id": command["request_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "type": payload.get("type", "general"),
            "source": payload.get("source", "unknown"),
            "content": payload.get("content", ""),
            "metadata": payload.get("metadata", {})
        }

        # Store in Redis
        self.redis.hset(
            "zanalytics:journal",
            entry["id"],
            json.dumps(entry)
        )

        # Publish to journal channel for real-time updates
        self.redis.publish("zanalytics:journal:new", json.dumps(entry))

        return {"status": "success", "entry_id": entry["id"]}

    async def _handle_dashboard_update(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dashboard state updates."""
        payload = command["payload"]

        # Publish update to dashboard channel
        update_message = {
            "request_id": command["request_id"],
            "target_component": payload.get("target_component"),
            "action": payload.get("action"),
            "params": payload.get("params", {})
        }

        self.redis.publish("zanalytics:dashboard:commands", json.dumps(update_message))

        return {"status": "success", "message": "Dashboard update published"}

    async def _handle_agent_trigger(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle triggering of analysis agents."""
        payload = command["payload"]
        agent_name = payload.get("agent_name")
        mission = payload.get("mission")
        context = payload.get("context", {})

        # Create agent task
        task = {
            "task_id": f"{command['request_id']}_agent",
            "agent_name": agent_name,
            "mission": mission,
            "context": context,
            "requested_at": datetime.utcnow().isoformat()
        }

        # Queue task for agent execution
        self.redis.lpush(f"zanalytics:agent_tasks:{agent_name}", json.dumps(task))

        return {"status": "success", "task_id": task["task_id"]}

    async def _handle_notification(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user notifications."""
        payload = command["payload"]

        notification = {
            "id": command["request_id"],
            "level": payload.get("level", "info"),
            "message": payload.get("message"),
            "timestamp": datetime.utcnow().isoformat(),
            "data": payload.get("data", {})
        }

        # Store notification
        self.redis.lpush("zanalytics:notifications", json.dumps(notification))

        # Publish for real-time delivery
        self.redis.publish("zanalytics:notifications:new", json.dumps(notification))

        return {"status": "success", "notification_id": notification["id"]}

    async def _handle_trade_idea(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle trade idea execution."""
        payload = command["payload"]

        # First log as journal entry
        await self._handle_journal_entry({
            "request_id": f"{command['request_id']}_journal",
            "action_type": "LOG_JOURNAL_ENTRY",
            "payload": {
                "type": "TradeIdea",
                "source": payload.get("source", "system"),
                "content": payload.get("content"),
                "metadata": payload.get("trade_setup", {})
            }
        })

        # Then update dashboard
        if "charts_to_highlight" in payload:
            for chart in payload["charts_to_highlight"]:
                await self._handle_dashboard_update({
                    "request_id": f"{command['request_id']}_chart_{chart['timeframe']}",
                    "action_type": "UPDATE_DASHBOARD_STATE",
                    "payload": {
                        "target_component": f"chart_{payload['trade_setup']['symbol']}_{chart['timeframe']}",
                        "action": "add_annotations",
                        "params": {
                            "annotations": chart["annotations"]
                        }
                    }
                })

        # Send high-priority notification
        await self._handle_notification({
            "request_id": f"{command['request_id']}_notify",
            "action_type": "NOTIFY_USER",
            "payload": {
                "level": "high",
                "message": f"New trade setup: {payload['trade_setup']['symbol']} {payload['trade_setup']['direction']}",
                "data": payload["trade_setup"]
            }
        })

        return {"status": "success", "message": "Trade idea processed and distributed"}

    async def _handle_market_update(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market context updates."""
        payload = command["payload"]

        # Update market context in Redis
        context_key = f"zanalytics:market_context:{payload.get('symbol', 'general')}"
        self.redis.hset(context_key, "data", json.dumps(payload))
        self.redis.hset(context_key, "updated_at", datetime.utcnow().isoformat())

        return {"status": "success", "message": "Market context updated"}

    async def _handle_schedule_followup(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scheduling of follow-up actions."""
        payload = command["payload"]

        followup = {
            "id": command["request_id"],
            "scheduled_for": payload.get("scheduled_time"),
            "action": payload.get("action"),
            "params": payload.get("params", {})
        }

        # Add to scheduled tasks
        self.redis.zadd(
            "zanalytics:scheduled_tasks",
            {json.dumps(followup): datetime.fromisoformat(followup["scheduled_for"]).timestamp()}
        )

        return {"status": "success", "followup_id": followup["id"]}

    async def _handle_archive(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle archiving of analyses."""
        payload = command["payload"]

        archive_entry = {
            "id": command["request_id"],
            "archived_at": datetime.utcnow().isoformat(),
            "type": payload.get("type", "analysis"),
            "data": payload.get("data", {}),
            "reason": payload.get("reason", "completed")
        }

        # Move to archive
        self.redis.hset(
            "zanalytics:archive",
            archive_entry["id"],
            json.dumps(archive_entry)
        )

        return {"status": "success", "archived_id": archive_entry["id"]}


# Singleton instance for easy access
_dispatcher_instance: Optional[ActionDispatcher] = None

def get_dispatcher(redis_client: redis.Redis, config: Dict[str, Any]) -> ActionDispatcher:
    """Get or create the singleton dispatcher instance."""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = ActionDispatcher(redis_client, config)
    return _dispatcher_instance
