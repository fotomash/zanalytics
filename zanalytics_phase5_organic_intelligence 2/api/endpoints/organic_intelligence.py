"""
API Endpoints for the Organic Intelligence Loop.
Integrates with existing Zanalytics API to provide new capabilities.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import redis

from core.command_processor import OrganicIntelligenceOrchestrator

# Create router for organic intelligence endpoints
router = APIRouter(prefix="/api/organic", tags=["organic_intelligence"])

# Pydantic models for request/response

class CommandRequest(BaseModel):
    """Request model for executing a command."""
    action_type: str = Field(..., description="Type of action to perform")
    payload: Dict[str, Any] = Field(..., description="Action-specific payload")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class LLMPromptRequest(BaseModel):
    """Request model for processing an LLM prompt."""
    prompt: str = Field(..., description="The prompt to send to the LLM")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    model: Optional[str] = Field("gpt-4", description="LLM model to use")

class ScheduleRequest(BaseModel):
    """Request model for scheduling a command."""
    schedule_time: str = Field(..., description="ISO format datetime for execution")
    command: CommandRequest = Field(..., description="Command to execute")

class StrategyStatusResponse(BaseModel):
    """Response model for strategy status."""
    strategy_id: str
    enabled: bool
    last_execution: Optional[str]
    next_scheduled: Optional[str]
    recent_trades: List[Dict[str, Any]]

# Initialize orchestrator (this would be done in main app startup)
orchestrator: Optional[OrganicIntelligenceOrchestrator] = None
redis_client: Optional[redis.Redis] = None

def init_organic_intelligence(redis: redis.Redis, config: Dict[str, Any]):
    """Initialize the organic intelligence system."""
    global orchestrator, redis_client
    redis_client = redis
    orchestrator = OrganicIntelligenceOrchestrator(redis, config)

# Endpoints

@router.post("/execute-command", summary="Execute a structured command")
async def execute_command(request: CommandRequest) -> Dict[str, Any]:
    """
    Execute a structured command immediately.

    This endpoint allows direct execution of any supported command type.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Organic Intelligence not initialized")

    command = {
        "request_id": f"api_{datetime.utcnow().timestamp()}",
        "action_type": request.action_type,
        "payload": request.payload,
        "metadata": request.metadata or {}
    }

    result = await orchestrator.command_processor.dispatcher.process_command(command)
    return result

@router.post("/process-prompt", summary="Process an LLM prompt")
async def process_prompt(request: LLMPromptRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Process a natural language prompt through the LLM and execute any resulting commands.

    This endpoint demonstrates the full Organic Intelligence Loop:
    1. Send prompt to LLM
    2. Parse structured commands from response
    3. Execute commands through the dispatcher
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Organic Intelligence not initialized")

    # In production, this would call your actual LLM API
    # For demonstration, we'll simulate an LLM response

    # Simulate LLM processing
    if "london" in request.prompt.lower() and "setup" in request.prompt.lower():
        # Simulate LLM detecting need for London Killzone analysis
        llm_response = json.dumps({
            "request_id": f"llm_{datetime.utcnow().timestamp()}",
            "action_type": "TRIGGER_AGENT_ANALYSIS",
            "payload": {
                "agent_name": "LondonKillzone_SMC_v1",
                "mission": "Analyze current market for London Killzone setups",
                "context": {
                    "user_prompt": request.prompt,
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            "human_readable_summary": "I'll analyze the current market for London Killzone trading opportunities."
        })
    else:
        # Default response
        llm_response = json.dumps({
            "request_id": f"llm_{datetime.utcnow().timestamp()}",
            "action_type": "LOG_JOURNAL_ENTRY",
            "payload": {
                "type": "Analysis",
                "source": "LLM",
                "content": f"Processed prompt: {request.prompt}"
            },
            "human_readable_summary": "I've logged your request for analysis."
        })

    # Process the LLM response
    result = await orchestrator.process_llm_response(llm_response)

    return {
        "status": "success",
        "llm_response": json.loads(llm_response),
        "execution_result": result
    }

@router.post("/schedule-command", summary="Schedule a command for future execution")
async def schedule_command(request: ScheduleRequest) -> Dict[str, Any]:
    """
    Schedule a command to be executed at a specific time.
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not initialized")

    # Create scheduled task
    scheduled_task = {
        "id": f"scheduled_{datetime.utcnow().timestamp()}",
        "scheduled_for": request.schedule_time,
        "action": request.command.action_type,
        "params": request.command.payload
    }

    # Add to scheduled tasks
    schedule_timestamp = datetime.fromisoformat(request.schedule_time.replace('Z', '+00:00')).timestamp()
    redis_client.zadd(
        "zanalytics:scheduled_tasks",
        {json.dumps(scheduled_task): schedule_timestamp}
    )

    return {
        "status": "success",
        "scheduled_id": scheduled_task["id"],
        "scheduled_for": request.schedule_time
    }

@router.get("/strategies", summary="Get all active strategies")
async def get_strategies() -> List[Dict[str, Any]]:
    """
    Get list of all active trading strategies.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Organic Intelligence not initialized")

    strategies = []
    for strategy_id, manifest in orchestrator.scheduling_agent.active_strategies.items():
        strategies.append({
            "strategy_id": strategy_id,
            "name": manifest.get("strategy_name"),
            "description": manifest.get("description"),
            "enabled": manifest.get("enabled", False),
            "schedule": manifest.get("schedule", {})
        })

    return strategies

@router.get("/strategies/{strategy_id}/status", summary="Get strategy status")
async def get_strategy_status(strategy_id: str) -> StrategyStatusResponse:
    """
    Get detailed status of a specific strategy.
    """
    if not orchestrator or not redis_client:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Check if strategy exists
    if strategy_id not in orchestrator.scheduling_agent.active_strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    manifest = orchestrator.scheduling_agent.active_strategies[strategy_id]

    # Get last execution info
    status_key = f"zanalytics:agent_status:{strategy_id}"
    status_data = redis_client.get(status_key)
    last_execution = None
    if status_data:
        status_info = json.loads(status_data)
        last_execution = status_info.get("timestamp")

    # Get recent trade ideas
    recent_trades = []
    journal_entries = redis_client.hgetall("zanalytics:journal")
    for entry_data in journal_entries.values():
        entry = json.loads(entry_data)
        if (entry.get("source") == f"{strategy_id}_Agent" and 
            entry.get("type") == "TradeIdea"):
            recent_trades.append(entry.get("metadata", {}))

    # Sort by timestamp and take last 5
    recent_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    recent_trades = recent_trades[:5]

    return StrategyStatusResponse(
        strategy_id=strategy_id,
        enabled=manifest.get("enabled", False),
        last_execution=last_execution,
        next_scheduled=None,  # Would calculate from scheduler
        recent_trades=recent_trades
    )

@router.post("/strategies/{strategy_id}/trigger", summary="Manually trigger a strategy")
async def trigger_strategy(strategy_id: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Manually trigger execution of a specific strategy.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Organic Intelligence not initialized")

    if strategy_id not in orchestrator.scheduling_agent.active_strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Trigger the strategy in background
    background_tasks.add_task(
        orchestrator.scheduling_agent._trigger_strategy,
        strategy_id
    )

    return {
        "status": "success",
        "message": f"Strategy {strategy_id} triggered for execution"
    }

@router.get("/journal/recent", summary="Get recent journal entries")
async def get_recent_journal(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent journal entries from the organic intelligence system.
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not initialized")

    entries = []
    journal_data = redis_client.hgetall("zanalytics:journal")

    for entry_data in journal_data.values():
        entry = json.loads(entry_data)
        entries.append(entry)

    # Sort by timestamp and limit
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return entries[:limit]

@router.get("/notifications/recent", summary="Get recent notifications")
async def get_recent_notifications(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent notifications from the system.
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not initialized")

    notifications = []
    notification_data = redis_client.lrange("zanalytics:notifications", 0, limit - 1)

    for notif_json in notification_data:
        notifications.append(json.loads(notif_json))

    return notifications

# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates from the Organic Intelligence system.
    """
    await websocket.accept()

    if not redis_client:
        await websocket.close(code=1003, reason="System not initialized")
        return

    # Subscribe to Redis channels
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(
        "zanalytics:journal:new",
        "zanalytics:notifications:new",
        "zanalytics:events"
    )

    try:
        while True:
            # Get message from Redis
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)

            if message and message['type'] == 'message':
                # Send to WebSocket client
                await websocket.send_json({
                    "channel": message['channel'].decode('utf-8'),
                    "data": json.loads(message['data'])
                })

    except WebSocketDisconnect:
        # Client disconnected
        await pubsub.unsubscribe()
        await pubsub.close()
