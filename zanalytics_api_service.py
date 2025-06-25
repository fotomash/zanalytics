from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from pydantic import BaseModel
import logging

# Import your custom modules (assuming they exist)
try:
    from data_flow_manager import DataFlowManager
    from agent_registry import AgentRegistry
    from zanalytics_adapter import ZanalyticsAdapter
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Running in standalone mode...")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ZANALYTICS API Service",
    description="Real-time trading data analysis and agent decision API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    def __init__(self):
        self.data_flow_manager = None
        self.agent_registry = None
        self.zanalytics_adapter = None
        self.websocket_connections: List[WebSocket] = []
        self.analysis_cache = {}
        self.agent_decisions = []
        self.system_status = "INITIALIZING"

    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize components if modules are available
            try:
                self.agent_registry = AgentRegistry()
                self.zanalytics_adapter = ZanalyticsAdapter()
                self.data_flow_manager = DataFlowManager(
                    zanalytics_callback=self.process_zanalytics_event
                )
                self.system_status = "ONLINE"
                logger.info("All components initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize full system: {e}")
                self.system_status = "STANDALONE"
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            self.system_status = "ERROR"

    async def process_zanalytics_event(self, event_data):
        """Process ZANALYTICS events"""
        try:
            # Cache the analysis
            self.analysis_cache[event_data.get('symbol', 'UNKNOWN')] = event_data

            # Broadcast to WebSocket clients
            await self.broadcast_to_websockets({
                "type": "analysis_update",
                "data": event_data,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"Processed ZANALYTICS event for {event_data.get('symbol', 'UNKNOWN')}")
        except Exception as e:
            logger.error(f"Error processing ZANALYTICS event: {e}")

    async def broadcast_to_websockets(self, message):
        """Broadcast message to all connected WebSocket clients"""
        if not self.websocket_connections:
            return

        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except:
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)

# Global app state
state = AppState()

# Pydantic models
class SystemStatus(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]
    active_symbols: int
    events_processed: int
    analysis_count: int

class AgentDecision(BaseModel):
    agent_id: str
    agent_type: str
    symbol: str
    decision: str
    confidence: float
    reasoning: str
    timestamp: str

class AnalysisSummary(BaseModel):
    symbol: str
    timeframe: str
    price: float
    trend: str
    signals: List[Dict]
    patterns: List[Dict]
    risk_metrics: Dict
    timestamp: str

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    await state.initialize()
    logger.info("ZANALYTICS API Service started successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ZANALYTICS API",
        "version": "1.0.0",
        "status": state.system_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status"""
    components = {
        "data_flow_manager": "ONLINE" if state.data_flow_manager else "OFFLINE",
        "agent_registry": "ONLINE" if state.agent_registry else "OFFLINE",
        "zanalytics_adapter": "ONLINE" if state.zanalytics_adapter else "OFFLINE",
        "websocket_connections": f"{len(state.websocket_connections)} active"
    }

    return SystemStatus(
        status=state.system_status,
        timestamp=datetime.now().isoformat(),
        components=components,
        active_symbols=len(state.analysis_cache),
        events_processed=len(state.agent_decisions),
        analysis_count=len(state.analysis_cache)
    )

@app.get("/agents/decisions")
async def get_agent_decisions():
    """Get recent agent decisions"""
    try:
        # Return recent decisions (last 50)
        recent_decisions = state.agent_decisions[-50:] if state.agent_decisions else []

        # If no real decisions, return mock data for demo
        if not recent_decisions:
            mock_decisions = [
                {
                    "agent_id": "risk_manager_001",
                    "agent_type": "RiskManager",
                    "symbol": "XAUUSD",
                    "decision": "REDUCE_EXPOSURE",
                    "confidence": 0.85,
                    "reasoning": "High volatility detected, reducing position size",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "agent_id": "macro_analyst_001",
                    "agent_type": "MacroAnalyst",
                    "symbol": "XAUUSD",
                    "decision": "BULLISH_BIAS",
                    "confidence": 0.72,
                    "reasoning": "Dollar weakness and inflation concerns support gold",
                    "timestamp": datetime.now().isoformat()
                }
            ]
            return {"decisions": mock_decisions, "total": len(mock_decisions)}

        return {"decisions": recent_decisions, "total": len(recent_decisions)}
    except Exception as e:
        logger.error(f"Error getting agent decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/summary/{symbol}")
async def get_analysis_summary(symbol: str, timeframe: str = "M1"):
    """Get analysis summary for a symbol"""
    try:
        # Check cache first
        if symbol in state.analysis_cache:
            cached_data = state.analysis_cache[symbol]
            return cached_data

        # If no cached data, return mock analysis for demo
        mock_analysis = {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": 2650.45 if symbol == "XAUUSD" else 1.0850,
            "trend": "BULLISH",
            "signals": [
                {"type": "BUY", "strength": 0.75, "source": "RSI_OVERSOLD"},
                {"type": "HOLD", "strength": 0.60, "source": "SUPPORT_LEVEL"}
            ],
            "patterns": [
                {"name": "HAMMER", "confidence": 0.82},
                {"name": "BULLISH_ENGULFING", "confidence": 0.65}
            ],
            "risk_metrics": {
                "volatility": 0.15,
                "max_drawdown": 0.03,
                "sharpe_ratio": 1.25
            },
            "timestamp": datetime.now().isoformat()
        }

        return mock_analysis
    except Exception as e:
        logger.error(f"Error getting analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_active_symbols():
    """Get list of active symbols"""
    symbols = list(state.analysis_cache.keys()) if state.analysis_cache else ["XAUUSD", "EURUSD"]
    return {"symbols": symbols, "total": len(symbols)}

@app.post("/analysis/trigger/{symbol}")
async def trigger_analysis(symbol: str, timeframe: str = "M1"):
    """Trigger analysis for a symbol"""
    try:
        if state.zanalytics_adapter:
            # Trigger real analysis
            result = await state.zanalytics_adapter.trigger_analysis(symbol, timeframe)
            return {"status": "triggered", "symbol": symbol, "timeframe": timeframe, "result": result}
        else:
            # Mock response
            return {"status": "triggered", "symbol": symbol, "timeframe": timeframe, "message": "Analysis queued"}
    except Exception as e:
        logger.error(f"Error triggering analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    state.websocket_connections.append(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(state.websocket_connections)}")

    try:
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "status": state.system_status,
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong)
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.websocket_connections:
            state.websocket_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(state.websocket_connections)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_status": state.system_status,
        "connections": len(state.websocket_connections)
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    print("🚀 Starting ZANALYTICS API Service...")
    print("📊 Service will be available at: http://localhost:5010")
    print("🔌 WebSocket endpoint: ws://localhost:5010/ws")
    print("📖 API docs: http://localhost:5010/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5010,
        log_level="info",
        access_log=True
    )
