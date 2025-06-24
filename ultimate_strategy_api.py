#!/usr/bin/env python3
"""
Ultimate Strategy API Server
FastAPI server that serves merged strategy outputs + live endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from ultimate_strategy_merger import StrategyOutputMerger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Ultimate Strategy API",
    description="XANA-ready trading strategy data merger and API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global merger instance
merger = None
DATA_DIR = "./data"

@app.on_event("startup")
async def startup_event():
    """Initialize merger on startup"""
    global merger
    merger = StrategyOutputMerger(data_dir=DATA_DIR, tick_window_size=100)
    logger.info(f"🚀 API started with data directory: {DATA_DIR}")

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "service": "Ultimate Strategy API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "data_dir": DATA_DIR
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    data_path = Path(DATA_DIR)

    # Count available files
    summary_files = len(list(data_path.glob("SUMMARY_*.json")))
    tick_files = len(list(data_path.glob("*TICK*.csv"))) + len(list(data_path.glob("*tick*.csv")))

    return {
        "status": "healthy",
        "data_directory": str(data_path.absolute()),
        "data_exists": data_path.exists(),
        "summary_files": summary_files,
        "tick_files": tick_files,
        "last_check": datetime.utcnow().isoformat()
    }

@app.get("/summary/consolidated")
async def get_consolidated_summary(symbol: str = "XAUUSD"):
    """
    GET THE MERGED SNAPSHOT - Main endpoint for ChatGPT/dashboards
    Returns consolidated multi-TF + microstructure data
    """
    try:
        if not merger:
            raise HTTPException(status_code=500, detail="Merger not initialized")

        # Generate fresh merged data
        merged_data = merger.merge_strategy_outputs(symbol)

        if "error" in merged_data:
            raise HTTPException(status_code=500, detail=f"Merger error: {merged_data['error']}")

        # Add API metadata
        merged_data["api"] = {
            "endpoint": "/summary/consolidated",
            "served_at": datetime.utcnow().isoformat(),
            "data_freshness": "live"
        }

        return merged_data

    except Exception as e:
        logger.error(f"Consolidated summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/latest")
async def get_latest_summaries(symbol: str = "XAUUSD"):
    """Get individual summary files (legacy endpoint)"""
    try:
        data_path = Path(DATA_DIR)
        summaries = {}

        # Find all summary files
        pattern = f"{symbol}_*_SUMMARY_*.json"
        files = list(data_path.glob(pattern))

        if not files:
            files = list(data_path.glob("SUMMARY_*.json"))

        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Extract timeframe from filename
                filename = file_path.name
                if "SUMMARY_" in filename:
                    tf = filename.split("_")[-1].replace(".json", "")
                    summaries[tf] = data

            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")

        return {
            "symbol": symbol,
            "summaries": summaries,
            "loaded_at": datetime.utcnow().isoformat(),
            "total_timeframes": len(summaries)
        }

    except Exception as e:
        logger.error(f"Latest summaries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/microstructure/tick-window")
async def get_tick_window(symbol: str = "XAUUSD", limit: int = 100):
    """Get just the tick microstructure window"""
    try:
        if not merger:
            raise HTTPException(status_code=500, detail="Merger not initialized")

        # Set temporary limit
        original_limit = merger.tick_window_size
        merger.tick_window_size = limit

        # Get microstructure
        microstructure = merger._load_tick_microstructure(symbol)

        # Restore original limit
        merger.tick_window_size = original_limit

        return {
            "symbol": symbol,
            "microstructure": microstructure,
            "tick_count": len(microstructure.get("tick_window", [])),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Tick window error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals/entry")
async def get_entry_signals(symbol: str = "XAUUSD"):
    """Get just the entry signals"""
    try:
        if not merger:
            raise HTTPException(status_code=500, detail="Merger not initialized")

        # Get full merged data
        merged_data = merger.merge_strategy_outputs(symbol)

        return {
            "symbol": symbol,
            "entry_signals": merged_data.get("entry_signals", {}),
            "confluence_score": merged_data.get("entry_signals", {}).get("confluence", {}).get("total_score", 0),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Entry signals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/merge/trigger")
async def trigger_merge(background_tasks: BackgroundTasks, symbol: str = "XAUUSD", save_output: bool = True):
    """Manually trigger a merge operation"""
    try:
        if not merger:
            raise HTTPException(status_code=500, detail="Merger not initialized")

        # Run merge
        merged_data = merger.merge_strategy_outputs(symbol)

        if save_output and "error" not in merged_data:
            # Save in background
            background_tasks.add_task(
                merger.save_merged_output, 
                merged_data, 
                f"{DATA_DIR}/merged_snapshot_latest.json"
            )

        return {
            "status": "success" if "error" not in merged_data else "error",
            "symbol": symbol,
            "timeframes_merged": len(merged_data.get("summaries", {})),
            "ticks_processed": len(merged_data.get("microstructure", {}).get("tick_window", [])),
            "output_saved": save_output,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Trigger merge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """Get current API configuration"""
    return {
        "data_directory": DATA_DIR,
        "tick_window_size": merger.tick_window_size if merger else 100,
        "supported_symbols": ["XAUUSD", "EURUSD", "GBPUSD"],  # Configurable
        "endpoints": {
            "consolidated": "/summary/consolidated",
            "tick_window": "/microstructure/tick-window", 
            "entry_signals": "/signals/entry",
            "trigger_merge": "/merge/trigger"
        }
    }

@app.put("/config")
async def update_config(data_dir: Optional[str] = None, tick_window_size: Optional[int] = None):
    """Update API configuration"""
    global merger, DATA_DIR

    changes = {}

    if data_dir:
        DATA_DIR = data_dir
        changes["data_dir"] = DATA_DIR

    if tick_window_size:
        if merger:
            merger.tick_window_size = tick_window_size
        changes["tick_window_size"] = tick_window_size

    # Reinitialize merger if needed
    if data_dir:
        merger = StrategyOutputMerger(data_dir=DATA_DIR, 
                                    tick_window_size=merger.tick_window_size if merger else 100)

    return {
        "status": "updated",
        "changes": changes,
        "timestamp": datetime.utcnow().isoformat()
    }

# WebSocket endpoint for live updates (optional)
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

@app.websocket("/ws/live")
async def websocket_live_updates(websocket: WebSocket, symbol: str = "XAUUSD"):
    """WebSocket for live strategy updates"""
    await websocket.accept()

    try:
        while True:
            # Get fresh data every 30 seconds
            if merger:
                merged_data = merger.merge_strategy_outputs(symbol)
                await websocket.send_json({
                    "type": "strategy_update",
                    "data": merged_data,
                    "timestamp": datetime.utcnow().isoformat()
                })

            await asyncio.sleep(30)  # Update every 30 seconds

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

def run_server(host: str = "0.0.0.0", port: int = 8000, data_dir: str = "./data"):
    """Run the FastAPI server"""
    global DATA_DIR
    DATA_DIR = data_dir

    logger.info(f"🚀 Starting Ultimate Strategy API...")
    logger.info(f"📊 Data directory: {DATA_DIR}")
    logger.info(f"🌐 Server: http://{host}:{port}")
    logger.info(f"📚 Docs: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ultimate Strategy API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--data-dir', default='./data', help='Data directory')

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, data_dir=args.data_dir)
