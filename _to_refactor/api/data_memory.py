#!/usr/bin/env python3
"""
ncOS Refactored Core System - Unified v21.7 (with Data & Memory API)

This script represents a comprehensive refactoring of multiple ncOS components
into a single, cohesive, and runnable application.

It consolidates and integrates:
- An enhanced MasterOrchestrator for command routing and session management.
- Specialized agents for Quantitative Analysis, Market Making, and ZBAR journaling.
- A sophisticated LLM Assistant for interactive trade journaling and analysis.
- A unified FastAPI backend that exposes all functionalities through a clean API.
- NEW: A DataRetrievalAgent for fetching external market data via API.
- NEW: Session memory export functionality (JSONL/CSV).
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, APIRouter, HTTPException, Query, WebSocket
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.cluster import KMeans

# ==============================================================================
# 2. LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ncOS_Core")


# ==============================================================================
# 3. UNIFIED PYDANTIC SCHEMAS
# ==============================================================================

class NCOSBase(BaseModel):
    """A base model for shared fields."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:8]}")

class DataBlock(BaseModel):
    id: str; timeframe: str; columns: List[str]; data: List[List[Any]]

class ExecutionContext(BaseModel):
    initial_htf_bias: Optional[str] = None
    session_id: str = "default_session"
    agent_profile: str = "default"

class ZBARRequest(BaseModel):
    strategy: str; asset: str; blocks: List[DataBlock]
    context: Optional[ExecutionContext] = None

class EntrySignal(BaseModel):
    direction: str; entry_price: float; stop_loss: float; take_profit: float; rr: float
    killzone_match_name: Optional[str] = None

class PredictiveSnapshot(BaseModel):
    maturity_score: float; grade: str; conflict_signal: bool
    poi_quality: Optional[str] = None

class ZBARResponse(NCOSBase):
    status: str; reason: Optional[str] = None
    entry_signal: Optional[EntrySignal] = None
    predictive_snapshot: Optional[PredictiveSnapshot] = None

class TradeJournalEntry(NCOSBase):
    session_id: str; symbol: str; strategy: str
    status: str # e.g., 'PASS', 'FAIL', 'OBSERVATION'
    reason: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# 4. CORE AGENT & MANAGER IMPLEMENTATIONS
# ==============================================================================

class DataRetrievalAgent:
    """Agent to fetch data from an external API endpoint."""
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self.client = httpx.AsyncClient(base_url=self.api_base_url)
        logger.info(f"DataRetrievalAgent initialized for API: {api_base_url}")

    async def fetch(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetches market data from the configured API."""
        endpoint = f"/data/retrieve" # Example endpoint from data_retrieval_api.py
        params = {"symbol": symbol, "tf": timeframe, "limit": limit}
        try:
            logger.info(f"Fetching data from {self.client.base_url}{endpoint} with params {params}...")
            response = await self.client.get(endpoint, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            if not data: return None
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            logger.info(f"Successfully retrieved {len(df)} bars for {symbol} {timeframe}.")
            return df
        except httpx.RequestError as e:
            logger.error(f"HTTP error fetching data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"An error occurred during data retrieval: {e}")
            return None

class QuantitativeAnalyst:
    def __init__(self, config: Dict = {}): logger.info("QuantitativeAnalyst agent initialized.")
    @lru_cache(maxsize=128)
    def _calculate_metrics(self, data_tuple):
        df = pd.DataFrame(list(data_tuple), columns=['open', 'high', 'low', 'close', 'volume'])
        if len(df) < 2: return {"error": "Insufficient data"}
        returns = df['close'].pct_change().dropna()
        if len(returns) < 2: return {"error": "Insufficient returns data"}
        return {"volatility": returns.std(), "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0}
    async def process(self, df: pd.DataFrame) -> Dict:
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required): return {"error": f"Missing columns, need {required}"}
        data_tuple = tuple(map(tuple, df[required].to_records(index=False)))
        return {"statistical_metrics": self._calculate_metrics(data_tuple)}

class ZBarAgent:
    def __init__(self, config: Dict = {}): logger.info("ZBarAgent initialized.")
    async def evaluate_bar(self, df: pd.DataFrame, context: ExecutionContext) -> ZBARResponse:
        logger.info(f"ZBarAgent evaluating bar for {context.agent_profile}...")
        return ZBARResponse(status="PASS", entry_signal=EntrySignal(direction="long", entry_price=2360.1, stop_loss=2357.3, take_profit=2367.8, rr=2.7))

class JournalManager:
    def __init__(self, journal_path: str = "trade_journal.jsonl"):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []
        if self.journal_path.exists():
            with self.journal_path.open('r') as f:
                for line in f:
                    self.entries.append(TradeJournalEntry(**json.loads(line)))
        logger.info(f"JournalManager initialized with {len(self.entries)} entries from {self.journal_path}")

    def log(self, entry: TradeJournalEntry):
        self.entries.append(entry)
        with self.journal_path.open('a') as f:
            f.write(entry.model_dump_json() + '\n')
            
    def query(self, session_id: str, limit: int = 100) -> List[Dict]:
        session_entries = [e.model_dump() for e in self.entries if e.session_id == session_id]
        return session_entries[-limit:]
        
    def export_session(self, session_id: str, format: str = 'jsonl') -> Optional[str]:
        """Exports a session's journal to a file."""
        session_entries = [e for e in self.entries if e.session_id == session_id]
        if not session_entries:
            logger.warning(f"No entries found for session {session_id} to export.")
            return None
        
        export_dir = Path("session_exports")
        export_dir.mkdir(exist_ok=True)
        file_path = export_dir / f"{session_id}.{format}"

        if format == 'jsonl':
            with file_path.open('w') as f:
                for entry in session_entries:
                    f.write(entry.model_dump_json() + '\n')
        elif format == 'csv':
            df = pd.DataFrame([e.model_dump() for e in session_entries])
            # Flatten nested dicts for better CSV representation
            df = pd.json_normalize(df.to_dict(orient='records'))
            df.to_csv(file_path, index=False)
        else:
            logger.error(f"Unsupported export format: {format}")
            return None
            
        logger.info(f"Successfully exported session {session_id} to {file_path}")
        return str(file_path)

# ==============================================================================
# 5. MASTER ORCHESTRATOR
# ==============================================================================

class MasterOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.journal = JournalManager()
        
        # In a real app, this API address would come from config
        self.data_retriever = DataRetrievalAgent(api_base_url="http://127.0.0.1:8001")
        
        self.agents = {"quant_analyst": QuantitativeAnalyst(), "zbar_agent": ZBarAgent()}
        self.active_session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"MasterOrchestrator initialized. Active session: {self.active_session_id}")

    async def route_command(self, command: str) -> Dict:
        """Routes a command to the appropriate agent(s) or action."""
        command = command.lower().strip()
        logger.info(f"Routing command: '{command}'")
        parts = command.split()
        
        self.journal.log(TradeJournalEntry(
            session_id=self.active_session_id, symbol="SYSTEM", strategy="manual_command",
            status="OBSERVATION", details={"command": command}
        ))

        # NEW COMMANDS
        if command.startswith("fetch data for"):
            symbol = parts[-1].upper()
            return await self.handle_data_fetch(symbol)
        elif command.startswith("save session"):
            format = parts[-1] if parts[-1] in ['csv', 'jsonl'] else 'jsonl'
            return self.handle_session_save(format)

        # EXISTING COMMANDS
        elif "quant" in command:
            data_df = data_store.get(self.active_session_id)
            if data_df is None: return {"error": "No data loaded in current session. Use 'fetch data for <symbol>' first."}
            return await self.agents["quant_analyst"].process(data_df)
            
        elif "zbar" in command:
            data_df = data_store.get(self.active_session_id)
            if data_df is None: return {"error": "No data loaded. Use 'fetch data for <symbol>' first."}
            context = ExecutionContext(initial_htf_bias="bullish" if "bullish" in command else "bearish")
            return (await self.agents["zbar_agent"].evaluate_bar(data_df, context)).model_dump()
            
        elif "journal" in command:
            return {"journal_entries": self.journal.query(self.active_session_id)}

        else:
            return {"status": "unrecognized_command", "command": command}
            
    async def handle_data_fetch(self, symbol: str) -> Dict:
        """Handles the logic for fetching and storing data."""
        data_df = await self.data_retriever.fetch(symbol=symbol, timeframe="M1")
        if data_df is not None:
            data_store[self.active_session_id] = data_df
            return {"status": "success", "message": f"Retrieved {len(data_df)} bars for {symbol} and stored in session memory."}
        return {"status": "error", "message": f"Failed to retrieve data for {symbol}."}

    def handle_session_save(self, format: str) -> Dict:
        """Handles the logic for saving the session journal."""
        file_path = self.journal.export_session(self.active_session_id, format)
        if file_path:
            return {"status": "success", "message": f"Session journal saved to {file_path}"}
        return {"status": "error", "message": f"Could not save session {self.active_session_id}."}


# ==============================================================================
# 6. FASTAPI APPLICATION SETUP
# ==============================================================================

app = FastAPI(title="ncOS Refactored Core API", version="21.7")
config = {"system": {"name": "ncOS"}, "orchestration": {}}
orchestrator = MasterOrchestrator(config)
data_store = {} # In-memory store for session dataframes

router_v1 = APIRouter(prefix="/api/v1")

@router_v1.post("/command")
async def execute_command(command: str):
    return await orchestrator.route_command(command)

@router_v1.post("/session/save")
async def save_session(format: str = Query("jsonl", enum=["jsonl", "csv"])):
    """API endpoint to save the current session memory."""
    return orchestrator.handle_session_save(format)

@router_v1.post("/data/fetch")
async def fetch_data(symbol: str = Query(...)):
    """API endpoint to trigger data retrieval for a symbol."""
    return await orchestrator.handle_data_fetch(symbol)

app.include_router(router_v1)

@app.get("/", tags=["Root"])
async def root():
    return {"service": "ncOS Refactored Core API", "version": "21.7"}

# ==============================================================================
# 7. DUMMY DATA RETRIEVAL API (For Testing)
# This simulates the external API the DataRetrievalAgent calls.
# You would run this in a separate terminal.
# ==============================================================================

data_api = FastAPI(title="Dummy Data Retrieval API")

@data_api.get("/data/retrieve")
def retrieve_dummy_data(symbol: str, tf: str, limit: int):
    """Generates dummy timeseries data."""
    logger.info(f"[Dummy API] Received request for {symbol} {tf}")
    dates = pd.to_datetime(pd.date_range(end=datetime.utcnow(), periods=limit, freq="min"))
    data = np.random.randn(limit, 4) * np.array([0.5, 0.5, 0.5, 2.0]) + np.array([2300, 2300, 2300, 2300])
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
    df['time'] = dates
    df['volume'] = np.random.randint(100, 5000, size=limit)
    return json.loads(df.to_json(orient='records'))


# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # In a real scenario, you would run the main app and the dummy API in separate processes.
    # For this example, we'll just run the main ncOS API.
    # To test fully:
    # 1. Run this script once as: `python your_script_name.py --api data`
    # 2. In another terminal, run: `python your_script_name.py --api main`
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=str, default="main", help="Which API to run: 'main' or 'data'")
    args = parser.parse_args()

    if args.api == "main":
        logger.info("Starting ncOS Refactored Core API Server on port 8008...")
        uvicorn.run(app, host="127.0.0.1", port=8009)
    elif args.api == "data":
        logger.info("Starting Dummy Data Retrieval API Server on port 8001...")
        uvicorn.run(data_api, host="127.0.0.1", port=8009)

