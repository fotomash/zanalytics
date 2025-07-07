# mt5_bridge_client.py
import asyncio
import aiohttp
import json
from typing import Dict, Any
from datetime import datetime

class MT5IngestionBridge:
    """Bridges MT5 data to the ingestion API"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session: aiohttp.ClientSession = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        
    async def send_candle(self, symbol: str, timeframe: str, ohlcv: Dict[str, Any], 
                         signals: Dict[str, Any] = None):
        """Send enriched candle to ingestion API"""
        
        payload = {
            "candle": {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                **ohlcv
            },
            "signals": signals or {}
        }
        
        async with self.session.post(
            f"{self.api_base_url}/api/v1/ingest/candle",
            json=payload
        ) as response:
            return await response.json()
    
    async def get_analysis(self, symbol: str, timeframe: str = "H1"):
        """Get confluence analysis for LLM"""
        async with self.session.get(
            f"{self.api_base_url}/api/v1/analysis/confluence/{symbol}",
            params={"timeframe": timeframe}
        ) as response:
            return await response.json()

# Example usage from MT5 script
async def example_mt5_integration():
    bridge = MT5IngestionBridge()
    await bridge.initialize()
    
    # Simulate MT5 candle with strategy signals
    ohlcv = {
        "open": 2340.50,
        "high": 2342.00,
        "low": 2339.80,
        "close": 2341.25,
        "volume": 1250
    }
    
    signals = {
        "wyckoff_phase": "C",
        "spring_event": True,
        "smc_zone_type": "breaker",
        "confluence_score": 85.0,
        "trade_entry": True
    }
    
    # Send to ingestion API
    result = await bridge.send_candle("XAUUSD", "H1", ohlcv, signals)
    print(f"Ingestion result: {result}")
    
    # Get analysis for LLM
    analysis = await bridge.get_analysis("XAUUSD", "H1")
    print(f"LLM Analysis: {analysis}")

if __name__ == "__main__":
    asyncio.run(example_mt5_integration())