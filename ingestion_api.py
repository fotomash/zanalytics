# ingestion_api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
from redis_writer import RedisSnapshotWriter, EnrichedRow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ZAnalytics Ingestion API",
    description="Unified data ingestion for trading intelligence",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
redis_writer: Optional[RedisSnapshotWriter] = None

@app.on_event("startup")
async def startup_event():
    global redis_writer
    redis_writer = RedisSnapshotWriter()
    await redis_writer.initialize()
    
    # Start auto-snapshot scheduler
    asyncio.create_task(redis_writer.auto_snapshot_scheduler(interval_hours=1))
    logger.info("Ingestion API started successfully")

# Data models
class RawCandle(BaseModel):
    """Raw candle data from MT5"""
    symbol: str
    timeframe: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    
class SignalData(BaseModel):
    """Strategy signals to be merged with candle"""
    wyckoff_phase: Optional[str] = "unknown"
    spring_event: Optional[bool] = False
    smc_zone_type: Optional[str] = "none"
    confluence_score: Optional[float] = 0.0
    trade_entry: Optional[bool] = False
    
class IngestRequest(BaseModel):
    """Complete ingestion request"""
    candle: RawCandle
    signals: Optional[SignalData] = SignalData()
    run_id: Optional[str] = Field(default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

class CandelEnricher:
    """The semantic transformer - converts raw data to enriched schema"""
    
    @staticmethod
    def enrich(raw_candle: RawCandle, signals: SignalData, run_id: str) -> EnrichedRow:
        """Transform raw candle + signals into enriched schema row"""
        
        # Calculate confluence score
        confluence = 0
        if signals.spring_event:
            confluence += 25
        if signals.smc_zone_type != "none":
            confluence += 20
        if signals.confluence_score:
            confluence += signals.confluence_score
            
        confluence = min(confluence, 100)
        
        # Determine trade quality
        if confluence >= 75:
            quality = "HIGH"
        elif confluence >= 50:
            quality = "MEDIUM"
        else:
            quality = "LOW"
            
        return EnrichedRow(
            timestamp=raw_candle.timestamp,
            symbol=raw_candle.symbol,
            timeframe=raw_candle.timeframe,
            open=raw_candle.open,
            high=raw_candle.high,
            low=raw_candle.low,
            close=raw_candle.close,
            volume=raw_candle.volume,
            confluence_score=confluence,
            trade_quality=quality,
            wyckoff_phase=signals.wyckoff_phase or "unknown",
            spring_event=signals.spring_event or False,
            smc_zone_type=signals.smc_zone_type or "none",
            trade_entry=signals.trade_entry or False,
            run_id=run_id,
            annotated=True,
            review_status="auto"
        )

# API Endpoints
@app.post("/api/v1/ingest/candle")
async def ingest_candle(request: IngestRequest, background_tasks: BackgroundTasks):
    """Main ingestion endpoint - enriches and stores candle data"""
    try:
        # Enrich the raw data
        enriched_row = CandelEnricher.enrich(
            raw_candle=request.candle,
            signals=request.signals,
            run_id=request.run_id
        )
        
        # Write to Redis
        await redis_writer.write_enriched_row(enriched_row)
        
        logger.info(f"Ingested {request.candle.symbol} - Confluence: {enriched_row.confluence_score}")
        
        return {
            "status": "success",
            "enriched_data": enriched_row.to_dict(),
            "message": f"Data enriched and stored for {request.candle.symbol}"
        }
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/latest/{symbol}")
async def get_latest_data(symbol: str, timeframe: str = "H1", count: int = 100):
    """Get latest enriched data for analysis"""
    try:
        data = await redis_writer.get_latest_data(symbol.upper(), timeframe, count)
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "count": len(data),
            "data": data
        }
        
    except Exception as e:
        logger.error(f"Data retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/snapshot/create")
async def create_snapshot(symbol: Optional[str] = None):
    """Manually trigger snapshot creation"""
    try:
        snapshot_file = await redis_writer.create_snapshot(symbol)
        
        return {
            "status": "success",
            "snapshot_file": snapshot_file,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Snapshot creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analysis/confluence/{symbol}")
async def analyze_confluence(symbol: str, timeframe: str = "H1"):
    """LLM-friendly confluence analysis endpoint"""
    try:
        data = await redis_writer.get_latest_data(symbol.upper(), timeframe, 10)
        
        if not data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
            
        # Calculate confluence stats
        avg_confluence = sum(item.get('confluence_score', 0) for item in data) / len(data)
        high_quality_setups = [item for item in data if item.get('trade_quality') == 'HIGH']
        spring_events = [item for item in data if item.get('spring_event')]
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "analysis": {
                "average_confluence": round(avg_confluence, 2),
                "high_quality_setups": len(high_quality_setups),
                "spring_events": len(spring_events),
                "latest_confluence": data[0].get('confluence_score', 0) if data else 0,
                "current_phase": data[0].get('wyckoff_phase', 'unknown') if data else 'unknown',
                "trade_recommendation": "LONG" if avg_confluence > 70 else "WAIT"
            },
            "recent_data": data[:5]  # Last 5 bars for context
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": "connected" if redis_writer and redis_writer.redis_client else "disconnected",
            "snapshots": "enabled"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)