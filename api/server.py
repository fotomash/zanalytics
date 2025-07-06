'''\nZAnalytics API Server\nAuto-generated basic API implementation.\n'''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from core.orchestrator import get_orchestrator, AnalysisRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ZAnalytics API",
    description="API for ZAnalytics trading analysis system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequestModel(BaseModel):
    symbol: str
    timeframe: str
    analysis_type: str = "combined"
    parameters: Optional[Dict[str, Any]] = {}

class AnalysisResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime

orchestrator = get_orchestrator()

@app.get("/")
async def root():
    return {
        "name": "ZAnalytics API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "orchestrator": orchestrator.is_running,
            "engines": orchestrator.get_engine_status()
        }
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequestModel):
    try:
        analysis_request = AnalysisRequest(
            request_id=f"{request.symbol}_{request.timeframe}_{datetime.now().timestamp()}",
            symbol=request.symbol,
            timeframe=request.timeframe,
            analysis_type=request.analysis_type,
            parameters=request.parameters
        )
        request_id = orchestrator.submit_request(analysis_request)
        result = orchestrator.get_result(request_id, timeout=30)
        if result:
            return AnalysisResponse(
                request_id=request_id,
                status="completed" if result.success else "failed",
                result=result.result_data if result.success else None,
                error=result.error,
                timestamp=result.timestamp
            )
        return AnalysisResponse(
            request_id=request_id,
            status="timeout",
            error="Analysis request timed out",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    return orchestrator.get_metrics()

@app.get("/symbols")
async def get_symbols():
    return {
        "symbols": orchestrator.config['orchestrator']['analysis']['symbols']
    }

@app.get("/timeframes")
async def get_timeframes():
    return {
        "timeframes": orchestrator.config['orchestrator']['analysis']['timeframes']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
