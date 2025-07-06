#!/usr/bin/env python3
"""
Trading Analytics API Server
FastAPI server providing REST endpoints for all analysis components
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Import our modules
from config_helper from config import orchestrator_config
from custom_gpt_router import gpt_router
from redis_server import redis_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config.api.title,
    description=config.api.description,
    version=config.api.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: Optional[str] = "auto"
    analysis_types: List[str] = ["wyckoff", "smc", "microstructure"]
    use_cache: bool = True
    include_gpt: bool = True

class SymbolListResponse(BaseModel):
    symbols: List[str]
    total_count: int
    cached_symbols: List[str]

class AnalysisResponse(BaseModel):
    symbol: str
    timestamp: str
    analysis_results: Dict
    gpt_insights: Optional[Dict] = None
    cache_hit: bool = False
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict
    version: str

# Helper functions
def load_json_analysis(symbol: str) -> Optional[Dict]:
    """Load JSON analysis from file system"""
    json_path = config.data.json_path
    
    # Try multiple file patterns
    patterns = [
        f"{symbol}_comprehensive.json",
        f"{symbol}_analysis.json",
        f"{symbol}.json",
        f"{symbol}_tick.json"
    ]
    
    for pattern in patterns:
        file_path = json_path / pattern
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
    
    return None

def load_parquet_data(symbol: str, timeframe: str = "5min") -> Optional[pd.DataFrame]:
    """Load parquet data for analysis"""
    parquet_path = config.data.parquet_path
    file_path = parquet_path / f"{symbol}_{timeframe}.parquet"
    
    if file_path.exists():
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Failed to load parquet data: {e}")
    
    return None

async def run_wyckoff_analysis(df: pd.DataFrame) -> Dict:
    """Run Wyckoff analysis on DataFrame"""
    try:
        # Import your Wyckoff analyzer here
        # from your_wyckoff_module import WyckoffAnalyzer
        # analyzer = WyckoffAnalyzer()
        # return analyzer.analyze_data(df)
        
        # Placeholder implementation
        return {
            "phases": {"accumulation": 25, "distribution": 15, "markup": 35, "markdown": 25},
            "trend_analysis": {"trend": "bullish"},
            "composite_operator": {"institutional_activity": "moderate"}
        }
    except Exception as e:
        logger.error(f"Wyckoff analysis failed: {e}")
        return {"error": str(e)}

# API Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services=config.get_system_status(),
        version=config.api.version
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    system_status = config.get_system_status()
    
    # Test Redis connection
    redis_healthy = redis_manager.redis_client.ping() if redis_manager.redis_client else False
    
    # Test data directories
    data_healthy = config.data.json_path.exists() and config.data.parquet_path.exists()
    
    overall_status = "healthy" if redis_healthy and data_healthy else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        services={
            **system_status,
            "redis_ping": redis_healthy,
            "data_directories": data_healthy
        },
        version=config.api.version
    )

@app.get("/symbols", response_model=SymbolListResponse)
async def get_symbols():
    """Get available symbols"""
    # Get symbols from parquet files
    parquet_symbols = []
    if config.data.parquet_path.exists():
        for file in config.data.parquet_path.glob("*.parquet"):
            symbol = file.stem.split("_")[0]
            if symbol not in parquet_symbols:
                parquet_symbols.append(symbol)
    
    # Get cached symbols
    cached_symbols = redis_manager.get_active_symbols()
    
    # Combine and deduplicate
    all_symbols = sorted(list(set(parquet_symbols + cached_symbols)))
    
    return SymbolListResponse(
        symbols=all_symbols,
        total_count=len(all_symbols),
        cached_symbols=cached_symbols
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_symbol(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Comprehensive symbol analysis"""
    start_time = datetime.now()
    cache_hit = False
    
    try:
        # Check cache first
        if request.use_cache:
            cached_data = redis_manager.get_symbol_data(request.symbol)
            if cached_data and cached_data.get("analysis_results"):
                cache_hit = True
                logger.info(f"Cache hit for {request.symbol}")
                
                analysis_results = {}
                for result in cached_data["analysis_results"]:
                    analysis_results[result["analysis_type"]] = result["data"]
                
                # Get GPT insights if requested
                gpt_insights = None
                if request.include_gpt:
                    gpt_insights = await gpt_router.get_comprehensive_insights(
                        request.symbol, analysis_results
                    )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return AnalysisResponse(
                    symbol=request.symbol,
                    timestamp=datetime.now().isoformat(),
                    analysis_results=analysis_results,
                    gpt_insights=gpt_insights,
                    cache_hit=cache_hit,
                    processing_time=processing_time
                )
        
        # Perform fresh analysis
        analysis_results = {}
        
        # Load JSON analysis if available
        json_data = load_json_analysis(request.symbol)
        if json_data:
            analysis_results["json"] = json_data
            
            # Store in Redis
            redis_manager.store_json_analysis(request.symbol, json_data)
        
        # Run requested analysis types
        for analysis_type in request.analysis_types:
            if analysis_type == "wyckoff":
                # Load parquet data and run Wyckoff analysis
                df = load_parquet_data(request.symbol, request.timeframe)
                if df is not None:
                    wyckoff_results = await run_wyckoff_analysis(df)
                    analysis_results["wyckoff"] = wyckoff_results
                    
                    # Cache results
                    redis_manager.store_analysis_result(
                        request.symbol, "wyckoff", wyckoff_results
                    )
            
            elif analysis_type == "microstructure":
                # Add microstructure analysis here
                analysis_results["microstructure"] = {"placeholder": "microstructure analysis"}
            
            elif analysis_type == "smc":
                # Add SMC analysis here
                analysis_results["smc"] = {"placeholder": "SMC analysis"}
        
        # Get GPT insights if requested
        gpt_insights = None
        if request.include_gpt and analysis_results:
            gpt_insights = await gpt_router.get_comprehensive_insights(
                request.symbol, analysis_results
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Schedule background cache update
        background_tasks.add_task(
            update_analysis_cache, request.symbol, analysis_results
        )
        
        return AnalysisResponse(
            symbol=request.symbol,
            timestamp=datetime.now().isoformat(),
            analysis_results=analysis_results,
            gpt_insights=gpt_insights,
            cache_hit=cache_hit,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Analysis failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{symbol}")
async def analyze_symbol_get(
    symbol: str = FastAPIPath(..., description="Trading symbol"),
    timeframe: str = Query("5min", description="Timeframe"),
    include_gpt: bool = Query(True, description="Include GPT analysis"),
    use_cache: bool = Query(True, description="Use cached results")
):
    """GET endpoint for symbol analysis"""
    request = AnalysisRequest(
        symbol=symbol,
        timeframe=timeframe,
        analysis_types=["wyckoff", "smc", "microstructure"],
        use_cache=use_cache,
        include_gpt=include_gpt
    )
    
    return await analyze_symbol(request, BackgroundTasks())

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = redis_manager.get_cache_stats()
        return {
            "cache_stats": stats,
            "active_symbols": redis_manager.get_active_symbols(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache/{symbol}")
async def clear_symbol_cache(symbol: str):
    """Clear cache for specific symbol"""
    try:
        pattern = f"*{symbol}*"
        cleared_keys = config.invalidate_cache(pattern)
        return {
            "message": f"Cleared {cleared_keys} cache entries for {symbol}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache")
async def clear_all_cache():
    """Clear all cache"""
    try:
        cleared_keys = config.invalidate_cache("*")
        return {
            "message": f"Cleared {cleared_keys} cache entries",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/json/{symbol}")
async def get_json_analysis(symbol: str):
    """Get JSON analysis for symbol"""
    try:
        # Try Redis first
        cached_json = redis_manager.get_json_analysis(symbol)
        if cached_json:
            return cached_json["data"]
        
        # Try file system
        json_data = load_json_analysis(symbol)
        if json_data:
            # Cache it
            redis_manager.store_json_analysis(symbol, json_data)
            return json_data
        
        raise HTTPException(status_code=404, detail=f"No JSON analysis found for {symbol}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def update_analysis_cache(symbol: str, analysis_results: Dict):
    """Background task to update analysis cache"""
    try:
        for analysis_type, data in analysis_results.items():
            redis_manager.store_analysis_result(symbol, analysis_type, data)
        logger.info(f"Updated cache for {symbol}")
    except Exception as e:
        logger.error(f"Failed to update cache for {symbol}: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("Starting Trading Analytics API...")
    
    # Test Redis connection
    if config.test_redis_connection():
        logger.info("✅ Redis connection successful")
    else:
        logger.warning("⚠️ Redis connection failed")
    
    # Check data directories
    if config.data.json_path.exists():
        logger.info(f"✅ JSON directory found: {config.data.json_path}")
    else:
        logger.warning(f"⚠️ JSON directory not found: {config.data.json_path}")
    
    if config.data.parquet_path.exists():
        logger.info(f"✅ Parquet directory found: {config.data.parquet_path}")
    else:
        logger.warning(f"⚠️ Parquet directory not found: {config.data.parquet_path}")
    
    logger.info("🚀 Trading Analytics API started successfully!")

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info"
    )