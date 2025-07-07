from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
from datetime import datetime
from typing import Dict, Any

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Redis connection
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

@app.post("/data")
async def receive_enriched_data(data: Dict[str, Any]):
    """Receive enriched data from MT5"""
    try:
        symbol = data.get('symbol')
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol required")

        # Add server timestamp
        data['server_timestamp'] = datetime.now().isoformat()

        # Store in Redis with TTL
        key = f"live:{symbol}"
        r.setex(key, 60, json.dumps(data))  # 60 second TTL

        # Also store in time series
        ts_key = f"timeseries:{symbol}"
        r.zadd(ts_key, {json.dumps(data): data['timestamp']})

        # Trim old data (keep last 1000)
        r.zremrangebyrank(ts_key, 0, -1001)

        return {"status": "success", "symbol": symbol}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/{symbol}")
async def get_enriched_data(symbol: str):
    """Get latest enriched data for symbol"""
    try:
        key = f"live:{symbol}"
        data = r.get(key)

        if not data:
            raise HTTPException(status_code=404, detail="No data for symbol")

        return json.loads(data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_symbols():
    """Get all symbols with live data"""
    try:
        keys = r.keys("live:*")
        symbols = [key.replace("live:", "") for key in keys]
        return symbols

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
