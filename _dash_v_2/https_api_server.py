
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import uvicorn
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Secure Market Data Ingestion API",
    description="An API to receive enriched market data from MT5 via a secure tunnel.",
    version="1.0.0"
)

# --- Pydantic Model for the incoming data ---
# This model is based on the structure of your DXY.CASH_comprehensive.json file.
# It ensures that the data we receive is valid.
class MarketBar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    # Use Field(..., alias=...) to handle keys that are not valid Python identifiers
    # Or just use a flexible Dict for all the indicator data
    indicators: Dict[str, Any] = Field(alias="indicators")

# --- API Endpoint ---
@app.post("/dxy_bar")
async def receive_dxy_bar(bar_data: Dict = Body(...)):
    """
    Receives a single, enriched 1-minute bar for DXY.

    This endpoint is designed to be called by the MT5 EA via an ngrok tunnel.
    """
    try:
        # You can add any processing logic here.
        # For now, we will just print it to confirm receipt and save it to a file.

        symbol = "DXY.CASH"
        timestamp_str = bar_data.get("timestamp", datetime.now().isoformat())
        close_price = bar_data.get("close", 0)

        print(f"✅ Received data for {symbol} @ {timestamp_str} - Close: {close_price}")

        # Save the received data to a log file for inspection
        with open("received_dxy_data.log", "a") as f:
            f.write(json.dumps(bar_data, indent=4))
            f.write("\n---\n")

        # Here you would typically:
        # 1. Store this data in your database (e.g., Redis, Parquet, SQL).
        # 2. Trigger analysis or alerts.
        # 3. Push updates to the Streamlit dashboard.

        return {
            "status": "success",
            "message": f"Data for {symbol} at {timestamp_str} received successfully."
        }

    except Exception as e:
        print(f"❌ Error processing incoming data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Secure Market Data API is running. Use the /dxy_bar endpoint to POST data."}

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting Secure API Server on http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

