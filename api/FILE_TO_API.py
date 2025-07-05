from fastapi import FastAPI, Query
import pandas as pd
from pathlib import Path
from typing import List, Optional

app = FastAPI()

@app.get("/api/v1/enriched/{symbol}")
async def get_enriched_data(
    symbol: str,
    timeframe: str = "H1",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    # Path to enriched CSV files
    base_path = Path("C:/MT5_Data/Exports/")
    file_path = base_path / f"{symbol}_{timeframe}_enriched.csv"
    
    if not file_path.exists():
        return {"error": f"No enriched data found for {symbol} {timeframe}"}
    
    # Load the enriched data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Apply time range filter if provided
    if start_date and end_date:
        start_datetime = pd.to_datetime(f"{start_date} {start_time or '00:00:00'}")
        end_datetime = pd.to_datetime(f"{end_date} {end_time or '23:59:59'}")
        df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]
    
    # Return as JSON
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data_points": len(df),
        "data": df.to_dict(orient="records")
    }