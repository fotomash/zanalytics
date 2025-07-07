# mt5_file_watcher.py
import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MT5FileHandler(FileSystemEventHandler):
    def __init__(self, api_url="http://localhost:8000/api/v1/ingest/candle"):
        self.api_url = api_url
        self.processed_files = set()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.csv') and 'candles' in event.src_path:
            asyncio.create_task(self.process_csv_file(event.src_path))
    
    async def process_csv_file(self, filepath):
        try:
            # Wait a bit for file to be fully written
            await asyncio.sleep(1)
            
            df = pd.read_csv(filepath)
            if len(df) == 0:
                return
            
            # Process last row (most recent data)
            last_row = df.iloc[-1]
            
            # Create API payload
            payload = {
                "candle": {
                    "symbol": last_row['symbol'],
                    "timeframe": last_row['timeframe'],
                    "timestamp": last_row['timestamp'],
                    "open": float(last_row['open']),
                    "high": float(last_row['high']),
                    "low": float(last_row['low']),
                    "close": float(last_row['close']),
                    "volume": int(last_row['volume'])
                },
                "signals": {
                    "wyckoff_phase": "unknown",
                    "spring_event": False,
                    "smc_zone_type": "none",
                    "confluence_score": 0.0,
                    "trade_entry": False
                }
            }
            
            # Send to API
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Processed {last_row['symbol']} - Confluence: {result['enriched_data']['confluence_score']}")
                    else:
                        logger.error(f"API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")

async def main():
    # Monitor MT5 export directory
    export_path = "C:/MT5_Data/Exports/"
    
    event_handler = MT5FileHandler()
    observer = Observer()
    observer.schedule(event_handler, export_path, recursive=True)
    observer.start()
    
    logger.info(f"Monitoring {export_path} for MT5 data...")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    asyncio.run(main())