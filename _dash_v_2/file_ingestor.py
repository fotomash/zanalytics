
import asyncio
import os
import time
from datetime import datetime
import pandas as pd
from utils.hybrid_data_pipeline import HybridDataPipeline, DataConfig, get_data_for_llm
import redis.asyncio as redis

# --- Configuration ---
# IMPORTANT: Update this path to the 'tick_data.csv' file inside your MT5 Data Folder
# Example Windows: 'C:/Users/YourUser/AppData/Roaming/MetaQuotes/Terminal/InstanceID/MQL5/Files/tick_data.csv'
# Example Linux/Wine: '/home/user/.wine/drive_c/users/user/Application Data/MetaQuotes/Terminal/InstanceID/MQL5/Files/tick_data.csv'
TICK_DATA_FILE_PATH = "/Users/tom/Documents/_tick_data/_bridge/BTCUSD_M1_bars.csv"

REDIS_HOST = "localhost"
REDIS_PORT = 6379

class FileIngestor:
    def __init__(self, file_path: str, pipeline: HybridDataPipeline, redis_client):
        self.file_path = file_path
        self.pipeline = pipeline
        self.redis_client = redis_client
        self._last_size = 0
        self._file = None
        print(f"File Ingestor initialized. Watching: {self.file_path}")

    async def start_watching(self):
        """Starts the loop to watch the file for new data."""
        print("Waiting for the data file to be created...")
        while not os.path.exists(self.file_path):
            await asyncio.sleep(2)

        print("Data file found. Starting ingestion process.")
        self._file = open(self.file_path, 'r')
        # Skip the header
        self._file.readline()
        # Go to the end of the file
        self._file.seek(0, os.SEEK_END)

        while True:
            try:
                new_lines = self._file.readlines()
                if new_lines:
                    for line in new_lines:
                        if not line.strip():
                            continue
                        await self.process_line(line)
                else:
                    # If no new lines, sleep briefly to avoid busy-waiting
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")
                await asyncio.sleep(5)


    async def process_line(self, line: str):
        """Parses a single CSV line and ingests it into the pipeline."""
        try:
            # Columns: timestamp,bid,ask,last,volume,symbol
            parts = line.strip().split(',')
            if len(parts) != 6:
                # Malformed line, skip
                return

            # Parse timestamp with milliseconds
            dt_obj = datetime.strptime(parts[0], '%Y.%m.%d %H:%M:%S.%f')

            tick_data = {
                "timestamp": dt_obj.isoformat(),
                "bid": float(parts[1]),
                "ask": float(parts[2]),
                "last": float(parts[3]),
                "volume": int(parts[4]),
                "symbol": parts[5],
                "bid_volume": 0, # Not available in this simple format, can be enhanced
                "ask_volume": 0  # Not available in this simple format, can be enhanced
            }

            # Ingest into the hybrid pipeline for minimal real-time enrichment
            enriched_tick = await self.pipeline.ingest_tick(tick_data)

            # Publish the minimally enriched data to Redis for the dashboard/API
            await self.redis_client.publish(f"ticks:{tick_data['symbol']}", str(enriched_tick))

            # Optional: Print to console to verify
            # print(f"Ingested Tick: {enriched_tick['symbol']} @ {enriched_tick['mid_price']:.5f}")

        except (ValueError, IndexError) as e:
            print(f"Could not parse line: '{line.strip()}'. Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during processing: {e}")


async def main():
    # Initialize the core components
    pipeline_config = DataConfig()
    pipeline = HybridDataPipeline(config=pipeline_config)
    redis_client = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)

    # Create and start the file ingestor
    ingestor = FileIngestor(
        file_path=TICK_DATA_FILE_PATH,
        pipeline=pipeline,
        redis_client=redis_client
    )

    # Run the ingestor loop
    await ingestor.start_watching()


if __name__ == "__main__":
    print("Starting File Ingestor Service...")
    # Check if the crucial file path has been updated
    if "tick_data.csv" in TICK_DATA_FILE_PATH and "PLEASE UPDATE" in open(__file__).read():
        print("\n*** WARNING: YOU MUST UPDATE 'TICK_DATA_FILE_PATH' in this script! ***\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("File Ingestor stopped by user.")

