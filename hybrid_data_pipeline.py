# Create this file at: /Volumes/tom/Documents/GitHub/zanalytics/hybrid_data_pipeline.py

import os
import pandas as pd
import redis
import json
from datetime import datetime, timedelta

class DataConfig:
    def __init__(self, source_dir=None, redis_url=None, timeframes=None, pair=None):
        self.source_dir = source_dir
        self.redis_url = redis_url
        self.timeframes = timeframes.split(',') if isinstance(timeframes, str) else timeframes
        self.pair = pair

class HybridDataPipeline:
    def __init__(self, config):
        self.config = config
        self.redis_client = redis.from_url(config.redis_url) if config.redis_url else None
        
    def process(self):
        """Process parquet files and store in Redis"""
        if not self.config.source_dir or not os.path.exists(self.config.source_dir):
            print(f"Source directory not found: {self.config.source_dir}")
            return False
            
        for tf in self.config.timeframes:
            # Find parquet files for this timeframe
            tf_dir = os.path.join(self.config.source_dir, tf)
            if not os.path.exists(tf_dir):
                print(f"Timeframe directory not found: {tf_dir}")
                continue
                
            parquet_files = [f for f in os.listdir(tf_dir) if f.endswith('.parquet')]
            if not parquet_files:
                print(f"No parquet files found in {tf_dir}")
                continue
                
            # Process each file
            for pq_file in parquet_files:
                file_path = os.path.join(tf_dir, pq_file)
                try:
                    df = pd.read_parquet(file_path)
                    self._store_in_redis(df, tf)
                    print(f"Processed {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
        return True
    
    def _store_in_redis(self, df, timeframe):
        """Store dataframe in Redis as time series"""
        if not self.redis_client:
            print("Redis client not initialized")
            return
            
        # Create Redis key
        key = f"ts:{self.config.pair}:{timeframe}"
        
        # Convert dataframe to JSON records
        for idx, row in df.iterrows():
            timestamp = int(idx.timestamp() * 1000) if hasattr(idx, 'timestamp') else int(datetime.now().timestamp() * 1000)
            data = row.to_dict()
            data['timestamp'] = timestamp
            
            # Store in Redis sorted set
            self.redis_client.zadd(key, {json.dumps(data): timestamp})
            
        print(f"Stored {len(df)} records in Redis key: {key}")

def get_data_for_llm(config):
    """Retrieve data from Redis for LLM processing"""
    if not config.redis_url:
        return None
        
    redis_client = redis.from_url(config.redis_url)
    
    # Get latest data for each timeframe
    results = {}
    for tf in config.timeframes:
        key = f"ts:{config.pair}:{tf}"
        # Get the latest 50 records
        data = redis_client.zrevrange(key, 0, 49, withscores=True)
        if data:
            results[tf] = [json.loads(item[0]) for item in data]
            
    return results