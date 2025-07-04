# redis_writer.py
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import redis.asyncio as redis
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class EnrichedRow:
    """Schema-compliant enriched market data row"""
    timestamp: str
    symbol: str
    timeframe: str
    
    # OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Enriched fields
    confluence_score: float
    trade_quality: str
    wyckoff_phase: str
    spring_event: bool
    smc_zone_type: str
    trade_entry: bool
    
    # Meta
    run_id: str
    annotated: bool
    review_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_redis_key(self) -> str:
        return f"tick:{self.symbol}:{self.timeframe}"

class RedisSnapshotWriter:
    """Manages Redis hot storage with automated Parquet snapshots"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 snapshot_dir: str = "./snapshots"):
        self.redis_url = redis_url
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.redis_client: Optional[redis.Redis] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        self.logger.info("Redis connection established")
        
    async def write_enriched_row(self, row: EnrichedRow, ttl: int = 3600):
        """Write enriched row to Redis with TTL"""
        if not self.redis_client:
            await self.initialize()
            
        key = row.get_redis_key()
        data = json.dumps(row.to_dict())
        
        # Write to Redis with TTL
        await self.redis_client.setex(key, ttl, data)
        
        # Also add to time-series for historical access
        ts_key = f"ts:{row.symbol}:{row.timeframe}"
        await self.redis_client.zadd(ts_key, {data: datetime.fromisoformat(row.timestamp).timestamp()})
        
        # Cleanup old entries (keep last 1000)
        await self.redis_client.zremrangebyrank(ts_key, 0, -1001)
        
    async def get_latest_data(self, symbol: str, timeframe: str, count: int = 100) -> List[Dict]:
        """Get latest enriched data for symbol"""
        if not self.redis_client:
            await self.initialize()
            
        ts_key = f"ts:{symbol}:{timeframe}"
        raw_data = await self.redis_client.zrevrange(ts_key, 0, count-1)
        
        return [json.loads(item) for item in raw_data]
    
    async def create_snapshot(self, symbol: Optional[str] = None) -> str:
        """Create Parquet snapshot from Redis data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if symbol:
            # Snapshot specific symbol
            pattern = f"ts:{symbol}:*"
            snapshot_file = self.snapshot_dir / f"{symbol}_{timestamp}.parquet"
        else:
            # Snapshot all data
            pattern = "ts:*"
            snapshot_file = self.snapshot_dir / f"full_snapshot_{timestamp}.parquet"
            
        keys = await self.redis_client.keys(pattern)
        all_data = []
        
        for key in keys:
            raw_items = await self.redis_client.zrevrange(key, 0, -1)
            for item in raw_items:
                try:
                    data = json.loads(item)
                    all_data.append(data)
                except json.JSONDecodeError:
                    continue
                    
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_parquet(snapshot_file, index=False)
            
            # Generate checksum for integrity
            checksum = hashlib.md5(snapshot_file.read_bytes()).hexdigest()
            checksum_file = snapshot_file.with_suffix('.md5')
            checksum_file.write_text(checksum)
            
            self.logger.info(f"Snapshot created: {snapshot_file} ({len(all_data)} rows)")
            return str(snapshot_file)
        
        return ""
    
    async def rotate_snapshots(self, max_snapshots: int = 24):
        """Rotate old snapshots (keep last N)"""
        snapshot_files = sorted(
            self.snapshot_dir.glob("*.parquet"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if len(snapshot_files) > max_snapshots:
            for old_file in snapshot_files[max_snapshots:]:
                old_file.unlink(missing_ok=True)
                # Also remove checksum
                old_file.with_suffix('.md5').unlink(missing_ok=True)
                self.logger.info(f"Rotated snapshot: {old_file}")
    
    async def auto_snapshot_scheduler(self, interval_hours: int = 1):
        """Background task for automatic snapshots"""
        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)
                await self.create_snapshot()
                await self.rotate_snapshots()
            except Exception as e:
                self.logger.error(f"Snapshot scheduler error: {e}")