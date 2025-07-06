#!/usr/bin/env python3
"""
Redis Server Management for Trading Analytics
Handles caching, real-time data, and session management
"""

import json
import redis
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from config_helper from config import orchestrator_config

@dataclass
class CacheStats:
    """Cache statistics"""
    total_keys: int
    memory_usage: str
    hit_ratio: float
    analysis_cache_keys: int
    json_cache_keys: int
    uptime: str

class TradingRedisManager:
    """Enhanced Redis manager for trading analytics"""
    
    def __init__(self):
        self.redis_client = config.get_redis_client()
        self.logger = logging.getLogger(__name__)
        
        # Cache key prefixes
        self.ANALYSIS_PREFIX = "analysis:"
        self.JSON_PREFIX = "json:"
        self.SYMBOL_PREFIX = "symbol:"
        self.SESSION_PREFIX = "session:"
        self.STATS_PREFIX = "stats:"
        
    def store_analysis_result(self, symbol: str, analysis_type: str, data: Dict, ttl: int = 3600) -> bool:
        """Store analysis result with automatic expiration"""
        key = f"{self.ANALYSIS_PREFIX}{symbol}:{analysis_type}:{datetime.now().strftime('%Y%m%d_%H')}"
        
        try:
            # Add metadata
            enriched_data = {
                "symbol": symbol,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "ttl": ttl
            }
            
            # Store with expiration
            self.redis_client.setex(key, ttl, json.dumps(enriched_data, default=str))
            
            # Update symbol index
            self._update_symbol_index(symbol, key)
            
            self.logger.info(f"Stored analysis result: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store analysis result: {e}")
            return False
    
    def get_analysis_result(self, symbol: str, analysis_type: str, hours_back: int = 24) -> Optional[Dict]:
        """Get most recent analysis result for symbol and type"""
        try:
            # Search for recent results
            pattern = f"{self.ANALYSIS_PREFIX}{symbol}:{analysis_type}:*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return None
            
            # Get most recent
            keys.sort(reverse=True)
            result = self.redis_client.get(keys[0])
            
            if result:
                return json.loads(result)
                
        except Exception as e:
            self.logger.error(f"Failed to get analysis result: {e}")
        
        return None
    
    def store_json_analysis(self, symbol: str, json_data: Dict, source: str = "auto") -> bool:
        """Store JSON analysis data"""
        key = f"{self.JSON_PREFIX}{symbol}:{source}"
        
        try:
            enriched_data = {
                "symbol": symbol,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "data": json_data
            }
            
            # Store JSON data (longer TTL)
            self.redis_client.setex(key, 7200, json.dumps(enriched_data, default=str))
            
            # Update symbol index
            self._update_symbol_index(symbol, key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store JSON analysis: {e}")
            return False
    
    def get_json_analysis(self, symbol: str, source: str = "auto") -> Optional[Dict]:
        """Get JSON analysis data"""
        key = f"{self.JSON_PREFIX}{symbol}:{source}"
        
        try:
            result = self.redis_client.get(key)
            if result:
                return json.loads(result)
        except Exception as e:
            self.logger.error(f"Failed to get JSON analysis: {e}")
        
        return None
    
    def get_symbol_data(self, symbol: str) -> Dict:
        """Get all cached data for a symbol"""
        symbol_data = {
            "symbol": symbol,
            "analysis_results": [],
            "json_data": [],
            "last_updated": None
        }
        
        try:
            # Get symbol index
            index_key = f"{self.SYMBOL_PREFIX}{symbol}:index"
            keys = self.redis_client.smembers(index_key)
            
            latest_timestamp = None
            
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    parsed_data = json.loads(data)
                    
                    if key.startswith(self.ANALYSIS_PREFIX):
                        symbol_data["analysis_results"].append(parsed_data)
                    elif key.startswith(self.JSON_PREFIX):
                        symbol_data["json_data"].append(parsed_data)
                    
                    # Track latest timestamp
                    timestamp = parsed_data.get("timestamp")
                    if timestamp and (not latest_timestamp or timestamp > latest_timestamp):
                        latest_timestamp = timestamp
            
            symbol_data["last_updated"] = latest_timestamp
            
        except Exception as e:
            self.logger.error(f"Failed to get symbol data: {e}")
        
        return symbol_data
    
    def _update_symbol_index(self, symbol: str, key: str):
        """Update symbol index for efficient lookups"""
        index_key = f"{self.SYMBOL_PREFIX}{symbol}:index"
        try:
            self.redis_client.sadd(index_key, key)
            # Set expiration on index (longer than individual keys)
            self.redis_client.expire(index_key, 86400)  # 24 hours
        except Exception as e:
            self.logger.error(f"Failed to update symbol index: {e}")
    
    def get_active_symbols(self) -> List[str]:
        """Get list of symbols with cached data"""
        try:
            pattern = f"{self.SYMBOL_PREFIX}*:index"
            keys = self.redis_client.keys(pattern)
            
            symbols = []
            for key in keys:
                # Extract symbol from key
                symbol = key.replace(f"{self.SYMBOL_PREFIX}", "").replace(":index", "")
                symbols.append(symbol)
            
            return sorted(symbols)
            
        except Exception as e:
            self.logger.error(f"Failed to get active symbols: {e}")
            return []
    
    def create_user_session(self, user_id: str, session_data: Dict) -> str:
        """Create user session"""
        session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        key = f"{self.SESSION_PREFIX}{session_id}"
        
        try:
            session_info = {
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "data": session_data
            }
            
            # Store session (4 hour TTL)
            self.redis_client.setex(key, 14400, json.dumps(session_info, default=str))
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            return ""
    
    def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        try:
            info = self.redis_client.info()
            
            # Count keys by type
            analysis_keys = len(self.redis_client.keys(f"{self.ANALYSIS_PREFIX}*"))
            json_keys = len(self.redis_client.keys(f"{self.JSON_PREFIX}*"))
            total_keys = info.get("db0", {}).get("keys", 0)
            
            # Memory usage
            memory_usage = info.get("used_memory_human", "Unknown")
            
            # Uptime
            uptime_seconds = info.get("uptime_in_seconds", 0)
            uptime = str(timedelta(seconds=uptime_seconds))
            
            return CacheStats(
                total_keys=total_keys,
                memory_usage=memory_usage,
                hit_ratio=0.0,  # Would need to track this separately
                analysis_cache_keys=analysis_keys,
                json_cache_keys=json_keys,
                uptime=uptime
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return CacheStats(0, "Unknown", 0.0, 0, 0, "Unknown")
    
    def cleanup_expired_data(self) -> int:
        """Manual cleanup of expired data"""
        cleaned = 0
        try:
            # Get all keys with TTL
            for pattern in [f"{self.ANALYSIS_PREFIX}*", f"{self.JSON_PREFIX}*", f"{self.SESSION_PREFIX}*"]:
                keys = self.redis_client.keys(pattern)
                for key in keys:
                    ttl = self.redis_client.ttl(key)
                    if ttl == -1:  # No expiration set
                        # Set default expiration based on key type
                        if self.ANALYSIS_PREFIX in key:
                            self.redis_client.expire(key, 3600)
                        elif self.JSON_PREFIX in key:
                            self.redis_client.expire(key, 7200)
                        elif self.SESSION_PREFIX in key:
                            self.redis_client.expire(key, 14400)
                        cleaned += 1
                        
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
        
        return cleaned

# Global Redis manager instance
redis_manager = TradingRedisManager()