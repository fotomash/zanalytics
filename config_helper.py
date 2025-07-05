#!/usr/bin/env python3
"""
Configuration Helper for Trading Analytics API
Manages environment variables, Redis, and API configurations
"""

import os
import json
import redis
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

@dataclass
class APIConfig:
    """API Configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    title: str = "Trading Analytics API v4.0"
    description: str = "Professional Wyckoff, SMC, and Microstructure Analysis"
    version: str = "4.0.0"
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8501"]

@dataclass
class RedisConfig:
    """Redis Configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    decode_responses: bool = True
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    socket_keepalive: bool = True
    socket_keepalive_options: Dict = None
    
    def __post_init__(self):
        if self.socket_keepalive_options is None:
            self.socket_keepalive_options = {}

@dataclass
class DataConfig:
    """Data Configuration"""
    base_dir: str = "/Users/tom/Documents/_trade/_exports/_tick"
    parquet_dir: str = "parquet"
    raw_dir: str = "_raw"
    json_dir: str = "midas_analysis"
    bar_dir: str = "_bars"
    enriched_dir: str = "parquet"
    
    @property
    def parquet_path(self) -> Path:
        return Path(self.base_dir) / self.parquet_dir
    
    @property
    def json_path(self) -> Path:
        return Path(self.base_dir) / self.json_dir
    
    @property
    def raw_path(self) -> Path:
        return Path(self.base_dir) / self.raw_dir

@dataclass
class GPTConfig:
    """GPT Configuration"""
    api_key: str = ""
    model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30

class ConfigHelper:
    """Central configuration management"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or ".env"
        self.redis_client = None
        self._load_config()
        
    def _load_config(self):
        """Load configuration from environment and files"""
        # Load from .env file if exists
        env_path = Path(self.config_file)
        if env_path.exists():
            self._load_env_file(env_path)
        
        # Initialize configurations
        self.api = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "false").lower() == "true"
        )
        
        self.redis = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD")
        )
        
        self.data = DataConfig(
            base_dir=os.getenv("DATA_BASE_DIR", "/Users/tom/Documents/_trade/_exports/_tick"),
            json_dir=os.getenv("JSON_DIR", "midas_analysis")
        )
        
        self.gpt = GPTConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("GPT_MODEL", "gpt-4"),
            max_tokens=int(os.getenv("GPT_MAX_TOKENS", "4096"))
        )
        
    def _load_env_file(self, env_path: Path):
        """Load environment variables from file"""
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    def get_redis_client(self) -> redis.Redis:
        """Get Redis client with connection pooling"""
        if self.redis_client is None:
            self.redis_client = redis.Redis(
                host=self.redis.host,
                port=self.redis.port,
                db=self.redis.db,
                password=self.redis.password,
                decode_responses=self.redis.decode_responses,
                socket_timeout=self.redis.socket_timeout,
                socket_connect_timeout=self.redis.socket_connect_timeout,
                socket_keepalive=self.redis.socket_keepalive,
                socket_keepalive_options=self.redis.socket_keepalive_options,
                health_check_interval=30
            )
        return self.redis_client
    
    def test_redis_connection(self) -> bool:
        """Test Redis connection"""
        try:
            client = self.get_redis_client()
            client.ping()
            return True
        except Exception as e:
            logging.error(f"Redis connection failed: {e}")
            return False
    
    def cache_analysis_result(self, key: str, data: Dict, ttl: int = 3600):
        """Cache analysis result in Redis"""
        try:
            client = self.get_redis_client()
            client.setex(key, ttl, json.dumps(data, default=str))
            return True
        except Exception as e:
            logging.error(f"Failed to cache result: {e}")
            return False
    
    def get_cached_result(self, key: str) -> Optional[Dict]:
        """Get cached analysis result"""
        try:
            client = self.get_redis_client()
            result = client.get(key)
            if result:
                return json.loads(result)
        except Exception as e:
            logging.error(f"Failed to get cached result: {e}")
        return None
    
    def invalidate_cache(self, pattern: str = "*"):
        """Invalidate cache entries matching pattern"""
        try:
            client = self.get_redis_client()
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)
                return len(keys)
        except Exception as e:
            logging.error(f"Failed to invalidate cache: {e}")
        return 0
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug
            },
            "redis": {
                "connected": self.test_redis_connection(),
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db
            },
            "data": {
                "base_dir": str(self.data.base_dir),
                "json_dir_exists": self.data.json_path.exists(),
                "parquet_dir_exists": self.data.parquet_path.exists()
            },
            "gpt": {
                "configured": bool(self.gpt.api_key),
                "model": self.gpt.model
            }
        }

# Global configuration instance
config = ConfigHelper()