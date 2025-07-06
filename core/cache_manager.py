"""Intelligent Caching System for Zanalytics
Implements multi-level caching with TTL and invalidation.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Callable
from functools import wraps

import redis
from diskcache import Cache


class CacheManager:
    """Multi-level caching system with memory, disk, and Redis support."""

    def __init__(self,
                 cache_dir: str = ".cache",
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl

        # Disk cache
        self.disk_cache = Cache(cache_dir)

        # Redis cache
        try:
            self.redis_client = redis.Redis(host=redis_host,
                                            port=redis_port,
                                            decode_responses=True)
            self.redis_client.ping()
            self.redis_available = True
        except Exception:
            self.redis_client = None
            self.redis_available = False

        # In-memory cache
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

    def _generate_key(self, namespace: str, *args, **kwargs) -> str:
        """Generate a cache key from namespace and arguments."""
        key_data = {"namespace": namespace, "args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, namespace: str, *args, **kwargs) -> Optional[Any]:
        """Retrieve a value from cache."""
        key = self._generate_key(namespace, *args, **kwargs)

        # Memory
        entry = self.memory_cache.get(key)
        if entry and datetime.now() < entry["expires"]:
            return entry["value"]
        if entry:
            self.memory_cache.pop(key, None)

        # Redis
        if self.redis_available:
            try:
                value = self.redis_client.get(f"{namespace}:{key}")
                if value is not None:
                    return json.loads(value)
            except Exception:
                pass

        # Disk
        try:
            return self.disk_cache.get(key)
        except Exception:
            return None

    def set(self, namespace: str, value: Any, ttl: Optional[int] = None,
            *args, **kwargs) -> None:
        """Store a value in the cache."""
        key = self._generate_key(namespace, *args, **kwargs)
        ttl = ttl or self.default_ttl
        expires = datetime.now() + timedelta(seconds=ttl)

        # Memory
        self.memory_cache[key] = {"value": value, "expires": expires}

        # Redis
        if self.redis_available:
            try:
                self.redis_client.setex(f"{namespace}:{key}", ttl, json.dumps(value))
            except Exception:
                pass

        # Disk
        try:
            self.disk_cache.set(key, value, expire=ttl)
        except Exception:
            pass

    def invalidate(self, namespace: str, *args, **kwargs) -> None:
        """Invalidate a specific cache entry."""
        key = self._generate_key(namespace, *args, **kwargs)
        self.memory_cache.pop(key, None)
        if self.redis_available:
            try:
                self.redis_client.delete(f"{namespace}:{key}")
            except Exception:
                pass
        try:
            self.disk_cache.delete(key)
        except Exception:
            pass

    def invalidate_namespace(self, namespace: str) -> None:
        """Invalidate all entries in a namespace."""
        # Memory
        for key in list(self.memory_cache.keys()):
            if key.startswith(self._generate_key(namespace)):
                self.memory_cache.pop(key, None)

        # Redis
        if self.redis_available:
            try:
                for key in self.redis_client.scan_iter(f"{namespace}:*"):
                    self.redis_client.delete(key)
            except Exception:
                pass

        # Disk
        try:
            for key in list(self.disk_cache.iterkeys()):
                if str(key).startswith(namespace):
                    self.disk_cache.delete(key)
        except Exception:
            pass

    def clear_all(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        if self.redis_available:
            try:
                self.redis_client.flushall()
            except Exception:
                pass
        try:
            self.disk_cache.clear()
        except Exception:
            pass


def cached(namespace: str, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "_cache_manager"):
                wrapper._cache_manager = CacheManager()
            cache_manager: CacheManager = wrapper._cache_manager
            result = cache_manager.get(namespace, *args, **kwargs)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache_manager.set(namespace, result, ttl, *args, **kwargs)
            return result

        wrapper.invalidate = lambda *a, **kw: wrapper._cache_manager.invalidate(namespace, *a, **kw)
        wrapper.invalidate_namespace = lambda: wrapper._cache_manager.invalidate_namespace(namespace)
        wrapper.clear_all = lambda: wrapper._cache_manager.clear_all()
        return wrapper

    return decorator

