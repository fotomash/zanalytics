"""
Production configuration for Zanalytics
"""

import os
from typing import Dict, Any

class ProductionConfig:
    """Production environment configuration"""

    # Application settings
    APP_NAME = "Zanalytics"
    VERSION = "2.0.0"
    ENVIRONMENT = "production"
    DEBUG = False
    TESTING = False

    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'change-this-in-production')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'change-this-in-production')
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = 86400 * 30  # 30 days

    # Database settings
    DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:pass@localhost/zanalytics')
    DATABASE_POOL_SIZE = 20
    DATABASE_MAX_OVERFLOW = 40
    DATABASE_POOL_TIMEOUT = 30

    # Redis settings
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    REDIS_MAX_CONNECTIONS = 50

    # API settings
    API_RATE_LIMIT = "100/minute"
    API_TIMEOUT = 30
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "/app/logs/zanalytics.log"
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5

    # Performance settings
    CACHE_TTL = 3600  # 1 hour
    QUERY_TIMEOUT = 60  # seconds
    MAX_WORKERS = 4

    # Monitoring settings
    ENABLE_METRICS = True
    METRICS_PORT = 9090
    HEALTH_CHECK_INTERVAL = 30

    # Feature flags
    FEATURES = {
        'advanced_analytics': True,
        'real_time_processing': True,
        'ml_insights': True,
        'export_functionality': True
    }

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }
