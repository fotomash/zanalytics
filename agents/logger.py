# logger.py - Fixed version with automatic directory creation
import os
import json
import logging.config
from pathlib import Path

def setup_logging():
    """Setup logging with automatic directory creation"""
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Load logging config
    config_path = Path("config/logging.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        # Fallback configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/zanalytics.log'),
                logging.StreamHandler()
            ]
        )
    
    return logging.getLogger(__name__)