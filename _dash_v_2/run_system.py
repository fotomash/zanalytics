#!/usr/bin/env python3
"""
System Startup Script
Runs the complete trading analytics system
"""

import subprocess
import sys
import time
import logging
from config_helper from config import orchestrator_config

def start_redis():
    """Start Redis server"""
    try:
        # Test if Redis is already running
        if config.test_redis_connection():
            print("✅ Redis already running")
            return True
        
        # Start Redis server
        print("🔄 Starting Redis server...")
        subprocess.Popen(["redis-server"], stdout=subprocess.DEVNULL)
        
        # Wait for Redis to start
        for i in range(10):
            time.sleep(1)
            if config.test_redis_connection():
                print("✅ Redis started successfully")
                return True
        
        print("❌ Failed to start Redis")
        return False
        
    except Exception as e:
        print(f"❌ Redis startup error: {e}")
        return False

def start_api():
    """Start API server"""
    try:
        print("🚀 Starting API server...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api_server:app", 
            "--host", config.api.host,
            "--port", str(config.api.port),
            "--reload" if config.api.debug else "--no-reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down API server...")
    except Exception as e:
        print(f"❌ API startup error: {e}")

def main():
    print("🏁 Starting Trading Analytics System...")
    
    # Start Redis
    if not start_redis():
        print("❌ Cannot continue without Redis")
        sys.exit(1)
    
    # Start API
    start_api()

if __name__ == "__main__":
    main()