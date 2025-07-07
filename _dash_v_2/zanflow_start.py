#!/usr/bin/env python3
"""
ZANFLOW Unified Startup - Clean Version
This script starts all required services in the correct order
"""

import subprocess
import time
import sys
import os
import signal
from pathlib import Path

class ZanflowStarter:
    def __init__(self):
        self.processes = []
        self.running = True

    def check_service(self, name, check_cmd):
        """Check if a service is running"""
        try:
            result = subprocess.run(check_cmd, shell=True, capture_output=True)
            return result.returncode == 0
        except:
            return False

    def start_redis(self):
        """Start Redis server"""
        print("🔴 Starting Redis...")

        # Check if already running
        if self.check_service("Redis", "redis-cli ping"):
            print("✅ Redis already running")
            return True

        # Start Redis
        process = subprocess.Popen(
            ["redis-server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.processes.append(("Redis", process))

        # Wait for Redis to start
        for i in range(10):
            time.sleep(1)
            if self.check_service("Redis", "redis-cli ping"):
                print("✅ Redis started successfully")
                return True

        print("❌ Failed to start Redis")
        return False

    def start_api(self):
        """Start Flask API server"""
        print("\n🌐 Starting API Server...")

        api_path = Path("api_server.py")
        if not api_path.exists():
            print("❌ api_server.py not found!")
            return False

        process = subprocess.Popen(
            [sys.executable, "api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(("API Server", process))

        time.sleep(3)
        print("✅ API Server started on http://localhost:8080")
        return True

    def start_dashboard(self):
        """Start Streamlit dashboard"""
        print("\n📊 Starting Dashboard...")

        dashboard_path = Path("main.py")
        if not dashboard_path.exists():
            print("❌ main.py not found!")
            return False

        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(("Dashboard", process))

        time.sleep(3)
        print("✅ Dashboard started on http://localhost:8501")
        return True

    def signal_handler(self, signum, frame):
        """Handle shutdown gracefully"""
        print("\n\n⏹️  Shutting down ZANFLOW...")
        self.running = False

        for name, process in self.processes:
            print(f"  Stopping {name}...")
            process.terminate()

        # Give processes time to shutdown
        time.sleep(2)

        # Force kill if needed
        for name, process in self.processes:
            if process.poll() is None:
                process.kill()

        print("✅ All services stopped")
        sys.exit(0)

    def run(self):
        """Main startup sequence"""
        print("🌊 ZANFLOW Trading System v2")
        print("=" * 50)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Check current directory
        if not Path("api_server.py").exists():
            print("❌ Error: Please run this script from the _dash_v_2 directory")
            print("   cd zanalytics-main/_dash_v_2")
            print("   python zanflow_start.py")
            return

        # Start services in order
        if not self.start_redis():
            return

        if not self.start_api():
            return

        if not self.start_dashboard():
            return

        print("\n" + "=" * 50)
        print("✅ ZANFLOW is running!")
        print("\n📍 Access Points:")
        print("   Dashboard: http://localhost:8501")
        print("   API: http://localhost:5000")
        print("   Redis: localhost:6379")
        print("\n⚡ MetaTrader Connection:")
        print("   URL: http://localhost:5000/webhook")
        print("\nPress Ctrl+C to stop all services")
        print("=" * 50)

        # Keep running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.signal_handler(None, None)

if __name__ == "__main__":
    starter = ZanflowStarter()
    starter.run()
