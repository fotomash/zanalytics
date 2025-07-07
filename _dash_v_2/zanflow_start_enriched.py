#!/usr/bin/env python3
"""
ZANFLOW Startup Script - Port 8080 Version
"""

import subprocess
import time
import sys
import os

def start_service(name, command):
    """Start a service"""
    print(f"🚀 Starting {name}...")
    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux/Mac
            subprocess.Popen(command, shell=True)
        time.sleep(2)
        print(f"✅ {name} started")
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")

def main():
    print("""
    ╔══════════════════════════════════════╗
    ║     ZANFLOW SYSTEM - PORT 8080       ║
    ╚══════════════════════════════════════╝
    """)

    # Start Redis
    print("1️⃣ Starting Redis Server...")
    if os.name == 'nt':
        start_service("Redis", "redis-server")
    else:
        start_service("Redis", "redis-server &")

    # Start API Server on port 8080
    print("\n2️⃣ Starting enriched_api_server Server on port 8080...")
    start_service("API Server", f"{sys.executable} enriched_api_server")

    # Start Dashboard
    print("\n3️⃣ Starting Dashboard...")
    start_service("Dashboard", f"{sys.executable} -m streamlit run pages/home.py")

    print("""
    ✅ All services started!

    📌 Access points:
    - Dashboard: http://localhost:8501
    - API Server: http://localhost:8080
    - MT5 Webhook: http://localhost:8080/webhook

    🔧 MT5 Configuration:
    Add to allowed URLs: 127.0.0.1:8080
    """)

    # Keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()
