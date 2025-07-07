#!/usr/bin/env python3
"""
ZANFLOW Ngrok Integration
Exposes your local API to the internet
"""

import subprocess
import requests
import json
import time
from pyngrok import ngrok

def setup_ngrok(port=8080):
    """Setup ngrok tunnel"""
    print(f"🌐 Starting ngrok tunnel on port {port}...")
    # Start ngrok
    public_url = ngrok.connect(port, "http")
    print(f"✅ Ngrok tunnel established!")
    print(f"📍 Public URL: {public_url}")
    print(f"📍 Local URL: http://localhost:{port}")

    # Get tunnel info
    tunnels = ngrok.get_tunnels()
    for tunnel in tunnels:
        print(f"🔗 Tunnel: {tunnel.name}")
        print(f"   Public: {tunnel.public_url}")
        print(f"   Local: {tunnel.config['addr']}")

    print(f" Update your MT5 EA with:")
    print(f"   URL: {public_url}/webhook")
    print(f"   Add to MT5 allowed URLs: {public_url.replace('https://', '').replace('http://', '')}")

    return public_url

def monitor_tunnel():
    """Monitor ngrok tunnel status"""
    while True:
        try:
            tunnels = ngrok.get_tunnels()
            if tunnels:
                print(f"✅ Tunnel active: {tunnels[0].public_url}")
            else:
                print("❌ No active tunnels")
        except Exception as e:
            print(f"❌ Error: {e}")

        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    # Install ngrok if needed
    try:
        import pyngrok
    except ImportError:
        print("Installing pyngrok...")
        subprocess.run(["pip", "install", "pyngrok"])

    # Setup tunnel
    public_url = setup_ngrok(8080)

    # Keep running
    try:
        monitor_tunnel()
    except KeyboardInterrupt:
        print("👋 Closing ngrok tunnel...")
        ngrok.disconnect(public_url)
        ngrok.kill()
