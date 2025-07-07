#!/usr/bin/env python3
"""
ZANFLOW Ngrok Integration - Fixed Version
Handles existing sessions properly
"""

import subprocess
import requests
import json
import time
import os
import sys

def kill_existing_ngrok():
    """Kill any existing ngrok processes"""
    print("🔍 Checking for existing ngrok processes...")

    try:
        # Try to kill using pyngrok first
        from pyngrok import ngrok
        ngrok.kill()
        print("✅ Killed existing pyngrok session")
    except:
        pass

    # Also try system kill
    if os.name == 'posix':  # Mac/Linux
        os.system("pkill -f ngrok")
    else:  # Windows
        os.system("taskkill /F /IM ngrok.exe 2>nul")

    time.sleep(2)

def setup_ngrok_manual(port=8080):
    """Setup ngrok using command line"""
    print(f"🌐 Starting ngrok tunnel on port {port}...")

    # Kill existing sessions first
    kill_existing_ngrok()

    # Start ngrok in background
    if os.name == 'posix':  # Mac/Linux
        cmd = f"ngrok http {port} > /tmp/ngrok.log 2>&1 &"
    else:  # Windows
        cmd = f"start /B ngrok http {port}"

    os.system(cmd)
    print("⏳ Waiting for ngrok to start...")
    time.sleep(4)

    # Get the public URL from ngrok API
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        tunnels = response.json()['tunnels']

        for tunnel in tunnels:
            if tunnel['proto'] == 'https':
                public_url = tunnel['public_url']
                print(f"✅ Ngrok tunnel established!")
                print(f"📍 Public URL: {public_url}")
                print(f"📍 Local URL: http://localhost:{port}")
                print(f"\n⚡ Update your MT5 EA with:")
                print(f"   URL: {public_url}/webhook")
                print(f"   Add to MT5: {public_url.replace('https://', '')}")
                return public_url
    except Exception as e:
        print(f"❌ Error getting tunnel info: {e}")
        print("\n🔧 Try running ngrok manually:")
        print(f"   ngrok http {port}")

    return None

def setup_ngrok_alternative(port=8080):
    """Alternative setup using localtunnel"""
    print("\n🔄 Trying alternative: localtunnel...")

    # Install localtunnel if needed
    try:
        import localtunnel
    except ImportError:
        print("Installing localtunnel...")
        os.system("npm install -g localtunnel")

    print(f"Starting localtunnel on port {port}...")
    os.system(f"lt --port {port}")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════╗
    ║     ZANFLOW Ngrok Setup - Fixed      ║
    ╚══════════════════════════════════════╝
    """)

    # Try manual setup
    public_url = setup_ngrok_manual(8080)

    if not public_url:
        print("\n❌ Ngrok setup failed. Here are your options:\n")

        print("Option 1: Kill ngrok and try again")
        print("   killall ngrok  # Mac/Linux")
        print("   taskkill /F /IM ngrok.exe  # Windows")
        print("   python ngrok_setup_fixed.py")

        print("\nOption 2: Run ngrok manually")
        print("   ngrok http 8080")

        print("\nOption 3: Use alternative tunneling")
        print("   # Install: npm install -g localtunnel")
        print("   lt --port 8080")

        print("\nOption 4: Use your router's port forwarding")
        print("   Forward external port 8080 to internal 192.168.x.x:8080")
