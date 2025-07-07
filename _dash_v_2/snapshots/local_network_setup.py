#!/usr/bin/env python3
"""
Local Network Access for ZANFLOW
Access from other computers on same network
"""

import socket
import subprocess
import platform

def get_local_ip():
    """Get local IP address"""
    try:
        # Create a socket to external site
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def setup_local_access():
    """Setup for local network access"""
    local_ip = get_local_ip()

    print(f"""
    🏠 Local Network Access Setup
    =============================

    Your API is accessible on your local network at:

    📍 http://{local_ip}:8080
    📍 http://{local_ip}:8080/webhook

    From other computers on same network:
    1. Make sure they're on same WiFi/Network
    2. Use the IP above in MT5: {local_ip}:8080
    3. No ngrok needed!

    ⚠️  Firewall Settings:
    """)

    if platform.system() == "Darwin":  # Mac
        print("   Mac: System Preferences → Security → Firewall → Allow incoming")
    elif platform.system() == "Windows":
        print("   Windows: Allow Python through Windows Firewall")
        print("   Run: netsh advfirewall firewall add rule name=\"ZANFLOW\" dir=in action=allow protocol=TCP localport=8080")
    else:  # Linux
        print("   Linux: sudo ufw allow 8080/tcp")

    print(f"""

    📱 Access from Phone/Tablet:
    1. Connect to same WiFi
    2. Open browser to: http://{local_ip}:8080

    🖥️  Multiple MT5 Terminals:
    Can all connect to: {local_ip}:8080
    """)

if __name__ == "__main__":
    setup_local_access()
