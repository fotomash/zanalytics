import requests
import json
import sys
from pathlib import Path

# Path to your saved trace file
trace_file = Path("BTCUSD_2025-06-21_trace.json")  # Or use sys.argv[1] for dynamic CLI use

if not trace_file.exists():
    print(f"Trace file {trace_file} not found.")
    sys.exit(1)

with open(trace_file, "r") as f:
    trace_data = json.load(f)

url = "https://emerging-tiger-fair.ngrok-free.app/upload-trace"

try:
    response = requests.post(url, json=trace_data)
    response.raise_for_status()
    print("✅ Trace pushed successfully.")
    print(response.json())
except Exception as e:
    print("❌ Failed to push trace:")
    print(e)

