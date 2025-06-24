import json
import os
from pathlib import Path

import requests

TRACE_FILE = Path("BTCUSD_2025-06-21_trace.json")
API_ENDPOINT = os.environ.get("TRACE_API_ENDPOINT", "http://localhost:8000/trace")

if not TRACE_FILE.exists():
    print(f"Trace file {TRACE_FILE} not found.")
    raise SystemExit(1)

with open(TRACE_FILE, "r") as f:
    trace_data = json.load(f)

try:
    response = requests.post(API_ENDPOINT, json=trace_data)
    response.raise_for_status()
    print(f"✅ Trace pushed successfully to {API_ENDPOINT}.")
    print(response.json())
except requests.exceptions.RequestException as e:
    print("❌ Failed to push trace:")
    print(e)
