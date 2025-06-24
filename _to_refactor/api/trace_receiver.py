from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import datetime
import json

app = FastAPI()
TRACE_DIR = "received_traces"
os.makedirs(TRACE_DIR, exist_ok=True)

@app.post("/upload-trace")
async def upload_trace(request: Request):
    try:
        trace_data = await request.json()
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{TRACE_DIR}/trace_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(trace_data, f, indent=2)

        return JSONResponse(content={"status": "success", "saved_to": filename}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)