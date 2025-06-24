#!/usr/bin/env python3
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="NCOS v21 Phoenix Mesh API", version="21.7")

@app.get("/")
async def root():
    return {"name": "NCOS v21 Phoenix Mesh API", "status": "operational"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
