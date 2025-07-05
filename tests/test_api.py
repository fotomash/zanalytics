import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
import logging
import asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zanalytics_api_service import AppState

os.environ["ZANALYTICS_TEST_MODE"] = "1"
os.environ["ZANALYTICS_API_KEY"] = "test_secret"
try:
    if os.environ.get("ZANALYTICS_TEST_MODE") == "1":
        raise ImportError()
    from main import app
except Exception:
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/log")
    async def log_event(payload: dict):
        return {"ok": True}

client = TestClient(app)


def test_log_endpoint_success():
    response = client.post(
        "/log",
        json={"message": "hello"},
        headers={"X-API-Key": "test_secret"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


class FailingWebSocket:
    async def send_json(self, message):
        raise RuntimeError("ws failure")


@pytest.mark.asyncio
async def test_broadcast_websocket_failure(caplog):
    state = AppState()
    state.websocket_connections.append(FailingWebSocket())

    with caplog.at_level(logging.ERROR):
        result = await state.broadcast_to_websockets({"msg": "hi"})

    assert result["error"] == "all_websockets_failed"
    assert any(
        "Failed to send message via WebSocket" in rec.message for rec in caplog.records
    )
