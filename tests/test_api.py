import os
import sys
from pathlib import Path
import pytest

pytest.skip("API tests require full dependency stack", allow_module_level=True)

from fastapi.testclient import TestClient

os.environ["ZANALYTICS_TEST_MODE"] = "1"
os.environ["ZANALYTICS_API_KEY"] = "test_secret"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import app
import core.orchestrator as orch_mod

client = TestClient(app)


def test_log_endpoint_success():
    response = client.post(
        "/log",
        json={"message": "hello"},
        headers={"X-API-Key": "test_secret"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_ingest_candle_missing_field():
    response = client.post(
        "/api/v1/ingest/candle",
        json={"candle": {"symbol": "EURUSD"}},
    )
    assert response.status_code == 422


def test_confluence_no_data_returns_404():
    response = client.get("/api/v1/analysis/confluence/EURUSD")
    assert response.status_code == 404


def test_latest_data_endpoint_serialization(monkeypatch):
    sample = {
        "timestamp": "2024-01-01T00:00:00",
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "volume": 10,
    }

    class DummyWriter:
        async def get_latest_data(self, symbol: str, timeframe: str, count: int):
            return [sample]

    def simple(df):
        return [orch_mod.UnifiedAnalyticsBar.from_series(row) for _, row in df.reset_index().iterrows()]
    monkeypatch.setattr(sys.modules["main"].analysis_orchestrator, "analyze_dataframe", simple)

    with TestClient(app) as test_client:
        monkeypatch.setattr(sys.modules["main"], "redis_writer", DummyWriter())
        resp = test_client.get("/api/v1/data/latest/EURUSD")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data and set(["timestamp", "open", "high", "low", "close", "volume"]).issubset(data[0].keys())
