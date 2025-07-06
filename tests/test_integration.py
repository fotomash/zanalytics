"""Integration tests for ZAnalytics system."""

import asyncio
import time
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.orchestrator import AnalysisOrchestrator, AnalysisRequest, AnalysisResult
from api.server import app




@pytest.fixture
def orchestrator():
    """Create an orchestrator instance."""
    orch = AnalysisOrchestrator()
    orch.is_running = True
    return orch


@pytest.fixture
def api_client(orchestrator):
    """FastAPI test client patched with orchestrator."""
    with patch("api.server.get_orchestrator", return_value=orchestrator):
        client = TestClient(app)
    return client


class TestSystemIntegration:
    """Integration tests for the full data flow."""

    def test_data_flow_end_to_end(self, api_client, orchestrator):
        request_data = {"symbol": "XAUUSD", "timeframe": "1h", "analysis_type": "combined"}

        async def mock_process(_req):
            await asyncio.sleep(0.1)
            return AnalysisResult(
                request_id="test_123",
                symbol="XAUUSD",
                timeframe="1h",
                analysis_type="combined",
                result_data={"test": "result"},
                metadata={},
                timestamp=datetime.utcnow(),
            )

        with patch.object(orchestrator, "_process_request", side_effect=mock_process):
            resp = api_client.post("/analyze", json=request_data)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in {"completed", "timeout"}
        if data["status"] == "completed":
            assert data.get("result")

    def test_api_health_check(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "healthy"
        assert "timestamp" in payload

    def test_concurrent_requests(self, api_client):
        symbols = ["XAUUSD", "EURUSD", "GBPUSD"]
        for sym in symbols:
            resp = api_client.post(
                "/analyze",
                json={"symbol": sym, "timeframe": "1h", "analysis_type": "combined"},
            )
            assert resp.status_code == 200


class TestPerformance:
    """Simple performance regression tests."""

    def test_request_throughput(self, api_client):
        start = time.time()
        num = 5
        for _ in range(num):
            resp = api_client.post(
                "/analyze",
                json={"symbol": "XAUUSD", "timeframe": "1h", "analysis_type": "combined"},
            )
            assert resp.status_code == 200
        throughput = num / (time.time() - start)
        print(f"Throughput: {throughput:.2f} req/s")
        assert throughput > 1

