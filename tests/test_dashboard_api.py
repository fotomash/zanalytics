import os
import sys
from pathlib import Path
import pytest

streamlit = pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultimate_strategy_api import app
from fastapi.testclient import TestClient


def test_consolidated_summary_endpoint():
    with TestClient(app) as client:
        resp = client.get("/summary/consolidated")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        for key in ["symbol", "summaries", "microstructure", "entry_signals", "api"]:
            assert key in data
