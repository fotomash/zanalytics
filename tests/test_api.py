import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

os.environ["ZANALYTICS_TEST_MODE"] = "1"
os.environ["ZANALYTICS_API_KEY"] = "test_secret"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import app

client = TestClient(app)


def test_log_endpoint_success():
    response = client.post(
        "/log",
        json={"message": "hello"},
        headers={"X-API-Key": "test_secret"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
