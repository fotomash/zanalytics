import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

os.environ["ZANALYTICS_TEST_MODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import app

client = TestClient(app)


def test_log_endpoint_success():
    response = client.post(
        "/log",
        json={"message": "hello"},
        headers={"X-API-Key": "YOUR_SECURE_API_KEY"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
