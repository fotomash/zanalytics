import os


def get_api_url() -> str:
    """Return base API URL for dashboards."""
    return os.getenv("ZAN_API_URL", "http://localhost:8000")
