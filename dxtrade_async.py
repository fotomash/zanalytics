import httpx
import pandas as pd
import logging

log = logging.getLogger(__name__)


class DXTradeAPIAsync:
    """Asynchronous DXtrade API client using httpx.AsyncClient."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.token = None
        self.client = httpx.AsyncClient()

    async def authenticate(self) -> bool:
        url = f"{self.base_url}/auth/login"
        payload = {"username": self.username, "password": self.password}
        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            self.token = resp.json().get("token")
            if self.token:
                self.client.headers.update({"Authorization": f"Bearer {self.token}"})
                log.info("Authenticated with DXtrade API (async).")
                return True
            log.error("Authentication failed. No token returned.")
            return False
        except Exception as e:
            log.error(f"DXtrade async auth failed: {e}")
            return False

    async def fetch_ticks(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch tick data asynchronously and return as DataFrame."""
        url = f"{self.base_url}/marketdata/ticks"
        params = {"symbol": symbol, "from": start, "to": end}
        try:
            resp = await self.client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json().get("ticks", [])
            if not data:
                log.warning("No tick data returned (async).")
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df[["timestamp", "price", "volume", "bid", "ask"]]
        except Exception as e:
            log.error(f"Error fetching ticks asynchronously: {e}")
            return pd.DataFrame()

    async def close(self) -> None:
        await self.client.aclose()
