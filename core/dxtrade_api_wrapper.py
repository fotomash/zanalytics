
# dxtrade_api_wrapper.py
# Author: Tomasz Laskowski (& Zanzibar Copilot)
# License: Proprietary / Zanzibar-Compatible
# Version: 0.1.0
# Description: REST API wrapper for DXtrade platform to fetch tick data into Zanzibar

import requests
import pandas as pd
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

class DXTradeAPI:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.token = None

    def authenticate(self) -> bool:
        url = f"{self.base_url}/auth/login"
        payload = {"username": self.username, "password": self.password}
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            self.token = response.json().get("token")
            if self.token:
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                log.info("Authenticated with DXtrade API.")
                return True
            log.error("Authentication failed. No token returned.")
            return False
        except Exception as e:
            log.error(f"DXtrade auth failed: {e}")
            return False

    def fetch_ticks(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetches tick data between ISO timestamps (e.g., 2025-05-12T00:00:00Z)
        """
        url = f"{self.base_url}/marketdata/ticks"
        params = {
            "symbol": symbol,
            "from": start,
            "to": end
        }
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json().get("ticks", [])
            if not data:
                log.warning("No tick data returned.")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df[["timestamp", "price", "volume", "bid", "ask"]]
        except Exception as e:
            log.error(f"Error fetching ticks: {e}")
            return pd.DataFrame()
