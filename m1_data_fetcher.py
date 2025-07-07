# -*- coding: utf-8 -*-
"""
M1 fetcher via Finnhub
Usage: batch_fetch_m1(['BTCUSDT'], days_back=3)
"""
import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone, date
import asyncio

from typing import Optional

USE_DXTRADE_ASYNC = os.getenv("USE_DXTRADE_ASYNC", "0") == "1"
DXTRADE_BASE_URL = os.getenv("DXTRADE_BASE_URL")
DXTRADE_USERNAME = os.getenv("DXTRADE_USERNAME")
DXTRADE_PASSWORD = os.getenv("DXTRADE_PASSWORD")

if USE_DXTRADE_ASYNC:
    try:
        from dxtrade_async import DXTradeAPIAsync

        _ASYNC_CLIENT: Optional[DXTradeAPIAsync] = DXTradeAPIAsync(
            DXTRADE_BASE_URL or "",
            DXTRADE_USERNAME or "",
            DXTRADE_PASSWORD or "",
        )
    except Exception as err:  # pragma: no cover - import failure
        print(f"[WARN] Failed loading dxtrade_async: {err}")
        USE_DXTRADE_ASYNC = False
        _ASYNC_CLIENT = None

API_KEY = os.getenv("FINNHUB_API_KEY")


async def _query_async(symbol, start, end):
    if not _ASYNC_CLIENT:
        raise RuntimeError("DXTrade async client not configured")
    if _ASYNC_CLIENT.token is None:
        await _ASYNC_CLIENT.authenticate()
    ticks = await _ASYNC_CLIENT.fetch_ticks(symbol, start.isoformat(), end.isoformat())
    if ticks.empty:
        return ticks
    ticks = ticks.set_index("timestamp").sort_index()
    ohlc = ticks["price"].resample("1T").ohlc()
    ohlc["volume"] = ticks["volume"].resample("1T").sum()
    ohlc.reset_index(inplace=True)
    ohlc.rename(
        columns={
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
        },
        inplace=True,
    )
    return ohlc


def _query(symbol, start, end):
    if USE_DXTRADE_ASYNC:
        return asyncio.run(_query_async(symbol, start, end))
    url = "https://finnhub.io/api/v1/crypto/candle"
    params = {
        "symbol": symbol,
        "resolution": 1,
        "from": int(start.timestamp()),
        "to": int(end.timestamp()),
        "token": API_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    if j.get("s") != "ok":
        raise ValueError("Finnhub err: %s" % j.get("s"))
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(j["t"], unit="s", utc=True),
            "open": j["o"],
            "high": j["h"],
            "low": j["l"],
            "close": j["c"],
            "volume": j["v"],
        }
    )
    return df


def batch_fetch_m1(symbols, days_back=3, out_dir="tick_data/m1/"):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(days=days_back)
    os.makedirs(out_dir, exist_ok=True)
    for sym in symbols:
        try:
            df = _query(sym, start, now)
            out = (
                f"{out_dir}/{sym.split(':')[-1]}_M1_{start:%Y%m%d}_{now:%Y%m%d%H%M}.csv"
            )
            df.to_csv(out, index=False)
            print(f"[OK] {sym} -> {out} ({len(df)} rows)")
        except Exception as e:
            print(f"[ERR] {sym}: {e}")

    if USE_DXTRADE_ASYNC and _ASYNC_CLIENT:
        asyncio.run(_ASYNC_CLIENT.close())


# Quick test
if __name__ == "__main__":
    if not API_KEY and not USE_DXTRADE_ASYNC:
        raise EnvironmentError("Set FINNHUB_API_KEY or enable DXTRADE client")
    batch_fetch_m1(["BINANCE:BTCUSDT"], days_back=1)
