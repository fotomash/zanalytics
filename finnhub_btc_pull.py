"""
Pull BTC-USD 1-minute candles from Finnhub
-----------------------------------------
• Uses your API key: d075m51r01qrslhm9nr0d075m51r01qrslhm9nrg
• Time span: last 24 hours
• Output: CSV with columns open, high, low, close, vol
"""

import requests, pandas as pd
from datetime import datetime, timedelta, timezone

API_KEY = "d075m51r01qrslhm9nr0d075m51r01qrslhm9nrg"
SYMBOL  = "BINANCE:BTCUSDT"       # Exchange:symbol format
RES     = "1"                     # 1-minute resolution
DAYS_BACK = 1                     # how many days to fetch

now_utc   = datetime.now(timezone.utc)
from_utc  = now_utc - timedelta(days=DAYS_BACK)
from_unix = int(from_utc.timestamp())
to_unix   = int(now_utc.timestamp())

url = (
    "https://finnhub.io/api/v1/crypto/candle"
    f"?symbol={SYMBOL}&resolution={RES}"
    f"&from={from_unix}&to={to_unix}&token={API_KEY}"
)

resp = requests.get(url, timeout=10)
resp.raise_for_status()
data = resp.json()

if data.get("s") != "ok":
    raise RuntimeError(f"Finnhub returned error → {data}")

df = pd.DataFrame(
    {
        "ts":   [datetime.utcfromtimestamp(ts) for ts in data["t"]],
        "open": data["o"],
        "high": data["h"],
        "low":  data["l"],
        "close":data["c"],
        "vol":  data["v"],
    }
).set_index("ts")

outfile = f"BTCUSD_M1_{from_utc:%Y%m%d%H%M}_{now_utc:%Y%m%d%H%M}.csv"
df.to_csv(outfile, index_label="datetime")
print(f"✔ Saved {len(df)} candles → {outfile}")