 # Maps internal keys to readable symbol names for dashboard display
DISPLAY_NAMES = {
    'dxy_quote': 'DXY',
    'vix_quote': 'VIX',
    'gold_quote': 'Gold',
    'oil_quote': 'Oil',
    'us10y_quote': 'US 10Y',
    'de10y_quote': 'Germany 10Y',
    'nasdaq_quote': 'NASDAQ',
    'spx_quote': 'S&P 500',
    'dax_quote': 'DAX',
    'gbpusd_quote': 'GBP/USD',
    'eurusd_quote': 'EUR/USD',
    # Add other keys if needed
}

import os
import pickle
import hashlib
import json
import logging
logging.basicConfig(level=logging.INFO)

# --- PATCH: Ensure .cache directory exists ---
def ensure_cache_dir():
    os.makedirs(".cache", exist_ok=True)
ensure_cache_dir()

# --- PATCH: Improved cache logic ---
def auto_cache(key, fetch_fn, refresh=False, *args, **kwargs):
    ensure_cache_dir()
    # Always serialize the key (including args/kwargs) as JSON for hashing
    keystr = json.dumps(
        {
            "key": key,
            "args": args,
            "kwargs": kwargs,
        },
        sort_keys=True,
        default=str,
    )
    cache_hash = hashlib.md5(keystr.encode('utf-8')).hexdigest()
    cache_file = os.path.join(".cache", f"{cache_hash}.pkl")
    logging.info(f"[CACHE] key={keystr}, file={cache_file}, refresh={refresh}, exists={os.path.exists(cache_file)}")
    if not refresh and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    result = fetch_fn(*args, **kwargs)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


# --- PATCH: Macro Sentiment v2 with context and deduplication ---
import hashlib, json
def load_or_fetch_macro_sentiment(snapshot, asset_news, today_econ_events, market_movers, refresh=False):
    ensure_cache_dir()
    # Use a hash of the snapshot+asset_news+today_econ_events+market_movers for unique caching
    context = json.dumps({'snapshot': snapshot, 'asset_news': asset_news, 'today_econ_events': today_econ_events, 'market_movers': market_movers}, sort_keys=True)
    cache_hash = hashlib.md5(context.encode('utf-8')).hexdigest()
    cache_file = os.path.join(".cache", f"macro_sentiment_{cache_hash}.txt")
    if not refresh and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return f.read()
    result = fetch_openai_macro_sentiment_v2(snapshot, asset_news, today_econ_events, market_movers, refresh=refresh)
    with open(cache_file, "w") as f:
        f.write(result)
    return result
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from fredapi import Fred

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Zanalyttics Dashboard", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

client = OpenAI(api_key=st.secrets["openai_API"])

def get_market_movers(quotes_cache):
    # Find top 3 biggest % movers among the main assets
    movers = []
    for label, quote in quotes_cache.items():
        if isinstance(quote, dict) and "current" in quote and "change" in quote and quote.get("current") not in ("N/A", None) and quote.get("change") not in ("N/A", None):
            try:
                pct = 100 * float(quote["change"]) / float(quote["current"]) if float(quote["current"]) != 0 else 0
                movers.append((label, float(quote["current"]), float(quote["change"]), pct))
            except Exception:
                continue
    # Sort by absolute % change, descending
    movers = sorted(movers, key=lambda x: abs(x[3]), reverse=True)
    return movers[:3]

# --- PATCH: Macro Sentiment Prompt V2 ---
def fetch_openai_macro_sentiment_v2(snapshot, asset_news, today_econ_events, market_movers, refresh=False):
    """
    snapshot: dict with keys dxy, vix, gold, oil, us10y, de10y (each is a quote dict)
    asset_news: dict with keys as asset label, values as headlines/catalyst (str)
    """
    # --- DYNAMIC SUMMARY TABLE SECTION ---
    # Instruments for the summary table
    summary_instruments = [
        ("DXY", "dxy", False),
        ("VIX", "vix", False),
        ("Gold", "gold", False),
        ("Oil", "oil", False),
        ("US 10Y", "us10y", True),
        ("DE 10Y", "de10y", True),
        ("GBP/USD", "gbpusd", False),
        ("EUR/USD", "eurusd", False),
    ]
    summary_table_lines = []
    summary_table_lines.append("## 🧾 Summary Table (Macro Context Focus)\n")
    summary_table_lines.append("| Instrument | Current Price | Trend | Action | Key Macro Context |")
    summary_table_lines.append("|------------|----------------|--------|--------|-------------------|")
    for display, key, is_yield in summary_instruments:
        # Current price, add % for yields
        price = snapshot.get(key, {}).get("current", "N/A")
        # Filter out missing prices ("N/A" or empty)
        if price in ("N/A", "", None):
            continue
        if is_yield and price != "N/A":
            price = f"{price}%"
        # Macro context: first line of asset_news, or em dash
        news_val = asset_news.get(key, "")
        if news_val:
            macro_context = news_val.splitlines()[0] if news_val.splitlines() else news_val
        else:
            macro_context = "—"
        # Table row
        summary_table_lines.append(
            f"| {display} | {price} | TBD | TBD | {macro_context} |"
        )
    summary_table_md = "\n".join(summary_table_lines) + "\n"
    # --- END DYNAMIC SUMMARY TABLE SECTION ---

    # --- PATCH: Macro Calendar/Economic Events: Parse and format upcoming major events ---
    # Only show if at least one relevant event is found within 72h
    event_summaries = ""
    macro_events_block = ""
    try:
        df_events = None
        if today_econ_events is not None and isinstance(today_econ_events, dict) and today_econ_events:
            import pandas as pd
            df_events = pd.DataFrame(today_econ_events)
            # If dict-of-lists, transpose if needed
            if set(df_events.columns) >= {"date", "country", "event"}:
                pass
            elif set(df_events.index) >= {"date", "country", "event"}:
                df_events = df_events.T
        if df_events is not None and not df_events.empty:
            # Only keep events within 72 hours from now
            now = pd.Timestamp.now(tz=None)
            in_72h = now + pd.Timedelta(hours=72)
            df_events['date'] = pd.to_datetime(df_events['date'], errors='coerce')
            df_events = df_events[df_events['date'].notnull()]
            df_events_72h = df_events[(df_events['date'] >= now) & (df_events['date'] <= in_72h)]
            # Only keep high-impact events for central banks or major macro
            relevant_words = ["fed", "fomc", "powell", "ecb", "lagarde", "boe", "boj", "kuroda", "rate", "cpi", "nfp", "payroll", "unemployment", "inflation", "policy", "decision", "minutes", "interest"]
            def is_relevant(ev):
                evl = str(ev).lower()
                return any(w in evl for w in relevant_words)
            if not df_events_72h.empty:
                filtered = df_events_72h[df_events_72h['event'].map(is_relevant)]
                # Format as bullet list
                event_lines = []
                for _, row in filtered.iterrows():
                    dstr = row['date'].strftime("%Y-%m-%d")
                    desc = row['event']
                    # Optionally annotate central bank
                    event_lines.append(f"• {dstr}: {desc}")
                if event_lines:
                    event_summaries = "  \n".join(event_lines)
                    macro_events_block = f"""- Upcoming Events (Next 72h):  
{event_summaries}
"""
    except Exception:
        macro_events_block = ""
    # macro_events_block is empty if no relevant events
    prompt = f"""
You are a professional cross-asset market analyst. 

### MANDATORY OUTPUT STRUCTURE — Do not skip any section.

1. For EACH of these instruments:
   - DXY (Dollar Index), VIX (Volatility Index), Gold (XAU/USD), Oil (WTI), US10Y, DE10Y, NASDAQ, S&P 500, DAX, EUR/USD, GBP/USD, EUR/GBP
   
   For each:
   - **Current Price/Level**: [exact number and today’s movement]
   - **Key Levels**: Support [number], Resistance [number]
   - **Trend**: [Bullish/Bearish/Neutral] with clear, specific reason
   - **Action**: [Buy/Sell/Hold] at [level], with suggested stop/target

2. **Cross-Market Analysis**
   - *What is driving flows, risk-on/off regime, biggest correlations/divergences TODAY?*
   - Highlight safe havens, vol outliers, and if any asset’s move is unexplained.

3. **Breaking Macro News**
   - List 3-5 specific headlines impacting markets RIGHT NOW (no old news)
   - *Explain the market impact of each headline.*

4. **Trading Opportunities**
   - Give 2-3 high-conviction trades (entry, stop, target, rationale, R/R)

5. **Macro Calendar/Economic Events**
   {"- Upcoming Events (Next 72h): " + event_summaries if macro_events_block else "- What’s upcoming for US/EUR/UK/JPY macro? Highlight Fed, ECB, BOE, BOJ, CPI, NFP, etc."}
   {" " if not macro_events_block else ""}
   - Is there a central bank or major data event in the next 72h? *State when.*

6. **Sentiment/Volatility/Special**
   - What’s the “fear/greed” or sentiment index today? Any sector, asset, or FX pair showing extreme sentiment or volatility?
   - Which asset class is outperforming?

7. **Summary Table**
   - Tabulate all current prices, trends, and trade recommendations for each instrument above.

8. **RISK/SCENARIO WARNING**
   - Briefly note 1-2 plausible market scenarios and cross-asset risks for the next session.

**Formatting**:  
- Use markdown.  
- Bold key levels.  
- Use bullet points for news, actions, and trades.  
- Be specific, clear, and forward-looking.  
- Avoid any vague commentary.

**Your goal: actionable, trader-oriented intelligence, not a bland summary.**

## Market Snapshot
DXY: {snapshot['dxy']['current']}, VIX: {snapshot['vix']['current']}, Gold: {snapshot['gold']['current']}, Oil: {snapshot['oil']['current']}, US 10Y: {snapshot['us10y']['current']}%, DE 10Y: {snapshot['de10y']['current']}%

---
{summary_table_md}
---
### 💵 DXY (US Dollar Index)
- Price: {snapshot['dxy']['current']} ({snapshot['dxy']['change']:+})
- News/Catalyst:  
    **NEWS:**  
{asset_news['dxy']}
- Sentiment: (fill with your inference)
- Commentary: (reason for price move, impact on risk assets, actionable idea)

### 📉 VIX (Volatility Index)
- Price: {snapshot['vix']['current']} ({snapshot['vix']['change']:+})
- News/Catalyst:  
    **NEWS:**  
{asset_news['vix']}
- Sentiment: (inference)
- Commentary: (reasons for vol moves, implications for equities/FX)

### 💷 GBP/USD (Pound Sterling)
- **Price:** {snapshot['gbpusd']['current']} ({snapshot['gbpusd']['change']:+})
- **Chart:** _[insert sparkline/chart here]_
- **News/Catalyst:**  
    **NEWS:**  
{asset_news['gbpusd']}
- **BOE Policy/UK Macro:**  
  (BOE tone, inflation, wage data, Brexit risk)
- **UK 10Y Gilt:**  
    - **Yield:** {snapshot['uk10y']['current']} ({snapshot['uk10y']['change']:+})
    - **Chart:** _[insert sparkline/chart here]_
    - **Macro Commentary:**  
      Discuss recent UK macroeconomic events, BOE policy signals, or fiscal developments affecting Gilt yields.

### 💶 EUR/USD (Euro)
- **Price:** {snapshot['eurusd']['current']} ({snapshot['eurusd']['change']:+})
- **Chart:** _[insert sparkline/chart here]_
- **News/Catalyst:**  
    **NEWS:**  
{asset_news['eurusd']}
- **ECB Policy/Euro Macro:**  
  (ECB hawkish/dovish, EU growth, French/German spread)
- **Cross-Asset Impact:**  
  (Effects on commodities, equities, DAX, etc)
- **Actionable Insight:**  
  (Rising euro = DAX/Euro stocks outperformance, falling = flows to USD)

### 🪙 Gold (XAUUSD)
- Price: {snapshot['gold']['current']} ({snapshot['gold']['change']:+})
- News/Catalyst:  
    **NEWS:**  
{asset_news['gold']}
- Sentiment: (hedging, safe haven, or other flows)
- Commentary: (macro context, USD/yield impact)

### 🛢️ Oil (WTI)
- Price: {snapshot['oil']['current']} ({snapshot['oil']['change']:+})
- News/Catalyst:  
    **NEWS:**  
{asset_news['oil']}
- Sentiment: (demand/supply, OPEC, geopolitical, etc)
- Commentary: (macro theme, risk asset readthrough)

### 🇺🇸 US 10Y Yield
- Yield: {snapshot['us10y']['current']} ({snapshot['us10y']['change']:+})
- News/Catalyst:  
    **NEWS:**  
{asset_news['us10y']}
- Sentiment: (rates, bonds, central bank)
- Commentary: (yield moves and impact on macro/risk)

### 🇩🇪 DE 10Y Bund
- Yield: {snapshot['de10y']['current']} ({snapshot['de10y']['change']:+})
- News/Catalyst:  
    **NEWS:**  
{asset_news['de10y']}
- Sentiment: (rates, ECB, bunds, Europe macro)
- Commentary: (reason for move, impact on EUR/USD, etc)

---
**Cross-Market Summary:**  
- Briefly summarize regime (risk-on/off, flow, major cross-asset relationships, and biggest driver today).  
- List any asset with an unexplained move and say so.  
- Provide actionable takeaways for traders at the end.

*Output markdown, clear headers, actionable comments.*

"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Could not fetch macro sentiment: {e}"
def get_asset_news(asset_keywords):
    # This uses Finnhub and NewsAPI to pull headlines for a given asset
    news = []
    finnhub_df = edm.get_finnhub_headlines(symbols=tuple(asset_keywords), max_articles=4)
    if not finnhub_df.empty:
        for _, row in finnhub_df.iterrows():
            news.append(f"- {row['datetime']}: [{row['headline']}]({row['url']})")
    articles = edm.get_newsapi_headlines(page_size=4)
    for article in articles:
        if any(k.lower() in article['title'].lower() for k in asset_keywords):
            news.append(f"- {article['publishedAt'][:10]}: [{article['title']}]({article['url']})")
    return "\n".join(news[:4]) if news else "No major headlines for this asset."

def get_high_impact_news():
    keywords = ["FOMC", "CPI", "NFP", "unemployment", "GDP", "central bank", "rate hike", "inflation"]
    try:
        import requests
        st.write("DEBUG: get_high_impact_news - about to fetch news from Finnhub")
        response = requests.get("https://finnhub.io/api/v1/news?category=general&token=" + st.secrets["finnhub_API"])
        articles = response.json()
        st.write("DEBUG: Raw articles fetched from Finnhub:", articles)
        filtered = [a for a in articles if any(k.lower() in a["headline"].lower() for k in keywords)]
        st.write("DEBUG: Filtered high-impact news:", filtered)
        return filtered
    except:
        return []
class EconomicDataManager:
    def get_short_volume_data(self, ticker="AAPL", limit=7):
        import requests
        import pandas as pd

        api_key = st.secrets.get("polygon_api_key") or "DyEadGzDCLwCJomppjGgDFXXUCW94ONO"
        url = "https://api.polygon.io/stocks/v1/short-volume"
        params = {
            "ticker": ticker,
            "limit": limit,
            "sort": "date.desc",
            "apiKey": api_key
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json().get("results", [])
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "short_volume_ratio", "short_volume", "total_volume"]]
        except Exception as e:
            st.error(f"Failed to fetch short volume: {e}")
            return pd.DataFrame()

    def get_today_econ_events():
        today = pd.Timestamp.now().normalize()
        tomorrow = today + pd.Timedelta(days=1)
        try:
            df = edm.get_economic_events_api(
                country_list=['united states', 'germany', 'japan', 'united kingdom', 'euro area'],
                importance="high"
            )
            if not df.empty:
                mask = (df['date'].dt.date >= today.date()) & (df['date'].dt.date <= tomorrow.date())
                today_events = df[mask]
                if not today_events.empty:
                    return today_events[['date', 'country', 'event', 'actual', 'forecast', 'previous']]
            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()
    def get_finnhub_headlines(
        self,
        symbols=("XAU", "EURUSD", "GBPUSD", "BTC"),
        max_articles=12,
        category="forex",
    ):
        """
        Fetch headlines from Finnhub and keep only those that mention any of the
        symbols tuple (case-insensitive).
        """
        import requests, time, pandas as pd, datetime as dt

        API_KEY = st.secrets.get("finnhub_api_key") or "d07lgo1r01qrslhp3q3g"
        url = "https://finnhub.io/api/v1/news"
        params = {"category": category, "token": API_KEY}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            raw = r.json()
            filtered = [
                art for art in raw
                if any(sym.lower() in (art.get("headline", "") + art.get("summary", "")).lower() for sym in symbols)
            ][:max_articles]
            if not filtered:
                return pd.DataFrame()
            df = pd.DataFrame(filtered)
            df["datetime"] = df["datetime"].apply(
                lambda x: dt.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M")
            )
            df = df[["datetime", "source", "headline", "url"]]
            return df
        except Exception as e:
            st.error(f"Finnhub news fetch failed: {e}")
            return pd.DataFrame()
    def get_newsapi_headlines(self, country="gb", category="business", page_size=10):
        """
        Fetches top news headlines from NewsAPI for a given country and category.
        """
        import requests
        API_KEY = st.secrets.get("newsapi_key") or "713b3bd82121482aaa0ecdc9af77b6da"
        url = f"https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": API_KEY,
            "country": country,
            "category": category,
            "pageSize": page_size
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            articles = r.json().get("articles", [])
            return articles
        except Exception as e:
            st.error(f"Failed to fetch news: {e}")
            return []
    def get_economic_events_api(self, country_list=None, importance="high"):
        """
        Fetches upcoming economic events from Trading Economics API.
        Args:
            country_list: List of country names (e.g., ['united states', 'germany', 'united kingdom', 'euro area']) or None for all.
            importance: "high", "medium", "low", or None for all events.
        Returns:
            pd.DataFrame of events (date, country, event, actual, forecast, previous, importance).
        """
        import requests
        import pandas as pd

        API_KEY = st.secrets.get("trading_economics_api_key") or "1750867cdfc34c6:288nxdz64y932qq"
        base_url = "https://api.tradingeconomics.com/calendar"
        params = {"c": API_KEY}
        if country_list:
            params["countries"] = ",".join(country_list)
        importance_map = {"high": 3, "medium": 2, "low": 1}
        if importance in importance_map:
            params["importance"] = importance_map[importance]

        try:
            r = requests.get(base_url, params=params, timeout=15)
            r.raise_for_status()
            events = r.json()
            if not events:
                return pd.DataFrame()
            df = pd.DataFrame(events)
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'country', 'event', 'actual', 'forecast', 'previous', 'importance', 'unit', 'reference']]
            return df
        except Exception as e:
            st.error(f"Failed to fetch economic calendar: {e}")
            return pd.DataFrame()
    def get_index_quote_history(self, ticker: str, label: str, lookback=14):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=f"{lookback}d")
            if not hist.empty:
                hist = hist[['Close']]
                hist = hist.rename(columns={"Close": "Close"})
                hist["Date"] = hist.index
                return hist
        except Exception:
            return None
    """ Manages fetching live economic data using yfinance and web scraping. """

    def _safe_secret(self, key, default=""):
        return st.secrets.get(key) if key in st.secrets else default

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the data manager.
        The api_key is kept for potential future use but is not required for yfinance.
        """
        self.api_key = api_key
        fred_key = st.secrets.get("fred_api_key") or "6a980b8c2421503564570ecf4d765173"
        self.fred = Fred(api_key=fred_key)

    def get_svix_quote(self) -> Dict[str, Any]:
        """ Fetches the latest quote for SVIX using yfinance. """
        try:
            ticker = yf.Ticker("SVIX")
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current_price, prev_price = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                return {'name': 'SVIX', 'current': current_price, 'change': current_price - prev_price}
            return {'name': 'SVIX', 'error': 'No data'}
        except Exception as e:
            return {'name': 'SVIX', 'error': str(e)}

    def get_bond_yields(self) -> Dict[str, Dict[str, Any]]:
        """Fetch latest 10‑year bond yields from FRED."""
        tickers = {
            "US 2Y": "DGS2",
            "US 10Y": "DGS10",
            "US 30Y": "DGS30",
            "DE 2Y": "IRLTLT02DEM156N",
            "DE 10Y": "IRLTLT01DEM156N",
            "DE 30Y": "IRLTLT03DEM156N",
            "GB 2Y": "IRLTLT02GBM156N",
            "GB 10Y": "IRLTLT01GBM156N",
            "GB 30Y": "IRLTLT03GBM156N",
            "JP 2Y": "IRLTLT02JPM156N",
            "JP 10Y": "IRLTLT01JPM156N",
            "JP 30Y": "IRLTLT03JPM156N",
        }
        yield_data = {}
        for name, code in tickers.items():
            try:
                series = self.fred.get_series(code)
                if series is not None and not series.empty:
                    current = float(series.dropna().iloc[-1])
                    prev = float(series.dropna().iloc[-2]) if len(series.dropna()) >= 2 else current
                    yield_data[name] = {
                        'current': round(current, 3),
                        'change': round(current - prev, 3)
                    }
                else:
                    yield_data[name] = {'error': 'No data'}
            except Exception:
                yield_data[name] = {'error': 'Fetch failed'}
        return yield_data

    def get_index_quote(self, ticker: str, label: str) -> Dict[str, Any]:
        """Fetch last and delta for an index using yfinance."""
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="3d")
            if len(hist) >= 2:
                last, prev = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                return {'name': label, 'current': round(last, 2), 'change': round(last - prev, 2)}
            elif len(hist) == 1:
                last = hist['Close'].iloc[-1]
                return {'name': label, 'current': round(last, 2), 'change': 0}
            else:
                return {'name': label, 'error': 'No data'}
        except Exception as e:
            return {'name': label, 'error': str(e)}

    # Removed get_economic_events (obsolete scraping method)

    def get_polygon_news(self, ticker="C:EURUSD", limit=10):
        """Fetch recent news articles for a specific asset via Polygon.io."""
        import requests
        api_key = st.secrets.get("polygon_api_key") or "DyEadGzDCLwCJomppjGgDFXXUCW94ONO"
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit={limit}&apiKey={api_key}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as e:
            st.warning(f"Polygon news fetch failed for {ticker}: {e}")
            return []

    def get_polygon_fx_snapshot(self, pairs=("C:EURUSD", "C:GBPUSD", "C:USDJPY")):
        """
        Fetch real‑time FX bid/ask/last via Polygon snapshot endpoint.
        Returns a dict keyed by pair.
        """
        import requests, pandas as pd, datetime as dt
        api_key = st.secrets.get("polygon_api_key") or "DyEadGzDCLwCJomppjGgDFXXUCW94ONO"
        results = {}
        for p in pairs:
            url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/forex/tickers/{p}?apiKey={api_key}"
            try:
                r = requests.get(url, timeout=6)
                r.raise_for_status()
                data = r.json().get("ticker", {})
                if data:
                    last = data.get("close", {}).get("price")
                    change = data.get("day", {}).get("change")
                    pct = data.get("day", {}).get("percent_change")
                    results[p.replace('C:','')] = {
                        "last": round(last, 5) if last else None,
                        "Δ": f"{change:+.5f}" if change else None,
                        "%": f"{pct:+.2f}%" if pct else None,
                        "time": dt.datetime.utcfromtimestamp(data.get("updated")).strftime("%H:%M:%S") if data.get("updated") else ""
                    }
            except Exception as e:
                results[p] = {"error": str(e)}
        return pd.DataFrame.from_dict(results, orient="index")


edm = EconomicDataManager()


# --- Original Dashboard Below ---

#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from bs4 import BeautifulSoup  # For web scraping
import base64  # For image encoding
import yfinance as yf  # Added for reliable financial data
from fredapi import Fred

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')


# --- Utility Function for Background Image ---
def get_image_as_base64(path):
    """Reads an image file and returns its base64 encoded string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Background image not found at '{path}'. Please ensure it's in the same directory as the script.")
        return None


# --- Economic Data Manager ---
class EconomicDataManager:
    def get_short_volume_data(self, ticker="AAPL", limit=7):
        import requests
        import pandas as pd

        api_key = st.secrets.get("polygon_api_key") or "DyEadGzDCLwCJomppjGgDFXXUCW94ONO"
        url = "https://api.polygon.io/stocks/v1/short-volume"
        params = {
            "ticker": ticker,
            "limit": limit,
            "sort": "date.desc",
            "apiKey": api_key
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json().get("results", [])
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "short_volume_ratio", "short_volume", "total_volume"]]
        except Exception as e:
            st.error(f"Failed to fetch short volume: {e}")
            return pd.DataFrame()
    def get_finnhub_headlines(
        self,
        symbols=("XAU", "EURUSD", "GBPUSD", "BTC"),
        max_articles=12,
        category="forex",
    ):
        """
        Fetch headlines from Finnhub and keep only those that mention any of the
        symbols tuple (case-insensitive).
        """
        import requests, time, pandas as pd, datetime as dt

        API_KEY = st.secrets.get("finnhub_api_key") or "d07lgo1r01qrslhp3q3g"
        url = "https://finnhub.io/api/v1/news"
        params = {"category": category, "token": API_KEY}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            raw = r.json()
            filtered = [
                art for art in raw
                if any(sym.lower() in (art.get("headline", "") + art.get("summary", "")).lower() for sym in symbols)
            ][:max_articles]
            if not filtered:
                return pd.DataFrame()
            df = pd.DataFrame(filtered)
            df["datetime"] = df["datetime"].apply(
                lambda x: dt.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M")
            )
            df = df[["datetime", "source", "headline", "url"]]
            return df
        except Exception as e:
            st.error(f"Finnhub news fetch failed: {e}")
            return pd.DataFrame()
    def get_newsapi_headlines(self, country="gb", category="business", page_size=10):
        """
        Fetches top news headlines from NewsAPI for a given country and category.
        """
        import requests
        API_KEY = st.secrets.get("newsapi_key") or "713b3bd82121482aaa0ecdc9af77b6da"
        url = f"https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": API_KEY,
            "country": country,
            "category": category,
            "pageSize": page_size
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            articles = r.json().get("articles", [])
            return articles
        except Exception as e:
            st.error(f"Failed to fetch news: {e}")
            return []
    def get_economic_events_api(self, country_list=None, importance="high"):
        """
        Fetches upcoming economic events from Trading Economics API.
        Args:
            country_list: List of country names (e.g., ['united states', 'germany', 'united kingdom', 'euro area']) or None for all.
            importance: "high", "medium", "low", or None for all events.
        Returns:
            pd.DataFrame of events (date, country, event, actual, forecast, previous, importance).
        """
        import requests
        import pandas as pd

        API_KEY = st.secrets.get("trading_economics_api_key") or "1750867cdfc34c6:288nxdz64y932qq"
        base_url = "https://api.tradingeconomics.com/calendar"
        params = {"c": API_KEY}
        if country_list:
            params["countries"] = ",".join(country_list)
        importance_map = {"high": 3, "medium": 2, "low": 1}
        if importance in importance_map:
            params["importance"] = importance_map[importance]

        try:
            r = requests.get(base_url, params=params, timeout=15)
            r.raise_for_status()
            events = r.json()
            if not events:
                return pd.DataFrame()
            df = pd.DataFrame(events)
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'country', 'event', 'actual', 'forecast', 'previous', 'importance', 'unit', 'reference']]
            return df
        except Exception as e:
            st.error(f"Failed to fetch economic calendar: {e}")
            return pd.DataFrame()
    def get_index_quote_history(self, ticker: str, label: str, lookback=14):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=f"{lookback}d")
            if not hist.empty:
                hist = hist[['Close']]
                hist = hist.rename(columns={"Close": "Close"})
                hist["Date"] = hist.index
                return hist
        except Exception:
            return None
    """ Manages fetching live economic data using yfinance and web scraping. """

    def _safe_secret(self, key, default=""):
        return st.secrets.get(key) if key in st.secrets else default

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the data manager.
        The api_key is kept for potential future use but is not required for yfinance.
        """
        self.api_key = api_key
        fred_key = st.secrets.get("fred_api_key") or "6a980b8c2421503564570ecf4d765173"
        self.fred = Fred(api_key=fred_key)

    def get_svix_quote(self) -> Dict[str, Any]:
        """ Fetches the latest quote for SVIX using yfinance. """
        try:
            ticker = yf.Ticker("SVIX")
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current_price, prev_price = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                return {'name': 'SVIX', 'current': current_price, 'change': current_price - prev_price}
            return {'name': 'SVIX', 'error': 'No data'}
        except Exception as e:
            return {'name': 'SVIX', 'error': str(e)}

    def get_macro_indicators(self):
        indicators = {}
        try:
            indicators['CPI'] = float(self.fred.get_series('CPIAUCSL').dropna().iloc[-1])
        except:
            indicators['CPI'] = 'N/A'
        try:
            indicators['Unemployment'] = float(self.fred.get_series('UNRATE').dropna().iloc[-1])
        except:
            indicators['Unemployment'] = 'N/A'
        try:
            indicators['GDP'] = float(self.fred.get_series('GDP').dropna().iloc[-1])
        except:
            indicators['GDP'] = 'N/A'
        try:
            indicators['ISM'] = float(self.fred.get_series('NAPM').dropna().iloc[-1])
        except:
            indicators['ISM'] = 'N/A'
        return indicators

    def get_bond_yields(self) -> Dict[str, Dict[str, Any]]:
        """Fetch latest 10‑year bond yields from FRED."""
        tickers = {
            "US 10Y": "DGS10",
            "DE 10Y": "IRLTLT01DEM156N",
            "GB 10Y": "IRLTLT01GBM156N",
            "JP 10Y": "IRLTLT01JPM156N",
        }
        yield_data = {}
        for name, code in tickers.items():
            try:
                series = self.fred.get_series(code)
                if series is not None and not series.empty:
                    current = float(series.dropna().iloc[-1])
                    prev = float(series.dropna().iloc[-2]) if len(series.dropna()) >= 2 else current
                    yield_data[name] = {
                        'current': round(current, 3),
                        'change': round(current - prev, 3)
                    }
                else:
                    yield_data[name] = {'error': 'No data'}
            except Exception:
                yield_data[name] = {'error': 'Fetch failed'}
        return yield_data

    def get_index_quote(self, ticker: str, label: str) -> Dict[str, Any]:
        """Fetch last and delta for an index using yfinance."""
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="3d")
            if len(hist) >= 2:
                last, prev = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                return {'name': label, 'current': round(last, 2), 'change': round(last - prev, 2)}
            elif len(hist) == 1:
                last = hist['Close'].iloc[-1]
                return {'name': label, 'current': round(last, 2), 'change': 0}
            else:
                return {'name': label, 'error': 'No data'}
        except Exception as e:
            return {'name': label, 'error': str(e)}

    # Removed get_economic_events (obsolete scraping method)

    def get_polygon_news(self, ticker="C:EURUSD", limit=10):
        """Fetch recent news articles for a specific asset via Polygon.io."""
        import requests
        api_key = st.secrets.get("polygon_api_key") or "DyEadGzDCLwCJomppjGgDFXXUCW94ONO"
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit={limit}&apiKey={api_key}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as e:
            st.warning(f"Polygon news fetch failed for {ticker}: {e}")
            return []

    def get_polygon_fx_snapshot(self, pairs=("C:EURUSD", "C:GBPUSD", "C:USDJPY")):
        """
        Fetch real‑time FX bid/ask/last via Polygon snapshot endpoint.
        Returns a dict keyed by pair.
        """
        import requests, pandas as pd, datetime as dt
        api_key = st.secrets.get("polygon_api_key") or "DyEadGzDCLwCJomppjGgDFXXUCW94ONO"
        results = {}
        for p in pairs:
            url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/forex/tickers/{p}?apiKey={api_key}"
            try:
                r = requests.get(url, timeout=6)
                r.raise_for_status()
                data = r.json().get("ticker", {})
                if data:
                    last = data.get("close", {}).get("price")
                    change = data.get("day", {}).get("change")
                    pct = data.get("day", {}).get("percent_change")
                    results[p.replace('C:','')] = {
                        "last": round(last, 5) if last else None,
                        "Δ": f"{change:+.5f}" if change else None,
                        "%": f"{pct:+.2f}%" if pct else None,
                        "time": dt.datetime.utcfromtimestamp(data.get("updated")).strftime("%H:%M:%S") if data.get("updated") else ""
                    }
            except Exception as e:
                results[p] = {"error": str(e)}
        return pd.DataFrame.from_dict(results, orient="index")


class MarketOverviewDashboard:
    def display_bond_yields_sparklines(self):
        import plotly.graph_objects as go
        import yfinance as yf

        st.markdown("<h5 style='text-align:center;'>🌍 Bond Yields & Key FX Rates</h5>", unsafe_allow_html=True)
        st.markdown(
            '''
            <div style='
                background-color: rgba(0, 0, 0, 0.25);
                padding: 1.1rem;
                margin: 0.8rem 0 1.4rem 0;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
            '>
            ''',
            unsafe_allow_html=True,
        )

        panel_assets = [
            ("US 10Y", "DGS10", "bond"),
            ("Germany 10Y", "IRLTLT01DEM156N", "bond"),
            ("Japan 10Y", "IRLTLT01JPM156N", "bond"),
            ("UK 10Y", "IRLTLT01GBM156N", "bond"),
            ("EUR/USD", "EURUSD=X", "fx"),
            ("GBP/USD", "GBPUSD=X", "fx"),
        ]
        cols = st.columns(len(panel_assets))
        for i, (label, ticker, kind) in enumerate(panel_assets):
            with cols[i]:
                try:
                    if kind == "bond":
                        series = self.economic_manager.fred.get_series(ticker).dropna()
                        latest = float(series.iloc[-1])
                        prev = float(series.iloc[-2]) if len(series) > 1 else latest
                        val = round(latest, 3)
                        delta = round(latest - prev, 3)
                    else:
                        fx_hist = yf.Ticker(ticker).history(period="60d")['Close']
                        latest = float(fx_hist.iloc[-1])
                        prev = float(fx_hist.iloc[-2]) if len(fx_hist) > 1 else latest
                        val = round(latest, 4)
                        delta = round(latest - prev, 4)
                        series = fx_hist
                    st.metric(label, f"{val}%" if kind == "bond" else f"{val}", delta)
                    # Sparkline
                    if len(series) >= 2:
                        yvals = series[-50:].values
                        xvals = list(range(len(yvals)))
                        fig_spark = go.Figure()
                        fig_spark.add_trace(go.Scatter(
                            x=xvals,
                            y=yvals,
                            mode="lines",
                            line=dict(color="#FFD600", width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        ))
                        fig_spark.update_layout(
                            margin=dict(l=0, r=0, t=10, b=10),
                            height=80,
                            width=160,
                            paper_bgcolor="rgba(0,0,0,0.0)",
                            plot_bgcolor="rgba(0,0,0,0.0)",
                        )
                        fig_spark.update_xaxes(visible=False, showgrid=False, zeroline=False)
                        fig_spark.update_yaxes(visible=False, showgrid=False, zeroline=False)
                        st.plotly_chart(fig_spark, use_container_width=True)
                except Exception:
                    st.metric(label, "N/A", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    def display_live_fx_quotes(self):
        st.markdown("### 💱 Live FX Quotes")
        df_fx = self.economic_manager.get_polygon_fx_snapshot(
            pairs=("C:EURUSD", "C:GBPUSD", "C:USDJPY")
        )
        if df_fx.empty:
            st.caption("FX snapshots unavailable.")
        else:
            st.dataframe(df_fx, use_container_width=True, hide_index=False)

    def display_finnhub_news(self):
        st.markdown("### 🗞️ Market-Moving Headlines (Gold · EUR USD · GBP USD · BTC)")
        df_news = self.economic_manager.get_finnhub_headlines()
        if df_news.empty:
            st.info("No relevant Finnhub headlines right now.")
            return
        for _, row in df_news.iterrows():
            st.markdown(f"**[{row['headline']}]({row['url']})**")
            st.caption(f"{row['source']} — {row['datetime']}")
            st.markdown("---")

    def display_polygon_news(self):
        st.markdown("### 🗞️ Polygon.io News Feed")
        tickers = ["C:EURUSD", "C:GBPUSD", "X:BTCUSD"]
        for ticker in tickers:
            st.markdown(f"#### {ticker}")
            articles = self.economic_manager.get_polygon_news(ticker=ticker)
            if not articles:
                st.caption("No news available.")
                continue
            for art in articles:
                st.markdown(f"**[{art['title']}]({art['article_url']})**")
                st.caption(f"{art['publisher']['name']} — {art['published_utc'][:10]}")
                st.markdown("---")
    def __init__(self):
        """
        Initializes the dashboard, loading configuration from Streamlit secrets.
        """
        try:
            # Load data directory from secrets.toml
            data_directory = st.secrets["data_directory"]
        except (FileNotFoundError, KeyError):
            # Fallback to a default directory if secrets.toml or the key is missing
            data_directory = "./data"

        self.data_dir = Path(data_directory)
        self.supported_pairs = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "ETHUSD", "USDCAD", "AUDUSD",
                                "NZDUSD"]
        self.timeframes = ["1min", "5min", "15min", "30min", "1H", "4H", "1D", "1W"]

        self.economic_manager = EconomicDataManager(api_key=None)

        if 'chart_theme' not in st.session_state:
            st.session_state.chart_theme = 'plotly_dark'

    def run(self):


        img_base64 = get_image_as_base64("image_af247b.jpg")
        if img_base64:
            background_style = f"""
            <style>
            [data-testid="stAppViewContainer"] > .main {{
                background-image: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url(data:image/jpeg;base64,{img_base64});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .stMetric {{ border-radius: 10px; padding: 15px; background-color: #2a2a39; }}
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            </style>
            """
            st.markdown(background_style, unsafe_allow_html=True)

        # Check if the configured data directory exists
        if not self.data_dir.exists():
            st.error(f"Data directory not found at: `{self.data_dir}`")
            st.info(
                "Please create this directory or configure the correct path in your `.streamlit/secrets.toml` file.")
            st.code('data_directory = "/path/to/your/data"')
            return

        with st.spinner("🛰️ Scanning all data sources..."):
            data_sources = self.scan_all_data_sources()

        self.create_sidebar(data_sources)
        self.display_home_page(data_sources)

        # To show the bond yields panel on any page:
        # self.display_bond_yields_sparklines()

    def create_sidebar(self, data_sources):
        st.sidebar.title("Data Status")
        summary = [{'Pair': pair, 'Timeframes Found': len(data_sources.get(pair, {}))} for pair in self.supported_pairs
                   if data_sources.get(pair)]
        if summary:
            st.sidebar.dataframe(pd.DataFrame(summary).set_index('Pair'), use_container_width=True)
        else:
            st.sidebar.warning("No data found.")

    def display_home_page(self, data_sources):
        # Always show macro sentiment snapshot first
        self.display_macro_sentiment()

        # st.markdown("## 🏠 Welcome to the Zanalyttics Dashboard")

        market_data = self._load_market_data(data_sources)

        # --- PATCH: Modernize Key Index & Commodity Trends with snapshot mini-cards ---
        # Trend metrics cache, 3 per row, consistent mini-card style
        refresh_market_data = st.session_state.get("refresh_market_data", False)
        def render_snapshot_grouped(metrics):
            st.markdown(
                '''
                <div style='background-color: rgba(0,0,0,0.25); padding: 1.1rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.12); margin-bottom:1.4rem;'>
                ''', unsafe_allow_html=True
            )
            for i in range(0, len(metrics), 3):
                row = metrics[i:i+3]
                cols = st.columns(len(row))
                for col, (label, quote, yvals) in zip(cols, row):
                    value = quote.get("current", "N/A")
                    delta = quote.get("change", 0)
                    color = "#26de81" if delta and float(delta) > 0 else "#fc5c65" if delta and float(delta) < 0 else "#e7eaf0"
                    with col:
                        st.markdown(f"<div style='text-align:center;color:#e7eaf0;font-size:1.05em;'>{label}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align:center;font-size:2.4em;font-weight:bold;color:white'>{value}</div>", unsafe_allow_html=True)
                        if delta not in ("N/A", None):
                            arrow = "▲" if float(delta) > 0 else "▼"
                            st.markdown(f"<div style='text-align:center;font-size:1.1em;color:{color};padding-bottom:0.2em;'>{arrow} {delta}</div>", unsafe_allow_html=True)
                        # Sparkline (cached)
                        if yvals:
                            import plotly.graph_objects as go
                            fig_spark = go.Figure()
                            fig_spark.add_trace(go.Scatter(
                                x=list(range(len(yvals))),
                                y=yvals,
                                mode="lines",
                                line=dict(color="#FFD600", width=2),
                                showlegend=False,
                                hoverinfo="skip",
                            ))
                            fig_spark.update_layout(
                                margin=dict(l=0, r=0, t=10, b=10),
                                height=70,
                                width=150,
                                paper_bgcolor="rgba(0,0,0,0.0)",
                                plot_bgcolor="rgba(0,0,0,0.0)",
                            )
                            fig_spark.update_xaxes(visible=False, showgrid=False, zeroline=False)
                            fig_spark.update_yaxes(visible=False, showgrid=False, zeroline=False)
                            st.plotly_chart(fig_spark, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        trend_keys = [
            ("NASDAQ", "nasdaq_quote", "nasdaq_chart", "^IXIC", "NASDAQ"),
            ("US30", "us30_quote", "us30_chart", "^DJI", "US30"),
            ("DAX", "dax_quote", "dax_chart", "^GDAXI", "DAX"),
            ("Gold", "gold_quote", "gold_chart", "GC=F", "Gold"),
            ("DXY", "dxy_quote", "dxy_chart", "DX-Y.NYB", "DXY"),
            ("EURUSD", "eurusd_quote", "eurusd_chart", "EURUSD=X", "EURUSD"),
        ]
        trend_metrics = []
        for label, qkey, ckey, ticker, tlabel in trend_keys:
            quote = auto_cache((qkey, ticker, tlabel), lambda t=ticker, l=tlabel: edm.get_index_quote(t, l), refresh=refresh_market_data)
            chart_hist = auto_cache((ckey, ticker, tlabel), lambda t=ticker, l=tlabel: edm.get_index_quote_history(t, l, lookback=20)['Close'].tolist() if edm.get_index_quote_history(t, l, lookback=20) is not None else [], refresh=refresh_market_data)
            trend_metrics.append((label, quote, chart_hist))

        st.markdown("### 📈 Key Index & Commodity Trends")
        render_snapshot_grouped(trend_metrics)

        self.display_next_week_events()
        self.display_news_headlines()
        st.markdown("⚠️ FX quotes and news feeds are temporarily unavailable due to API restrictions. Please check your API keys and access rights.")

    # display_selected_index_trends removed (replaced by modern snapshot cards)

    def display_news_headlines(self):
        st.markdown("### 🗞️ UK Business News")
        articles = self.economic_manager.get_newsapi_headlines(country="gb", category="business", page_size=5)
        if articles:
            for article in articles:
                st.markdown(f"**[{article['title']}]({article['url']})**")
                st.caption(f"{article['source']['name']} — {article['publishedAt'][:10]}")
                st.markdown(f"{article['description']}")
                st.markdown("---")
        else:
            st.info("No recent news available.")

    def display_multi_timeframe_candles(self, market_data):
        """Displays a streamlined multi-timeframe trend and candle view (no volume profile)."""
        st.markdown("### Multi-Timeframe Trend Analysis")

        for pair, tfs_data in market_data.items():
            with st.expander(f"**{pair}**"):
                available_tfs = [tf for tf in self.timeframes if tf in tfs_data]
                if not available_tfs:
                    st.caption("No data available for this pair.")
                    continue

                cols = st.columns(len(available_tfs))
                for i, tf in enumerate(available_tfs):
                    with cols[i]:
                        st.markdown(f"**{tf}**")
                        df = tfs_data[tf]
                        if 'close' in df.columns and 'open' in df.columns and len(df) >= 2:
                            last_candle = df.iloc[-1]
                            is_bullish = last_candle['close'] > last_candle['open']
                            # Defensive check for sma_20 before calculating trend
                            if 'sma_20' in last_candle and pd.notna(last_candle['sma_20']):
                                trend = "🟢 Up" if last_candle['close'] > last_candle['sma_20'] else "🔴 Down"
                            else:
                                trend = "N/A"
                            st.markdown(f"{'🟢' if is_bullish else '🔴'} Candle")
                            st.markdown(f"Trend: {trend}")
                        else:
                            st.caption("No data")

    def display_macro_sentiment(self):
        import plotly.graph_objects as go
        import re
        import os, pickle
        # --- PATCH: Unified Refresh Button at the top ---
        refresh_market_data = st.button("🔄 Refresh Data", key="refresh_market_data")

        # --- PATCH: Cache all metrics ONCE at the top; group snapshot charts in rows of 3 ---
        # Chart keys: label, quote_key, chart_key, ticker, label
        chart_keys = [
            ("DXY", "dxy_quote", "dxy_chart", "DX-Y.NYB", "DXY"),
            ("VIX", "vix_quote", "vix_chart", "^VIX", "VIX"),
            ("Gold", "gold_quote", "gold_chart", "GC=F", "Gold"),
            ("Oil", "oil_quote", "oil_chart", "CL=F", "Oil"),
            ("US 10Y", "us10y_quote", "us10y_chart", "DGS10", "US 10Y"),
            ("DE 10Y", "de10y_quote", "de10y_chart", "IRLTLT01DEM156N", "DE 10Y"),
            ("NASDAQ", "nasdaq_quote", "nasdaq_chart", "^IXIC", "NASDAQ"),
            ("S&P", "spx_quote", "spx_chart", "^GSPC", "S&P 500"),
            ("DAX", "dax_quote", "dax_chart", "^GDAXI", "DAX"),
        ]



        # --- PATCH: Deduplicate by pre-fetching/caching all quotes and all chart histories in two dictionaries, then using those below ---
        quotes_cache = {}
        today_econ_events = auto_cache(
            ("today_econ_events", "united states", "germany", "japan", "united kingdom", "euro area", "high"),
            lambda: edm.get_economic_events_api(
                country_list=['united states', 'germany', 'japan', 'united kingdom', 'euro area'],
                importance="high"
            ),
            refresh=refresh_market_data
        )
        if not today_econ_events.empty:
            today = pd.Timestamp.now().normalize()
            tomorrow = today + pd.Timedelta(days=1)
            mask = (today_econ_events['date'].dt.date >= today.date()) & (today_econ_events['date'].dt.date <= tomorrow.date())
            today_econ_events = today_econ_events[mask][['date', 'country', 'event', 'actual', 'forecast', 'previous']]
        else:
            today_econ_events = pd.DataFrame()
        market_movers = get_market_movers(quotes_cache)
        history_cache = {}
        for label, qkey, ckey, ticker, tlabel in chart_keys:
            if "10Y" not in label:
                quotes_cache[qkey] = auto_cache((qkey, ticker, tlabel), lambda t=ticker, l=tlabel: edm.get_index_quote(t, l), refresh=refresh_market_data)
                history_cache[ckey] = auto_cache(
                    (ckey, ticker, tlabel),
                    lambda t=ticker, l=tlabel: (
                        edm.get_index_quote_history(t, l, lookback=20)['Close'].tolist()
                        if edm.get_index_quote_history(t, l, lookback=20) is not None else []
                    ),
                    refresh=refresh_market_data
                )
            else:
                quotes_cache[qkey] = auto_cache((qkey, label), lambda l=label: edm.get_bond_yields().get(label, {}), refresh=refresh_market_data)
                history_cache[ckey] = auto_cache(
                    (ckey, ticker),
                    lambda t=ticker: edm.fred.get_series(t).dropna().tail(20).tolist(),
                    refresh=refresh_market_data
                )
        today_econ_events = auto_cache(
            ("today_econ_events", "united states", "germany", "japan", "united kingdom", "euro area", "high"),
            lambda: edm.get_economic_events_api(
                country_list=['united states', 'germany', 'japan', 'united kingdom', 'euro area'],
                importance="high"
            ),
            refresh=refresh_market_data
        )
        if not today_econ_events.empty:
            today = pd.Timestamp.now().normalize()
            tomorrow = today + pd.Timedelta(days=1)
            mask = (today_econ_events['date'].dt.date >= today.date()) & (
                        today_econ_events['date'].dt.date <= tomorrow.date())
            today_econ_events = today_econ_events[mask][['date', 'country', 'event', 'actual', 'forecast', 'previous']]
        else:
            today_econ_events = pd.DataFrame()
        market_movers = get_market_movers(quotes_cache)

        cached_metrics = []
        for label, qkey, ckey, ticker, tlabel in chart_keys:
            cached_metrics.append((label, quotes_cache[qkey], history_cache[ckey]))

        # --- PATCH: Render snapshot metrics grouped in rows of 3 at the top ---
        def render_snapshot_grouped(metrics):
            st.markdown(
                '''
                <div style='background-color: rgba(0,0,0,0.25); padding: 1.1rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.12); margin-bottom:1.4rem;'>
                ''', unsafe_allow_html=True
            )
            for i in range(0, len(metrics), 3):
                row = metrics[i:i+3]
                cols = st.columns(len(row))
                for col, (label, quote, yvals) in zip(cols, row):
                    value = quote.get("current", "N/A")
                    delta = quote.get("change", 0)
                    # Format FX pairs with 5 decimals
                    if isinstance(value, float) and "/" in label:
                        value = f"{value:.5f}"
                    color = "#26de81" if delta and float(delta) > 0 else "#fc5c65" if delta and float(delta) < 0 else "#e7eaf0"
                    with col:
                        st.markdown(f"<div style='text-align:center;color:#e7eaf0;font-size:1.05em;'>{label}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align:center;font-size:2.4em;font-weight:bold;color:white'>{value}</div>", unsafe_allow_html=True)
                        if delta not in ("N/A", None):
                            arrow = "▲" if float(delta) > 0 else "▼"
                            st.markdown(f"<div style='text-align:center;font-size:1.1em;color:{color};padding-bottom:0.2em;'>{arrow} {delta}</div>", unsafe_allow_html=True)
                        # Sparkline (cached)
                        if yvals:
                            fig_spark = go.Figure()
                            fig_spark.add_trace(go.Scatter(
                                x=list(range(len(yvals))),
                                y=yvals,
                                mode="lines",
                                line=dict(color="#FFD600", width=2),
                                showlegend=False,
                                hoverinfo="skip",
                            ))
                            fig_spark.update_layout(
                                margin=dict(l=0, r=0, t=10, b=10),
                                height=70,
                                width=150,
                                paper_bgcolor="rgba(0,0,0,0.0)",
                                plot_bgcolor="rgba(0,0,0,0.0)",
                            )
                            fig_spark.update_xaxes(visible=False, showgrid=False, zeroline=False)
                            fig_spark.update_yaxes(visible=False, showgrid=False, zeroline=False)
                            st.plotly_chart(fig_spark, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        render_snapshot_grouped(cached_metrics)

        if isinstance(today_econ_events, pd.DataFrame) and not today_econ_events.empty:
            st.markdown("#### 📅 Today's Key Economic Releases")
            st.dataframe(today_econ_events, use_container_width=True, hide_index=True)
        if market_movers:
            st.markdown("#### 🔥 Biggest Market Moves Today")
            for label, curr, chg, pct in market_movers:
                display_name = DISPLAY_NAMES.get(label, label)
                arrow = "▲" if chg > 0 else "▼"
                st.markdown(f"**{display_name}:** {curr} ({arrow} {chg:+.2f}, {pct:+.2f}%)")

        # --- PATCH: Enrich prompt context ---
        # Prepare snapshot and news/catalyst context (use asset keys: dxy, vix, gold, oil, us10y, de10y)
        asset_keys = {
            'dxy': ['DXY', 'Dollar', 'USD'],
            'vix': ['VIX', 'Volatility'],
            'gold': ['Gold', 'XAU', 'Precious'],
            'oil': ['Oil', 'WTI', 'Crude'],
            'us10y': ['US 10Y', 'Treasury', 'Yield'],
            'de10y': ['DE 10Y', 'Bund', 'German'],
        }
        snapshot = {
            'dxy': cached_metrics[0][1], 'vix': cached_metrics[1][1],
            'gold': cached_metrics[2][1], 'oil': cached_metrics[3][1],
            'us10y': cached_metrics[4][1], 'de10y': cached_metrics[5][1],
        }
        # --- PATCH: Ensure 'gbpusd', 'eurusd', and 'uk10y' are always present in snapshot ---
        snapshot['gbpusd'] = auto_cache(("gbpusd_quote", "GBPUSD=X", "GBPUSD"), lambda: edm.get_index_quote("GBPUSD=X", "GBPUSD"), refresh=refresh_market_data)
        snapshot['eurusd'] = auto_cache(("eurusd_quote", "EURUSD=X", "EURUSD"), lambda: edm.get_index_quote("EURUSD=X", "EURUSD"), refresh=refresh_market_data)
        uk_gilt = edm.get_bond_yields().get('GB 10Y', {})
        snapshot['uk10y'] = uk_gilt
        # ---
        # --- PATCH: Unique asset_news cache keys for each instrument ---
        asset_news = {}
        for k, keywords in asset_keys.items():
            asset_news[k] = auto_cache(
                ("asset_news", k, tuple(keywords)),
                lambda kw=keywords: get_asset_news(kw),
                refresh=refresh_market_data
            )

        # Also fetch news for gbpusd and eurusd for the macro sentiment prompt
        gbp_kw = ['GBPUSD', 'Pound', 'Sterling', 'BOE', 'UK']
        eur_kw = ['EURUSD', 'Euro', 'ECB', 'EUR']
        asset_news['gbpusd'] = auto_cache(
            ("asset_news", "gbpusd", tuple(gbp_kw)),
            lambda: get_asset_news(gbp_kw), refresh=refresh_market_data)
        asset_news['eurusd'] = auto_cache(
            ("asset_news", "eurusd", tuple(eur_kw)),
            lambda: get_asset_news(eur_kw), refresh=refresh_market_data)

        # --- PATCH: Fetch and display macro analysis using richer prompt ---
        # --- PATCH: Unique cache key for macro sentiment context (avoid collisions) ---
        macro_md = load_or_fetch_macro_sentiment(
            snapshot,
            asset_news,
            today_econ_events.to_dict() if isinstance(today_econ_events, pd.DataFrame) else {},
            market_movers,
            refresh=refresh_market_data
        )

        st.markdown(macro_md, unsafe_allow_html=True)

    def display_next_week_events(self):
        """Displays next week's key economic events."""
        st.markdown("### 🗓️ Next Week's High-Impact Events")
        today = datetime.today()
        start_of_next_week = today - timedelta(days=today.weekday()) + timedelta(weeks=1)
        end_of_next_week = start_of_next_week + timedelta(days=6)

        events_df = self.economic_manager.get_economic_events_api(
            country_list=['united states', 'germany', 'japan', 'united kingdom', 'euro area'],
            importance="high"
        )
        if not events_df.empty:
            next_week_events = events_df[(events_df['date'].dt.date >= start_of_next_week.date()) & (
                        events_df['date'].dt.date <= end_of_next_week.date())]
            if not next_week_events.empty:
                next_week_events['prognosis'] = next_week_events['event'].apply(
                    lambda x: "High Volatility Expected" if any(k in x.lower() for k in
                                                                ['cpi', 'fomc', 'interest rate', 'gdp',
                                                                 'unemployment']) else "Market Moving"
                )
                st.dataframe(next_week_events[['date', 'country', 'event', 'prognosis']].rename(
                    columns={'date': 'Date', 'country': 'Country', 'event': 'Event', 'prognosis': 'Prognosis'}),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No high-impact events found for next week.")
        else:
            st.info("Could not retrieve economic events.")

    def _load_market_data(self, data_sources):
        """Helper to load all market data."""
        market_data = {}
        for pair, files in data_sources.items():
            if pair not in market_data: market_data[pair] = {}
            for tf, file_path in files.items():
                df = self.load_comprehensive_data(file_path, max_records=50)
                if df is not None and not df.empty:
                    market_data[pair][tf] = df
        return market_data

    def create_market_heatmap(self, market_data):
        st.markdown("### Multi-Timeframe Momentum Heatmap")
        heatmap_data = {pair: {tf: np.nan for tf in self.timeframes} for pair in self.supported_pairs}
        for pair, tfs_data in market_data.items():
            for tf, df in tfs_data.items():
                if 'close' in df.columns and len(df) >= 2:
                    last_close, prev_close = df['close'].iloc[-1], df['close'].iloc[-2]
                    if prev_close > 0:
                        heatmap_data[pair][tf] = ((last_close / prev_close) - 1) * 100
        heatmap_df = pd.DataFrame(heatmap_data).T[self.timeframes].dropna(how='all')
        if not heatmap_df.empty:
            fig = px.imshow(heatmap_df, text_auto=".2f%", aspect="auto", color_continuous_scale='RdYlGn',
                            labels=dict(x="Timeframe", y="Pair", color="Momentum %"),
                            title="Momentum of Most Recent Candle on Each Timeframe")
            fig.update_layout(template=st.session_state.get('chart_theme', 'plotly_dark'), height=500,
                              coloraxis_showscale=False)
            fig.update_xaxes(side="top")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to generate heatmap.")

    def create_correlation_matrix(self, market_data):
        st.markdown("### Price Correlation Matrix (Daily)")
        daily_returns = {pair: data['1D']['close'].pct_change() for pair, data in market_data.items() if
                         '1D' in data and 'close' in data['1D'].columns}
        returns_df = pd.DataFrame(daily_returns).tail(30)
        if len(returns_df.columns) >= 2:
            corr_matrix = returns_df.corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
            fig.update_layout(template=st.session_state.get('chart_theme', 'plotly_dark'), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data. Requires daily data for at least two pairs.")

    def scan_all_data_sources(self):
        data_sources = {}
        all_files = glob.glob(os.path.join(self.data_dir, "**", "*.*"), recursive=True)
        for f_path in all_files:
            for pair in self.supported_pairs:
                if pair in f_path and f_path.endswith(('.csv', '.parquet')):
                    for tf in self.timeframes:
                        if tf in f_path:
                            if pair not in data_sources: data_sources[pair] = {}
                            data_sources[pair][tf] = f_path
                            break
        return data_sources

    def load_comprehensive_data(self, file_path, max_records=None):
        try:
            df = pd.read_parquet(file_path) if file_path.endswith('.parquet') else pd.read_csv(file_path, sep=None,
                                                                                               engine='python')
            df.columns = [col.lower().strip() for col in df.columns]
            for col in ['timestamp', 'datetime', 'date']:
                if col in df.columns:
                    df.set_index(pd.to_datetime(df[col]), inplace=True)
                    df.drop(columns=[col], inplace=True)
                    break
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)

            # Enrich data with SMA for trend calculation
            if 'sma_20' not in df.columns and 'close' in df.columns and len(df) >= 20:
                df['sma_20'] = df['close'].rolling(window=20).mean()

            if max_records: df = df.tail(max_records)
            return df
        except Exception:
            return None


if __name__ == "__main__":
    dashboard = MarketOverviewDashboard()
    dashboard.run()
