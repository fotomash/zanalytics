
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Zanalyttics Dashboard", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

client = OpenAI(api_key=st.secrets["opanai_API"])

def fetch_openai_macro_sentiment():
    prompt = (
        "Generate a comprehensive intermarket sentiment analysis focused on: "
        "VIX, DXY, US10Y, and German Bonds. Format in markdown. "
        "Include current interpretation, behavior, cross-market interaction, and trader guidance. "
        "Conclude with a summary of key risks or macro pressure points."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Could not fetch macro sentiment: {e}"

def get_high_impact_news():
    keywords = ["FOMC", "CPI", "NFP", "unemployment", "GDP", "central bank", "rate hike", "inflation"]
    try:
        import requests
        response = requests.get("https://finnhub.io/api/v1/news?category=general&token=" + st.secrets["finnhub_API"])
        articles = response.json()
        filtered = [a for a in articles if any(k.lower() in a["headline"].lower() for k in keywords)]
        return filtered
    except:
        return []

st.markdown("### 🧨 High-Impact Economic Releases")
news = get_high_impact_news()
if news:
    for n in news[:5]:
        st.markdown(f"- **[{n['headline']}]({n['url']})**  
📅 <small>{n['datetime']}</small>", unsafe_allow_html=True)
else:
    st.info("No FOMC/CPI-related headlines detected.")

if st.button("📈 Fetch Full Intermarket Sentiment"):
    st.markdown(fetch_openai_macro_sentiment())


# --- Original Dashboard Below ---

#!/usr/bin/env python3
"""
Zanalyttics Market Overview Dashboard

A focused dashboard for at-a-glance market intelligence, featuring a multi-timeframe
performance heatmap and correlation analysis.
"""
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


def display_macro_microcharts(self):
    import altair as alt
    st.markdown("### 📉 Macro Asset Microcharts")
    assets = [
        ("DX-Y.NYB", "DXY"),
        ("^VIX", "VIX"),
        ("^TNX", "US10Y"),
        ("BUND.DE", "Bunds")
    ]
    cols = st.columns(2)
    for i, (ticker, label) in enumerate(assets):
        hist = self.economic_manager.get_index_quote_history(ticker, label, lookback=14)
        if hist is not None and not hist.empty:
            if "Date" not in hist.columns:
                hist = hist.reset_index()
            chart = alt.Chart(hist).mark_line().encode(
                x=alt.X("Date:T", axis=None),
                y=alt.Y("Close:Q", scale=alt.Scale(zero=False), axis=None)
            ).properties(height=80, width=160, title=label)
            cols[i % 2].altair_chart(chart, use_container_width=False)

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
        """
        Display US, Germany, Japan, and UK 10Y yields with sparklines and deltas,
        using the exact style and structure from HOME.py.
        """
        import plotly.graph_objects as go

        # Title and panel styling block (identical to HOME.py)
        st.markdown("<h5 style='text-align:center;'>🌍 10‑Year Government Bond Yields</h5>", unsafe_allow_html=True)
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

        yield_tickers = {
            "US": "DGS10",
            "Germany": "IRLTLT01DEM156N",
            "Japan": "IRLTLT01JPM156N",
            "UK": "IRLTLT01GBM156N",
        }
        # --- Get latest and previous yields exactly as in HOME.py ---
        latest_yields = {}
        previous_yields = {}
        for country, ticker in yield_tickers.items():
            try:
                series = self.economic_manager.fred.get_series(ticker).dropna()
                latest = float(series.iloc[-1])
                prev = float(series.iloc[-2]) if len(series) > 1 else None
                latest_yields[country] = round(latest, 3)
                previous_yields[country] = round(prev, 3) if prev else None
            except Exception:
                latest_yields[country] = "N/A"
                previous_yields[country] = None

        cols = st.columns(len(latest_yields))
        for i, (country, val) in enumerate(latest_yields.items()):
            prev_val = previous_yields.get(country)
            delta = None
            if prev_val is not None and val != "N/A":
                delta = round(val - prev_val, 3)
            with cols[i]:
                # Metric (value + delta)
                st.metric(country, f"{val}%" if val != 'N/A' else val, delta)

                # Sparkline directly below the metric
                try:
                    ticker = yield_tickers.get(country)
                    if ticker:
                        series = self.economic_manager.fred.get_series(ticker).dropna()
                        if len(series) >= 2:
                            yvals = series.iloc[-50:].values
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
                    pass
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
        st.set_page_config(page_title="Zanalyttics Dashboard", page_icon="🚀", layout="wide",
                           initial_sidebar_state="expanded")

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

        # Removed heatmap and correlation matrix display and separators as requested

        # self.display_finnhub_news()
        # self.display_polygon_news()
        # self.display_live_fx_quotes()
        self.display_selected_index_trends()
        self.display_next_week_events()
        self.display_news_headlines()
        st.markdown("⚠️ FX quotes and news feeds are temporarily unavailable due to API restrictions. Please check your API keys and access rights.")

    def display_selected_index_trends(self):
        import altair as alt

        st.markdown("### 📈 Key Index & Commodity Trends")

        assets = [
            ("^IXIC", "NASDAQ"),
            ("^DJI", "US30"),
            ("^GDAXI", "DAX"),
            ("GC=F", "Gold"),
            ("DX-Y.NYB", "DXY"),
            ("EURUSD=X", "EURUSD")
        ]

        def arrow(val):
            if val == "N/A" or val == 0:
                return "→"
            return "↑" if val > 0 else "↓"

        for ticker, label in assets:
            # Get chart data
            hist = self.economic_manager.get_index_quote_history(ticker, label, lookback=14)
            # Get current value and delta
            quote = self.economic_manager.get_index_quote(ticker, label)
            if hist is not None and not hist.empty:
                if quote.get("error"):
                    display_title = f"#### {label} (unavailable)"
                else:
                    val = quote.get("current", "N/A")
                    delta = quote.get("change", "N/A")
                    arr = arrow(delta)
                    display_title = f"#### {label} {val} {arr}"
                st.markdown(display_title)
                chart = alt.Chart(hist).mark_line().encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Close:Q", title=f"{label} Price", scale=alt.Scale(zero=False))
                ).properties(height=180)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption(f"No data available for {label}.")

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
        import altair as alt

        st.markdown("<h3 style='margin-bottom:0.7rem'>🌍 Market Snapshot</h3>", unsafe_allow_html=True)
        
        # Fetch data
        dxy_hist = self.economic_manager.get_index_quote_history("DX-Y.NYB", "DXY", lookback=14)
        vix_hist = self.economic_manager.get_index_quote_history("^VIX", "VIX", lookback=14)
        dxy = self.economic_manager.get_index_quote("DX-Y.NYB", "DXY")
        vix = self.economic_manager.get_index_quote("^VIX", "VIX")
        bonds = self.economic_manager.get_bond_yields()

        def arrow(val):
            if val == "N/A" or val == 0:
                return "→"
            return "⬆️" if val > 0 else "⬇️"

        def color(val):
            if val == "N/A" or val == 0:
                return "inherit"
            return "#26de81" if val > 0 else "#fc5c65"

        # Build table data with colored arrows
        snapshot = []
        for market, data in [("DXY", dxy), ("VIX", vix)]:
            if data.get('error'):
                snapshot.append({
                    "Market": market,
                    "Value": "N/A",
                    "Δ": data.get('error')
                })
            else:
                delta = data["change"]
                snap = {
                    "Market": market,
                    "Value": data["current"],  # Ensure numeric for Arrow compatibility
                    "Δ": f"<span style='color:{color(delta)}'>{delta:+.2f} {arrow(delta)}</span>"
                }
                snapshot.append(snap)
        for name, data in bonds.items():
            if data.get('error'):
                snapshot.append({
                    "Market": name,
                    "Value": "N/A",
                    "Δ": data.get('error')
                })
            else:
                snapshot.append({
                    "Market": name,
                    "Value": round(data['current'], 3),  # Numeric value, not string with %
                    "Δ": f"{data['change']:+.3f}"
                })
        # df = pd.DataFrame(snapshot)

        # Custom HTML table rendering to show styled arrows
        styled_rows = ""
        for row in snapshot:
            market = row["Market"]
            value = row["Value"]
            delta = row["Δ"]
            styled_rows += f"<tr><td>{market}</td><td>{value}</td><td>{delta}</td></tr>"

        html_table = f"""
<style>
.market-table {{
    border-collapse: collapse;
    width: 100%;
}}
.market-table th, .market-table td {{
    border: 1px solid #3a3f4b;
    padding: 8px 12px;
    text-align: left;
    font-size: 0.95rem;
}}
.market-table th {{
    background-color: #1f2c3b;
    color: #ffffff;
}}
.market-table td {{
    background-color: #1a222d;
    color: #e7eaf0;
}}
</style>
<table class="market-table">
<thead>
<tr><th>Market</th><th>Value</th><th>Δ</th></tr>
</thead>
<tbody>
{styled_rows}
</tbody>
</table>
"""
        st.markdown(html_table, unsafe_allow_html=True)

        # Add sparklines below
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### DXY Trend (14d)")
            if dxy_hist is not None:
                if "Date" not in dxy_hist.columns:
                    dxy_hist = dxy_hist.reset_index()
                chart_dxy = alt.Chart(dxy_hist).mark_line(color="#2d8cff").encode(
                    x=alt.X("Date:T", axis=alt.Axis(labels=False, ticks=False, title=None)),
                    y=alt.Y("Close:Q", scale=alt.Scale(zero=False), axis=alt.Axis(labels=False, ticks=False, title=None))
                ).properties(width=170, height=52)
                st.altair_chart(chart_dxy, use_container_width=True)
            else:
                st.write("No DXY data")
        with col2:
            st.markdown("##### VIX Trend (14d)")
            if vix_hist is not None:
                if "Date" not in vix_hist.columns:
                    vix_hist = vix_hist.reset_index()
                chart_vix = alt.Chart(vix_hist).mark_line(color="#fc5c65").encode(
                    x=alt.X("Date:T", axis=alt.Axis(labels=False, ticks=False, title=None)),
                    y=alt.Y("Close:Q", scale=alt.Scale(zero=False), axis=alt.Axis(labels=False, ticks=False, title=None))
                ).properties(width=170, height=52)
                st.altair_chart(chart_vix, use_container_width=True)
            else:
                st.write("No VIX data")

    self.display_macro_microcharts()

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
