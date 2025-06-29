#!/usr/bin/env python3
"""
Fixed Macro Sentiment Dashboard for ncOS Integration
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import yfinance as yf
from fredapi import Fred
import base64
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')

# Load configuration
CONFIG_PATH = Path(__file__).parent / "macro_sentiment_config.yaml"
if not CONFIG_PATH.exists():
    # Create default config if missing
    default_config = {
        'data_sources': {
            'yfinance': {
                'tickers': {
                    'indices': ['^GSPC', '^DJI', '^IXIC', '^VIX'],
                    'currencies': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'DX-Y.NYB'],
                    'commodities': ['GC=F', 'CL=F', 'SI=F', 'HG=F'],
                    'crypto': ['BTC-USD', 'ETH-USD']
                }
            },
            'fred': {
                'api_key': st.secrets.get('fred_api_key', '6a980b8c2421503564570ecf4d765173'),
                'series': ['DGS10', 'DGS2', 'DFF', 'UNRATE', 'CPIAUCSL', 'GDPC1']
            },
            'newsapi': {
                'api_key': st.secrets.get('newsapi_key', '713b3bd82121482aaa0ecdc9af77b6da'),
                'country': 'gb',
                'category': 'business'
            },
            'polygon': {
                'api_key': st.secrets.get('polygon_api_key', 'DyEadGzDCLwCJomppjGgDFXXUCW94ONO')
            },
            'finnhub': {
                'api_key': st.secrets.get('finnhub_api_key', 'd07lgo1r01qrslhp3q3g')
            }
        },
        'analysis': {
            'lookback_days': 14,
            'update_frequency': 3600
        }
    }
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(default_config, f)

# Load config
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

def get_image_as_base64(path):
    """Reads an image file and returns its base64 encoded string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return None

class EconomicDataManager:
    def __init__(self, config: Dict):
        self.config = config
        fred_key = config['data_sources']['fred']['api_key']
        self.fred = Fred(api_key=fred_key)
    
    def get_short_volume_data(self, ticker="AAPL", limit=7):
        api_key = self.config['data_sources']['polygon']['api_key']
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
    
    def get_finnhub_headlines(self, symbols=("XAU", "EURUSD", "GBPUSD", "BTC"), max_articles=12, category="forex"):
        API_KEY = self.config['data_sources']['finnhub']['api_key']
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
                lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M")
            )
            df = df[["datetime", "source", "headline", "url"]]
            return df
        except Exception as e:
            st.error(f"Finnhub news fetch failed: {e}")
            return pd.DataFrame()
    
    def get_newsapi_headlines(self, country=None, category=None, page_size=10):
        if country is None:
            country = self.config['data_sources']['newsapi']['country']
        if category is None:
            category = self.config['data_sources']['newsapi']['category']
            
        API_KEY = self.config['data_sources']['newsapi']['api_key']
        url = "https://newsapi.org/v2/top-headlines"
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
    
    def get_bond_yields(self) -> Dict[str, Dict[str, Any]]:
        """Fetch latest 10-year bond yields from FRED."""
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
    
    def get_polygon_fx_snapshot(self, pairs=("C:EURUSD", "C:GBPUSD", "C:USDJPY")):
        api_key = self.config['data_sources']['polygon']['api_key']
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
                        "time": datetime.utcfromtimestamp(data.get("updated")).strftime("%H:%M:%S") if data.get("updated") else ""
                    }
            except Exception as e:
                results[p] = {"error": str(e)}
        
        return pd.DataFrame.from_dict(results, orient="index")

class MarketOverviewDashboard:
    def __init__(self):
        self.config = CONFIG
        try:
            data_directory = st.secrets["data_directory"]
        except (FileNotFoundError, KeyError):
            data_directory = "./data"
        
        self.data_dir = Path(data_directory)
        self.supported_pairs = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "ETHUSD", "USDCAD", "AUDUSD", "NZDUSD"]
        self.timeframes = ["1min", "5min", "15min", "30min", "1H", "4H", "1D", "1W"]
        
        self.economic_manager = EconomicDataManager(self.config)
        
        if 'chart_theme' not in st.session_state:
            st.session_state.chart_theme = 'plotly_dark'
    
    def run(self):
        st.set_page_config(page_title="Zanalyttics Dashboard", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")
        
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
        
        if not self.data_dir.exists():
            st.error(f"Data directory not found at: `{self.data_dir}`")
            st.info("Please create this directory or configure the correct path in your `.streamlit/secrets.toml` file.")
            st.code('data_directory = "/path/to/your/data"')
            return
        
        with st.spinner("🛰️ Scanning all data sources..."):
            data_sources = self.scan_all_data_sources()
        
        self.create_sidebar(data_sources)
        self.display_home_page(data_sources)
    
    def create_sidebar(self, data_sources):
        st.sidebar.title("Data Status")
        summary = [{'Pair': pair, 'Timeframes Found': len(data_sources.get(pair, {}))} for pair in self.supported_pairs if data_sources.get(pair)]
        if summary:
            st.sidebar.dataframe(pd.DataFrame(summary).set_index('Pair'), use_container_width=True)
        else:
            st.sidebar.warning("No data found.")
    
    def display_home_page(self, data_sources):
        self.display_macro_sentiment()
        
        market_data = self._load_market_data(data_sources)
        
        self.display_selected_index_trends()
        self.display_next_week_events()
        self.display_news_headlines()
    
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
                    "Value": data["current"],
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
                    "Value": round(data['current'], 3),
                    "Δ": f"{data['change']:+.3f}"
                })
        
        # Custom HTML table rendering
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
            color: #ffff;
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
            st.markdown("#### DXY Trend (14d)")
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
            st.markdown("#### VIX Trend (14d)")
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
            hist = self.economic_manager.get_index_quote_history(ticker, label, lookback=14)
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
        articles = self.economic_manager.get_newsapi_headlines(page_size=5)
        if articles:
            for article in articles:
                st.markdown(f"**[{article['title']}]({article['url']})**")
                st.caption(f"{article['source']['name']} — {article['publishedAt'][:10]}")
                st.markdown(f"{article['description']}")
                st.markdown("---")
        else:
            st.info("No recent news available.")
    
    def display_next_week_events(self):
        """Placeholder for economic events - requires valid API"""
        st.markdown("### 🗓️ Economic Calendar")
        st.info("Economic events calendar requires Trading Economics API access.")
    
    def scan_all_data_sources(self):
        import glob
        data_sources = {}
        all_files = glob.glob(os.path.join(self.data_dir, "**", "*.*"), recursive=True)
        for f_path in all_files:
            for pair in self.supported_pairs:
                if pair in f_path and f_path.endswith(('.csv', '.parquet')):
                    for tf in self.timeframes:
                        if tf in f_path:
                            if pair not in data_sources:
                                data_sources[pair] = {}
                            data_sources[pair][tf] = f_path
                            break
        return data_sources
    
    def load_comprehensive_data(self, file_path, max_records=None):
        try:
            df = pd.read_parquet(file_path) if file_path.endswith('.parquet') else pd.read_csv(file_path, sep=None, engine='python')
            df.columns = [col.lower().strip() for col in df.columns]
            
            for col in ['timestamp', 'datetime', 'date']:
                if col in df.columns:
                    df.set_index(pd.to_datetime(df[col]), inplace=True)
                    df.drop(columns=[col], inplace=True)
                    break
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            
            if 'sma_20' not in df.columns and 'close' in df.columns and len(df) >= 20:
                df['sma_20'] = df['close'].rolling(window=20).mean()
            
            if max_records:
                df = df.tail(max_records)
            
            return df
        except Exception:
            return None
    
    def _load_market_data(self, data_sources):
        """Helper to load all market data."""
        market_data = {}
        for pair, files in data_sources.items():
            if pair not in market_data:
                market_data[pair] = {}
            for tf, file_path in files.items():
                df = self.load_comprehensive_data(file_path, max_records=50)
                if df is not None and not df.empty:
                    market_data[pair][tf] = df
        return market_data

if __name__ == "__main__":
    dashboard = MarketOverviewDashboard()
    dashboard.run()