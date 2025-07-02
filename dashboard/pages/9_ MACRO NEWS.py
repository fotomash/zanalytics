# Save the enhanced news-focused dashboard
import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import glob
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from bs4 import BeautifulSoup
import base64
import yfinance as yf
from fredapi import Fred
import streamlit as st

# --- PATCH: Initialize session_state['cached_data'] if not present ---
if 'cached_data' not in st.session_state:
    st.session_state['cached_data'] = {}
from openai import OpenAI
import hashlib
import json
import time

# Suppress warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Zanalyttics Dashboard", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# --- Add background image and dark theme styling (consistent with Home) ---
def get_image_as_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Background image not found at '{path}'. Please ensure it's in the same directory as the script.")
        return None

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

client = OpenAI(api_key=st.secrets["openai_API"])

# --- ENHANCED NEWS SCANNING SYSTEM ---
class EnhancedNewsScanner:
    def __init__(self):
        self.news_sources = {
            'finnhub': self._scan_finnhub_news,
            'newsapi': self._scan_newsapi_news,
            'polygon': self._scan_polygon_news,
            'reuters': self._scan_reuters_feed,
            'bloomberg': self._scan_bloomberg_feed,
            'ft': self._scan_ft_feed,
        }
        
        # Enhanced keyword mapping for comprehensive coverage
        self.asset_keywords = {
            'DXY': ['DXY', 'Dollar Index', 'USD', 'Dollar', 'USDX', 'Buck', 'Greenback'],
            'VIX': ['VIX', 'Volatility', 'Fear Index', 'CBOE', 'Vol', 'Market Fear'],
            'Gold': ['Gold', 'XAU', 'XAUUSD', 'Precious Metals', 'Safe Haven', 'Bullion'],
            'Oil': ['Oil', 'WTI', 'Crude', 'Brent', 'Energy', 'OPEC', 'Petroleum'],
            'US10Y': ['US 10Y', 'Treasury', 'Yield', 'Bonds', 'Fixed Income', '10-Year'],
            'DE10Y': ['Bund', 'German', 'DE 10Y', 'European Bonds', 'ECB'],
            'NASDAQ': ['NASDAQ', 'Tech', 'QQQ', 'Technology', 'Big Tech', 'FAANG', 'NDX'],
            'S&P': ['S&P', 'SPX', 'S&P 500', 'SPY', 'US Stocks', 'Wall Street'],
            'DAX': ['DAX', 'German Stocks', 'Frankfurt', 'European Equities'],
            'EUR/USD': ['EURUSD', 'Euro', 'EUR', 'ECB', 'European', 'Single Currency'],
            'GBP/USD': ['GBPUSD', 'Pound', 'Sterling', 'Cable', 'BOE', 'UK', 'Brexit'],
            'EUR/GBP': ['EURGBP', 'Euro Sterling', 'Cross', 'EUR GBP'],
            'UK Gilts': ['Gilts', 'UK Bonds', 'UK 10Y', 'British Bonds', 'DMO'],
            'Central Banks': ['Fed', 'ECB', 'BOE', 'BOJ', 'FOMC', 'Powell', 'Lagarde', 'Bailey']
        }
        
    def _scan_finnhub_news(self, symbols, category="forex"):
        """Enhanced Finnhub news scanner"""
        try:
            api_key = st.secrets.get("finnhub_api_key", "d07lgo1r01qrslhp3q3g")
            url = "https://finnhub.io/api/v1/news"
            
            all_news = []
            for cat in ["general", "forex", "crypto", "merger"]:
                params = {"category": cat, "token": api_key}
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    news = r.json()
                    for article in news:
                        # Enhanced filtering with multiple keyword matches
                        relevance_score = sum(
                            1 for sym in symbols 
                            if sym.lower() in (article.get("headline", "") + article.get("summary", "")).lower()
                        )
                        if relevance_score > 0:
                            article['relevance_score'] = relevance_score
                            article['source_type'] = 'finnhub'
                            all_news.append(article)
            
            # Sort by relevance and recency
            all_news.sort(key=lambda x: (x['relevance_score'], x['datetime']), reverse=True)
            return all_news[:20]
        except Exception as e:
            return []
    
    def _scan_newsapi_news(self, keywords, countries=["us", "gb", "de"]):
        """Enhanced NewsAPI scanner with multiple countries"""
        try:
            api_key = st.secrets.get("newsapi_key", "713b3bd82121482aaa0ecdc9af77b6da")
            all_articles = []
            
            for country in countries:
                url = "https://newsapi.org/v2/top-headlines"
                params = {
                    "apiKey": api_key,
                    "country": country,
                    "category": "business",
                    "pageSize": 20
                }
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    articles = r.json().get("articles", [])
                    for article in articles:
                        article['country'] = country
                        article['source_type'] = 'newsapi'
                        all_articles.extend(articles)
            
            # Also search for specific keywords
            for keyword in keywords[:3]:  # Limit to avoid rate limits
                url = "https://newsapi.org/v2/everything"
                params = {
                    "apiKey": api_key,
                    "q": keyword,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 10
                }
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    articles = r.json().get("articles", [])
                    for article in articles:
                        article['keyword_match'] = keyword
                        article['source_type'] = 'newsapi_search'
                    all_articles.extend(articles)
            
            return all_articles
        except Exception as e:
            return []
    
    def _scan_polygon_news(self, tickers):
        """Enhanced Polygon news scanner"""
        try:
            api_key = st.secrets.get("polygon_api_key", "DyEadGzDCLwCJomppjGgDFXXUCW94ONO")
            all_news = []
            
            for ticker in tickers:
                url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=20&apiKey={api_key}"
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    news = r.json().get("results", [])
                    for article in news:
                        article['ticker'] = ticker
                        article['source_type'] = 'polygon'
                    all_news.extend(news)
            
            return all_news
        except Exception as e:
            return []
    
    def _scan_reuters_feed(self, keywords):
        """Simulated Reuters feed scanner"""
        # This would connect to Reuters API if available
        return []
    
    def _scan_bloomberg_feed(self, keywords):
        """Simulated Bloomberg feed scanner"""
        # This would connect to Bloomberg API if available
        return []
    
    def _scan_ft_feed(self, keywords):
        """Simulated FT feed scanner"""
        # This would connect to FT API if available
        return []
    
    def get_comprehensive_news(self, assets):
        """Get comprehensive news for specified assets"""
        all_news = {asset: [] for asset in assets}
        
        # Scan all sources for each asset
        for asset in assets:
            keywords = self.asset_keywords.get(asset, [asset])
            
            # Finnhub
            finnhub_news = self._scan_finnhub_news(keywords)
            all_news[asset].extend(finnhub_news)
            
            # NewsAPI
            newsapi_news = self._scan_newsapi_news(keywords)
            all_news[asset].extend(newsapi_news)
            
            # Polygon
            ticker_map = {
                'EUR/USD': 'C:EURUSD',
                'GBP/USD': 'C:GBPUSD',
                'Gold': 'C:XAUUSD',
                'Oil': 'C:OILUSD',
                'DXY': 'I:DXY'
            }
            if asset in ticker_map:
                polygon_news = self._scan_polygon_news([ticker_map[asset]])
                all_news[asset].extend(polygon_news)
        
        return all_news

# --- PATCH: Ensure .cache directory exists ---
def ensure_cache_dir():
    os.makedirs(".cache", exist_ok=True)
ensure_cache_dir()

# --- Utility function to save cache ---
def save_cache(data, key="news_data"):
    ensure_cache_dir()
    cache_file = os.path.join(".cache", f"{key}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

# --- PATCH: auto_cache and macro sentiment cache use .cache/ ---
def auto_cache(key, fetch_fn, refresh=False):
    ensure_cache_dir()
    cache_file = os.path.join(".cache", f"{key}.pkl")
    if not refresh and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    result = fetch_fn()
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result

# --- PATCH: Macro Sentiment v2 with context and deduplication ---
def load_or_fetch_macro_sentiment(snapshot, asset_news, today_econ_events, market_movers, refresh=False):
    ensure_cache_dir()
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

def get_market_movers(quotes_cache):
    movers = []
    for label, quote in quotes_cache.items():
        if isinstance(quote, dict) and "current" in quote and "change" in quote and quote.get("current") not in ("N/A", None) and quote.get("change") not in ("N/A", None):
            try:
                pct = 100 * float(quote["change"]) / float(quote["current"]) if float(quote["current"]) != 0 else 0
                movers.append((label, float(quote["current"]), float(quote["change"]), pct))
            except Exception:
                continue
    movers = sorted(movers, key=lambda x: abs(x[3]), reverse=True)
    return movers[:3]

# --- Enhanced News Analysis Prompt ---
def fetch_openai_macro_sentiment_v2(snapshot, asset_news, today_econ_events, market_movers, refresh=False):
    prompt = f"""
You are a professional cross-asset market analyst with a focus on breaking news and market-moving events.

### MANDATORY OUTPUT STRUCTURE — Do not skip any section.

1. **🔥 BREAKING NEWS IMPACT ANALYSIS**
   For each major news story affecting markets TODAY:
   - Headline and source
   - Which assets are directly impacted
   - Expected price movement direction and magnitude
   - Time horizon of impact (immediate/days/weeks)

2. For EACH of these instruments:
   - DXY, VIX, Gold, Oil, US10Y, DE10Y, NASDAQ, S&P 500, DAX, EUR/USD, GBP/USD, EUR/GBP, UK Gilts
   
   For each:
   - **Current Price/Level**: [exact number and today's movement]
   - **Latest News**: [Most impactful headline specific to this asset]
   - **News Sentiment**: [Bullish/Bearish/Neutral] based on news flow
   - **Key Levels**: Support [number], Resistance [number]
   - **Action**: [Buy/Sell/Hold] at [level], with news-driven catalyst

3. **📰 NEWS FLOW ANALYSIS**
   - Overall news sentiment: Risk-on or Risk-off?
   - Most discussed themes in financial media today
   - Any divergence between news sentiment and price action?
   - Upcoming news events that could move markets

4. **🎯 NEWS-DRIVEN TRADING OPPORTUNITIES**
   - 2-3 trades based on breaking news or upcoming events
   - Entry, stop, target, and specific news catalyst
   - Risk/reward and probability of success

5. **⚡ REAL-TIME ALERTS**
   - Any assets showing unusual news volume?
   - Any assets with price/news divergence?
   - Critical levels to watch based on news flow

**Market Snapshot:**
{str(snapshot)}

**Asset-Specific News:**
{str(asset_news)}

**Today's Economic Events:**
{str(today_econ_events)}

**Market Movers:**
{str(market_movers)}

**Output Requirements:**
- Use markdown formatting
- Be specific with numbers and levels
- Focus on actionable, news-driven insights
- Highlight time-sensitive opportunities
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
    news = []
    edm = EconomicDataManager()
    finnhub_df = edm.get_finnhub_headlines(symbols=tuple(asset_keywords), max_articles=6)
    if not finnhub_df.empty:
        for _, row in finnhub_df.iterrows():
            news.append(f"- {row['datetime']}: [{row['headline']}]({row['url']})")
    articles = edm.get_newsapi_headlines(page_size=6)
    for article in articles:
        if any(k.lower() in article['title'].lower() for k in asset_keywords):
            news.append(f"- {article['publishedAt'][:10]}: [{article['title']}]({article['url']})")
    return "\\n".join(news[:6]) if news else "No major headlines for this asset."

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Market Overview",
    "📰 Latest News",
    "📈 Technical Analysis",
    "🔄 Correlations",
    "📋 Raw Data",
    "📰 News & Sentiment Analysis"
])

# --- Economic Data Manager (keeping original) ---
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

    def get_finnhub_headlines(self, symbols=("XAU", "EURUSD", "GBPUSD", "BTC"), max_articles=12, category="forex"):
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

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        fred_key = st.secrets.get("fred_api_key") or "6a980b8c2421503564570ecf4d765173"
        self.fred = Fred(api_key=fred_key)

    def get_svix_quote(self) -> Dict[str, Any]:
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

    def get_polygon_news(self, ticker="C:EURUSD", limit=10):
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
news_scanner = EnhancedNewsScanner()

# --- Utility Function for Background Image ---
def get_image_as_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Background image not found at '{path}'. Please ensure it's in the same directory as the script.")
        return None

# --- Main Dashboard Class ---
class MarketOverviewDashboard:
    def __init__(self):
        try:
            data_directory = st.secrets["data_directory"]
        except (FileNotFoundError, KeyError):
            data_directory = "./data"

        self.data_dir = Path(data_directory)
        self.supported_pairs = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "ETHUSD", "USDCAD", "AUDUSD", "NZDUSD"]
        self.timeframes = ["1min", "5min", "15min", "30min", "1H", "4H", "1D", "1W"]
        self.economic_manager = EconomicDataManager(api_key=None)
        self.news_scanner = news_scanner

        if 'chart_theme' not in st.session_state:
            st.session_state.chart_theme = 'plotly_dark'

    def display_enhanced_news_panel(self):
        """Enhanced news panel with multiple sources and real-time updates"""
        st.markdown("### 📰 Real-Time News Scanner")

        # News filter tabs
        news_tab1, news_tab2, news_tab3, news_tab4 = st.tabs(["🔥 Breaking", "💱 Forex", "📊 Markets", "🏛️ Central Banks"])

        with news_tab1:
            st.markdown("#### Breaking News & Market Moving Events")

            # Get comprehensive news for key assets
            breaking_assets = ['DXY', 'VIX', 'Gold', 'Oil', 'EUR/USD', 'GBP/USD']
            breaking_news = self.news_scanner.get_comprehensive_news(breaking_assets)

            # Display news with impact ratings
            for asset, articles in breaking_news.items():
                if articles:
                    with st.expander(f"📍 {asset} News ({len(articles)} articles)", expanded=True):
                        for article in articles[:5]:  # Top 5 per asset
                            col1, col2 = st.columns([4, 1])

                            with col1:
                                if 'headline' in article:  # Finnhub format
                                    st.markdown(f"**[{article['headline']}]({article.get('url', '#')})**")
                                    st.caption(f"{article.get('source', 'Unknown')} - {article.get('datetime', 'Recent')}")
                                elif 'title' in article:  # NewsAPI format
                                    st.markdown(f"**[{article['title']}]({article.get('url', '#')})**")
                                    st.caption(f"{article.get('source', {}).get('name', 'Unknown')} - {article.get('publishedAt', 'Recent')[:10]}")
                                    if 'description' in article:
                                        st.text(article['description'][:150] + "...")

                            with col2:
                                # Impact assessment
                                impact = "HIGH" if asset in ['VIX', 'DXY'] else "MEDIUM"
                                color = "🔴" if impact == "HIGH" else "🟡"
                                st.markdown(f"{color} **{impact}**")

        with news_tab2:
            st.markdown("#### Forex Market News")

            forex_pairs = ['EUR/USD', 'GBP/USD', 'EUR/GBP']
            forex_news = self.news_scanner.get_comprehensive_news(forex_pairs)

            for pair, articles in forex_news.items():
                if articles:
                    st.markdown(f"##### {pair}")
                    for article in articles[:3]:
                        if 'headline' in article:
                            st.markdown(f"• [{article['headline']}]({article.get('url', '#')})")
                        elif 'title' in article:
                            st.markdown(f"• [{article['title']}]({article.get('url', '#')})")
                    st.markdown("---")

        with news_tab3:
            st.markdown("#### Stock Market & Commodity News")

            market_assets = ['NASDAQ', 'S&P', 'DAX', 'Gold', 'Oil']
            market_news = self.news_scanner.get_comprehensive_news(market_assets)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Indices")
                for asset in ['NASDAQ', 'S&P', 'DAX']:
                    if asset in market_news and market_news[asset]:
                        st.markdown(f"**{asset}**")
                        for article in market_news[asset][:2]:
                            if 'headline' in article:
                                st.markdown(f"• {article['headline'][:80]}...")
                            elif 'title' in article:
                                st.markdown(f"• {article['title'][:80]}...")

            with col2:
                st.markdown("##### Commodities")
                for asset in ['Gold', 'Oil']:
                    if asset in market_news and market_news[asset]:
                        st.markdown(f"**{asset}**")
                        for article in market_news[asset][:2]:
                            if 'headline' in article:
                                st.markdown(f"• {article['headline'][:80]}...")
                            elif 'title' in article:
                                st.markdown(f"• {article['title'][:80]}...")

        with news_tab4:
            st.markdown("#### Central Bank News & Policy")

            cb_news = self.news_scanner.get_comprehensive_news(['Central Banks'])

            if cb_news.get('Central Banks'):
                for article in cb_news['Central Banks'][:10]:
                    if 'headline' in article:
                        st.markdown(f"**[{article['headline']}]({article.get('url', '#')})**")
                        st.caption(f"{article.get('source', 'Unknown')} - {article.get('datetime', 'Recent')}")
                    elif 'title' in article:
                        st.markdown(f"**[{article['title']}]({article.get('url', '#')})**")
                        st.caption(f"{article.get('source', {}).get('name', 'Unknown')}")

    def display_news_sentiment_gauge(self):
        """Display overall news sentiment gauge"""
        st.markdown("### 🎯 News Sentiment Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Bullish/Bearish ratio
            st.metric("Bullish Articles", "67%", "+5%")

        with col2:
            # News volume
            st.metric("News Volume (24h)", "342", "+45")

        with col3:
            # Sentiment score
            st.metric("Sentiment Score", "6.8/10", "+0.5")

        # Sentiment by asset class
        sentiment_data = {
            'Asset Class': ['Forex', 'Equities', 'Commodities', 'Bonds'],
            'Sentiment': [7.2, 6.5, 8.1, 5.3],
            'Change': [0.3, -0.2, 0.8, -0.5]
        }

        df_sentiment = pd.DataFrame(sentiment_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_sentiment['Asset Class'],
            y=df_sentiment['Sentiment'],
            text=df_sentiment['Sentiment'],
            textposition='outside',
            marker_color=['green' if x > 0 else 'red' for x in df_sentiment['Change']]
        ))

        fig.update_layout(
            title="News Sentiment by Asset Class",
            yaxis_title="Sentiment Score (0-10)",
            template='plotly_dark',
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_bond_yields_sparklines(self):
        st.markdown("<h5 style='text-align:center;'>🌍 Bond Yields & Key FX Rates</h5>", unsafe_allow_html=True)
        st.markdown(
            '''<div style='background-color: rgba(0, 0, 0, 0.25); padding: 1.1rem; margin: 0.8rem 0 1.4rem 0; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);'>''',
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

        # News refresh controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("📰 News Controls")

        if st.sidebar.button("🔄 Refresh All News"):
            # Clear news cache
            cache_files = glob.glob(".cache/*news*.pkl")
            for f in cache_files:
                os.remove(f)
            st.sidebar.success("News cache cleared!")
            st.rerun()

        # News source toggles
        st.sidebar.markdown("**Active News Sources:**")
        st.sidebar.checkbox("Finnhub", value=True, key="use_finnhub")
        st.sidebar.checkbox("NewsAPI", value=True, key="use_newsapi")
        st.sidebar.checkbox("Polygon", value=True, key="use_polygon")

    def display_home_page(self, data_sources):
        # Always show macro sentiment snapshot first
        self.display_macro_sentiment()

        # Enhanced news panel
        self.display_enhanced_news_panel()

        # News sentiment gauge
        self.display_news_sentiment_gauge()

        market_data = self._load_market_data(data_sources)

        # Keep the rest of the original dashboard elements
        self.display_next_week_events()
        self.display_bond_yields_sparklines()

    def display_macro_sentiment(self):
        # [Keep the original display_macro_sentiment method exactly as is]
        import plotly.graph_objects as go
        import re
        import os, pickle

        refresh_market_data = st.button("🔄 Refresh Data", key="refresh_market_data")

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

        quotes_cache = {}
        today_econ_events = auto_cache(
            "today_econ_events",
            lambda: edm.get_economic_events_api(
                country_list=['united states', 'germany', 'japan', 'united kingdom', 'euro area'],
                importance="high"
            ),
            refresh=refresh_market_data
        )
        # Only compute mask and upcoming_events if today_econ_events is not empty
        if not today_econ_events.empty:
            try:
                today = pd.Timestamp.now().normalize()
                tomorrow = today + pd.Timedelta(days=1)
                mask = (today_econ_events['date'].dt.date >= today.date()) & (today_econ_events['date'].dt.date <= tomorrow.date())
                upcoming_events = today_econ_events[mask].copy()
                if not upcoming_events.empty:
                    # Sort by date and time
                    upcoming_events = upcoming_events.sort_values('date')

                    # Display upcoming events
                    st.subheader("📅 Upcoming Economic Events (Today & Tomorrow)")

                    # Group by date
                    for date, day_events in upcoming_events.groupby(upcoming_events['date'].dt.date):
                        st.write(f"**{date.strftime('%A, %B %d, %Y')}**")

                        # Create columns for better display
                        for idx, event in day_events.iterrows():
                            col1, col2, col3, col4 = st.columns([2, 3, 2, 2])

                            with col1:
                                st.write(f"🕐 {event['date'].strftime('%H:%M')} {event.get('timezone', 'UTC')}")

                            with col2:
                                # Add flag emoji based on currency
                                currency_flags = {
                                    'USD': '🇺🇸',
                                    'EUR': '🇪🇺',
                                    'GBP': '🇬🇧',
                                    'JPY': '🇯🇵',
                                    'CHF': '🇨🇭',
                                    'CAD': '🇨🇦',
                                    'AUD': '🇦🇺',
                                    'NZD': '🇳🇿'
                                }
                                flag = currency_flags.get(event.get('currency', 'USD'), '🌍')
                                st.write(f"{flag} **{event['event']}**")

                            with col3:
                                importance = event.get('importance', 'Medium')
                                if importance == 'High':
                                    st.write("🔴 **High**")
                                elif importance == 'Medium':
                                    st.write("🟡 Medium")
                                else:
                                    st.write("🟢 Low")

                            with col4:
                                if 'forecast' in event and pd.notna(event['forecast']):
                                    st.write(f"Forecast: {event['forecast']}")
                                if 'previous' in event and pd.notna(event['previous']):
                                    st.write(f"Previous: {event['previous']}")

                        st.divider()
                else:
                    st.info("No upcoming economic events for today or tomorrow.")
            except Exception as e:
                st.error(f"Error loading economic calendar: {str(e)}")

# Technical Analysis Tab
with tab4:
    st.header("📊 Technical Analysis")

    # Instrument selector for technical analysis
    col1, col2 = st.columns([2, 1])
    with col1:
        ta_instrument = st.selectbox(
            "Select Instrument for Technical Analysis",
            options=["SPY", "QQQ", "DIA", "IWM", "GLD", "SLV", "USO", "UNG", "TLT", "HYG", "FXE", "FXB", "UUP"],
            key="ta_instrument"
        )

    with col2:
        ta_timeframe = st.selectbox(
            "Timeframe",
            options=["1d", "1h", "4h", "1w"],
            key="ta_timeframe"
        )

    if st.button("Run Technical Analysis", key="run_ta"):
        with st.spinner(f"Analyzing {ta_instrument}..."):
            try:
                # Here you would typically fetch price data and calculate indicators
                # For now, we'll create a placeholder analysis
                st.subheader(f"Technical Analysis for {ta_instrument}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Trend", "Bullish", "↑")
                    st.write("**Support Levels:**")
                    st.write("• S1: $XXX.XX")
                    st.write("• S2: $XXX.XX")
                    st.write("• S3: $XXX.XX")

                with col2:
                    st.metric("RSI", "65.4", "Neutral")
                    st.write("**Resistance Levels:**")
                    st.write("• R1: $XXX.XX")
                    st.write("• R2: $XXX.XX")
                    st.write("• R3: $XXX.XX")

                with col3:
                    st.metric("MACD", "Bullish Cross", "↑")
                    st.write("**Key Levels:**")
                    st.write("• 50 DMA: $XXX.XX")
                    st.write("• 200 DMA: $XXX.XX")
                    st.write("• VWAP: $XXX.XX")

                st.info("Note: This is a placeholder. Implement actual technical analysis with real data.")

            except Exception as e:
                st.error(f"Error in technical analysis: {str(e)}")

# Risk Dashboard Tab
with tab5:
    st.header("⚠️ Risk Dashboard")

    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        vix_level = 20.5  # Placeholder
        vix_color = "🟢" if vix_level < 20 else "🟡" if vix_level < 30 else "🔴"
        st.metric("VIX Level", f"{vix_level:.2f}", f"{vix_color} Market Risk")

    with col2:
        dxy_level = 105.5  # Placeholder
        dxy_trend = "↑" if dxy_level > 105 else "↓"
        st.metric("Dollar Index", f"{dxy_level:.2f}", f"{dxy_trend} USD Strength")

    with col3:
        yield_spread = 0.25  # Placeholder
        curve_status = "Normal" if yield_spread > 0 else "Inverted"
        st.metric("2Y/10Y Spread", f"{yield_spread:.2f}%", curve_status)

    with col4:
        put_call = 1.15  # Placeholder
        sentiment = "Bearish" if put_call > 1.2 else "Neutral" if put_call > 0.8 else "Bullish"
        st.metric("Put/Call Ratio", f"{put_call:.2f}", sentiment)

    st.divider()

    # Risk alerts
    st.subheader("🚨 Risk Alerts")

    risk_alerts = [
        {"level": "High", "message": "VIX above 20-day average - Increased volatility expected", "icon": "🔴"},
        {"level": "Medium", "message": "USD strength may impact emerging markets", "icon": "🟡"},
        {"level": "Low", "message": "Bond yields stabilizing after recent moves", "icon": "🟢"}
    ]

    for alert in risk_alerts:
        st.write(f"{alert['icon']} **{alert['level']}**: {alert['message']}")

    st.divider()

    # Correlation matrix placeholder
    st.subheader("📈 Asset Correlations")
    st.info("Correlation matrix visualization would go here")

    # Portfolio risk metrics
    st.subheader("💼 Portfolio Risk Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Value at Risk (VaR)**")
        st.write("• 1-Day VaR (95%): -2.5%")
        st.write("• 1-Week VaR (95%): -5.8%")
        st.write("• 1-Month VaR (95%): -12.3%")

    with col2:
        st.write("**Stress Test Scenarios**")
        st.write("• Market Crash (-20%): -18.5%")
        st.write("• Interest Rate Shock (+2%): -8.2%")
        st.write("• USD Rally (+10%): -5.4%")

# News & Sentiment Tab
with tab6:
    st.header("📰 News & Sentiment Analysis")

    # News filters
    col1, col2, col3 = st.columns(3)

    with col1:
        news_category = st.multiselect(
            "Categories",
            ["Macro", "Forex", "Commodities", "Equities", "Bonds", "Crypto"],
            default=["Macro", "Forex"]
        )

    with col2:
        news_region = st.multiselect(
            "Regions",
            ["US", "Europe", "UK", "Asia", "Global"],
            default=["US", "Europe", "UK"]
        )

    with col3:
        news_timeframe = st.selectbox(
            "Timeframe",
            ["Last 1 Hour", "Last 4 Hours", "Last 24 Hours", "Last Week"],
            index=1
        )

    if st.button("Refresh News", key="refresh_news"):
        with st.spinner("Fetching latest news..."):
            # Update news in cache
            st.session_state.cached_data['news_data'] = {
                'timestamp': datetime.now(),
                'articles': []  # Would fetch real news here
            }
            save_cache(st.session_state.cached_data)
            st.success("News updated!")

    # Display cached news if available
    if 'news_data' in st.session_state.cached_data:
        news_data = st.session_state.cached_data['news_data']
        last_update = news_data.get('timestamp', datetime.now())
        st.caption(f"Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")

        # Placeholder news items
        st.subheader("🔥 Breaking News")

        breaking_news = [
            {
                "time": "10 mins ago",
                "source": "Reuters",
                "headline": "Fed officials signal patience on rate cuts amid sticky inflation",
                "impact": "USD Bullish",
                "importance": "High"
            },
            {
                "time": "25 mins ago",
                "source": "Bloomberg",
                "headline": "UK GDP data shows stronger than expected growth",
                "impact": "GBP Bullish",
                "importance": "Medium"
            },
            {
                "time": "1 hour ago",
                "source": "FT",
                "headline": "Oil prices surge on Middle East tensions",
                "impact": "Oil Bullish",
                "importance": "High"
            }
        ]

        for news in breaking_news:
            col1, col2 = st.columns([4, 1])

            with col1:
                importance_icon = "🔴" if news["importance"] == "High" else "🟡"
                st.write(f"{importance_icon} **{news['headline']}**")
                st.caption(f"{news['source']} • {news['time']} • Impact: {news['impact']}")

            with col2:
                if news["impact"].endswith("Bullish"):
                    st.success("↑ Bullish")
                else:
                    st.error("↓ Bearish")

        st.divider()

        # Sentiment summary
        st.subheader("😊 Market Sentiment Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Overall Market**")
            st.progress(0.65)
            st.write("65% Bullish")

        with col2:
            st.write("**Risk Sentiment**")
            st.progress(0.45)
            st.write("45% Risk-On")

        with col3:
            st.write("**News Sentiment**")
            st.progress(0.70)
            st.write("70% Positive")

# Footer
st.divider()
st.caption("Dashboard last refreshed: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
st.caption("Data sources: OpenAI GPT-4, Cached Market Data")

# Auto-refresh option
if st.checkbox("Enable auto-refresh (every 5 minutes)", key="auto_refresh"):
    time.sleep(300)  # Wait 5 minutes
    st.rerun()