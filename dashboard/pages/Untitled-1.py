

import os
import json
import pickle
import hashlib
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from fredapi import Fred
from openai import OpenAI
import requests
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Professional Macro Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: rgba(28, 31, 36, 0.8);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 31, 36, 0.8);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .market-card {
        background-color: rgba(28, 31, 36, 0.9);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .alert-box {
        background-color: rgba(255, 75, 75, 0.1);
        border: 1px solid rgba(255, 75, 75, 0.3);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: rgba(0, 255, 127, 0.1);
        border: 1px solid rgba(0, 255, 127, 0.3);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize clients
logging.basicConfig(level=logging.INFO)
client = OpenAI(api_key=st.secrets.get("openai_API"))
fred = Fred(api_key=st.secrets.get("fred_api_key", "6a980b8c2421503564570ecf4d765173"))

# Professional asset mapping
PROFESSIONAL_SYMBOL_MAP = {
    # FX Majors
    "dxy_quote": "DX-Y.NYB",
    "eurusd_quote": "EURUSD=X", 
    "gbpusd_quote": "GBPUSD=X",
    "usdjpy_quote": "USDJPY=X",
    "audusd_quote": "AUDUSD=X",
    "usdcad_quote": "USDCAD=X",
    "usdchf_quote": "USDCHF=X",
    "nzdusd_quote": "NZDUSD=X",
    "eurgbp_quote": "EURGBP=X",
    
    # Volatility
    "vix_quote": "^VIX",
    "vvix_quote": "^VVIX",
    "ovx_quote": "^OVX",
    "gvz_quote": "^GVZ",
    
    # Commodities
    "gold_quote": "GC=F",
    "silver_quote": "SI=F",
    "oil_quote": "CL=F",
    "natgas_quote": "NG=F",
    "copper_quote": "HG=F",
    
    # Yields (will use FRED)
    "us10y_quote": "^TNX",
    "us2y_quote": "^IRX",
    "us30y_quote": "^TYX",
    "de10y_quote": "DE10Y-DE.BE",
    "uk10y_quote": "GB10Y-GB.BE",
    "jp10y_quote": "JP10Y-JP.BE",
    
    # Indices
    "spx_quote": "^GSPC",
    "nasdaq_quote": "^IXIC",
    "dji_quote": "^DJI",
    "rut_quote": "^RUT",
    "dax_quote": "^GDAXI",
    "ftse_quote": "^FTSE",
    "nikkei_quote": "^N225",
    "hsi_quote": "^HSI",
    
    # Crypto
    "btc_quote": "BTC-USD",
    "eth_quote": "ETH-USD"
}

# Enhanced display names
ENHANCED_DISPLAY_NAMES = {
    'dxy_quote': 'DXY',
    'eurusd_quote': 'EUR/USD',
    'gbpusd_quote': 'GBP/USD',
    'usdjpy_quote': 'USD/JPY',
    'audusd_quote': 'AUD/USD',
    'usdcad_quote': 'USD/CAD',
    'usdchf_quote': 'USD/CHF',
    'nzdusd_quote': 'NZD/USD',
    'eurgbp_quote': 'EUR/GBP',
    'vix_quote': 'VIX',
    'vvix_quote': 'VVIX',
    'ovx_quote': 'OVX (Oil VIX)',
    'gvz_quote': 'GVZ (Gold VIX)',
    'gold_quote': 'Gold',
    'silver_quote': 'Silver',
    'oil_quote': 'WTI Crude',
    'natgas_quote': 'Natural Gas',
    'copper_quote': 'Copper',
    'us10y_quote': 'US 10Y',
    'us2y_quote': 'US 2Y',
    'us30y_quote': 'US 30Y',
    'de10y_quote': 'DE 10Y',
    'uk10y_quote': 'UK 10Y',
    'jp10y_quote': 'JP 10Y',
    'spx_quote': 'S&P 500',
    'nasdaq_quote': 'NASDAQ',
    'dji_quote': 'Dow Jones',
    'rut_quote': 'Russell 2000',
    'dax_quote': 'DAX',
    'ftse_quote': 'FTSE 100',
    'nikkei_quote': 'Nikkei',
    'hsi_quote': 'Hang Seng',
    'btc_quote': 'Bitcoin',
    'eth_quote': 'Ethereum'
}

# Price validation thresholds
PRICE_THRESHOLDS = {
    "dxy_quote": (90, 120),
    "vix_quote": (10, 80),
    "gold_quote": (1500, 3000),
    "oil_quote": (20, 150),
    "us10y_quote": (0, 10),
    "de10y_quote": (-1, 10),
    "uk10y_quote": (0, 10),
    "gbpusd_quote": (1.0, 1.6),
    "eurusd_quote": (0.9, 1.5),
    "usdjpy_quote": (100, 160),
    "spx_quote": (3000, 6000),
    "nasdaq_quote": (10000, 20000),
    "btc_quote": (10000, 100000)
}

# Cache directory setup
def ensure_cache_dir():
    os.makedirs(".cache", exist_ok=True)
    os.makedirs(".cache/prices", exist_ok=True)
    os.makedirs(".cache/news", exist_ok=True)
    os.makedirs(".cache/analysis", exist_ok=True)

ensure_cache_dir()

# Professional price fetching with validation
def get_validated_prices(symbol_map, validate=True):
    """Fetch prices with professional-grade validation"""
    prices = {}
    errors = []
    
    with st.spinner("🔄 Fetching real-time market data..."):
        progress_bar = st.progress(0)
        total = len(symbol_map)
        
        for idx, (name, ticker) in enumerate(symbol_map.items()):
            try:
                if "10y" in name.lower() or "2y" in name.lower() or "30y" in name.lower():
                    # Use FRED for bond yields
                    if name == "us10y_quote":
                        series = fred.get_series("DGS10")
                    elif name == "us2y_quote":
                        series = fred.get_series("DGS2")
                    elif name == "us30y_quote":
                        series = fred.get_series("DGS30")
                    elif name == "de10y_quote":
                        series = fred.get_series("IRLTLT01DEM156N")
                    elif name == "uk10y_quote":
                        series = fred.get_series("IRLTLT01GBM156N")
                    else:
                        continue
                        
                    if series is not None and len(series) > 0:
                        current = float(series.dropna().iloc[-1])
                        prev = float(series.dropna().iloc[-2]) if len(series.dropna()) > 1 else current
                        prices[name] = {
                            "current": round(current, 3),
                            "change": round(current - prev, 3),
                            "pct_change": round((current - prev) / prev * 100, 2) if prev != 0 else 0,
                            "timestamp": datetime.now()
                        }
                else:
                    # Use yfinance for everything else
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    hist = ticker_obj.history(period="2d")
                    
                    if len(hist) >= 1:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) >= 2 else current
                        
                        prices[name] = {
                            "current": round(current, 4),
                            "change": round(current - prev, 4),
                            "pct_change": round((current - prev) / prev * 100, 2) if prev != 0 else 0,
                            "timestamp": datetime.now(),
                            "volume": hist['Volume'].iloc[-1] if 'Volume' in hist else None
                        }
                        
            except Exception as e:
                prices[name] = {"current": "N/A", "error": str(e)}
                errors.append(f"{name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / total)
        
        progress_bar.empty()
    
    # Validate prices
    if validate:
        validation_issues = validate_prices(prices)
        if validation_issues:
            with st.expander("⚠️ Price Validation Warnings", expanded=False):
                for issue in validation_issues:
                    st.warning(issue)
    
    return prices, errors

def validate_prices(snapshot):
    """Professional price validation"""
    warnings = []
    now = datetime.now().isoformat()
    
    for key, (low, high) in PRICE_THRESHOLDS.items():
        if key in snapshot:
            price_data = snapshot.get(key, {})
            price = price_data.get("current") if isinstance(price_data, dict) else price_data
            
            if isinstance(price, (int, float)):
                if not (low <= price <= high):
                    warnings.append(
                        f"[{now}] ⚠️ {ENHANCED_DISPLAY_NAMES.get(key, key)} = {price} "
                        f"outside expected range ({low}-{high})"
                    )
            elif price == "N/A":
                warnings.append(f"[{now}] ⚠️ {ENHANCED_DISPLAY_NAMES.get(key, key)} data unavailable")
    
    return warnings

# Enhanced news aggregation
def get_comprehensive_news(assets_focus=None):
    """Aggregate news from multiple sources with smart filtering"""
    all_news = {}
    
    # Define search keywords for each asset class
    search_terms = {
        "fx": ["dollar", "forex", "currency", "DXY", "EUR/USD", "GBP/USD", "central bank"],
        "commodities": ["gold", "oil", "crude", "WTI", "precious metals", "copper", "commodity"],
        "rates": ["bond", "yield", "treasury", "interest rate", "Fed", "ECB", "BOE", "BOJ"],
        "equities": ["stock", "S&P", "NASDAQ", "equity", "earnings", "market"],
        "macro": ["inflation", "CPI", "GDP", "unemployment", "PMI", "retail sales", "FOMC"]
    }
    
    try:
        # Finnhub news
        finnhub_key = st.secrets.get("finnhub_api_key", "d07lgo1r01qrslhp3q3g")
        for category in ["general", "forex", "crypto"]:
            url = f"https://finnhub.io/api/v1/news?category={category}&token={finnhub_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                articles = response.json()
                for article in articles[:20]:  # Limit to recent articles
                    # Smart categorization
                    headline = article.get("headline", "").lower()
                    summary = article.get("summary", "").lower()
                    content = headline + " " + summary
                    
                    categories = []
                    for cat, keywords in search_terms.items():
                        if any(kw.lower() in content for kw in keywords):
                            categories.append(cat)
                    
                    if categories:
                        for cat in categories:
                            if cat not in all_news:
                                all_news[cat] = []
                            all_news[cat].append({
                                "headline": article.get("headline"),
                                "summary": article.get("summary"),
                                "url": article.get("url"),
                                "datetime": datetime.fromtimestamp(article.get("datetime", 0)),
                                "source": "Finnhub"
                            })
        
        # NewsAPI
        newsapi_key = st.secrets.get("newsapi_key", "713b3bd82121482aaa0ecdc9af77b6da")
        for query in ["forex", "commodity", "federal reserve", "ECB", "inflation"]:
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={newsapi_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get("articles", [])[:10]:
                    # Categorize
                    title = article.get("title", "").lower()
                    desc = article.get("description", "").lower()
                    content = title + " " + desc
                    
                    categories = []
                    for cat, keywords in search_terms.items():
                        if any(kw.lower() in content for kw in keywords):
                            categories.append(cat)
                    
                    if categories:
                        for cat in categories:
                            if cat not in all_news:
                                all_news[cat] = []
                            all_news[cat].append({
                                "headline": article.get("title"),
                                "summary": article.get("description"),
                                "url": article.get("url"),
                                "datetime": datetime.fromisoformat(article.get("publishedAt", "").replace("Z", "")),
                                "source": "NewsAPI"
                            })
    
    except Exception as e:
        st.error(f"News aggregation error: {str(e)}")
    
    # Sort by datetime and remove duplicates
    for category in all_news:
        # Remove duplicates based on headline
        seen = set()
        unique_news = []
        for article in sorted(all_news[category], key=lambda x: x["datetime"], reverse=True):
            if article["headline"] not in seen:
                seen.add(article["headline"])
                unique_news.append(article)
        all_news[category] = unique_news[:15]  # Keep top 15 per category
    
    return all_news

# Professional macro sentiment analysis
def generate_professional_macro_analysis(snapshot, news_data, econ_events):
    """Generate Bloomberg-style macro analysis"""
    
    # Prepare the summary table
    summary_table_rows = []
    instruments = [
        ("DXY", "dxy_quote", False),
        ("EUR/USD", "eurusd_quote", False),
        ("GBP/USD", "gbpusd_quote", False),
        ("EUR/GBP", "eurgbp_quote", False),
        ("USD/JPY", "usdjpy_quote", False),
        ("Gold", "gold_quote", False),
        ("WTI Oil", "oil_quote", False),
        ("US 10Y", "us10y_quote", True),
        ("UK 10Y", "uk10y_quote", True),
        ("DE 10Y", "de10y_quote", True),
        ("S&P 500", "spx_quote", False),
        ("VIX", "vix_quote", False),
    ]
    
    for display, key, is_yield in instruments:
        if key in snapshot:
            data = snapshot[key]
            if isinstance(data, dict):
                price = data.get("current", "N/A")
                change = data.get("change", 0)
                pct = data.get("pct_change", 0)
                
                if price != "N/A":
                    if is_yield:
                        price_str = f"{price:.3f}%"
                        change_str = f"{change:+.3f}bps"
                    else:
                        if "USD" in display or "GBP" in display or "EUR" in display:
                            price_str = f"{price:.5f}"
                            change_str = f"{change:+.5f}"
                        else:
                            price_str = f"{price:.2f}"
                            change_str = f"{change:+.2f}"
                    
                    trend = "↑" if change > 0 else "↓" if change < 0 else "→"
                    
                    # Determine sentiment
                    if abs(pct) > 1:
                        sentiment = "Bullish" if change > 0 else "Bearish"
                    else:
                        sentiment = "Neutral"
                    
                    # Get relevant news context
                    news_context = "Monitoring market flows"
                    if "fx" in news_data and news_data["fx"]:
                        for article in news_data["fx"][:3]:
                            if display.lower().replace("/", "") in article["headline"].lower():
                                news_context = article["headline"][:60] + "..."
                                break
                    
                    summary_table_rows.append(
                        f"| {display} | {price_str} | {change_str} ({pct:+.1f}%) | "
                        f"{trend} {sentiment} | {news_context} |"
                    )
    
    # Format the complete prompt
    summary_table = "\\n".join(summary_table_rows)
    
    # Prepare economic events summary
    events_summary = ""
    if not econ_events.empty:
        top_events = econ_events.head(5)
        events_list = []
        for _, event in top_events.iterrows():
            events_list.append(
                f"• {event['country']}: {event['event']} "
                f"(Forecast: {event.get('forecast', 'N/A')}, Previous: {event.get('previous', 'N/A')})"
            )
        events_summary = "\\n".join(events_list)
    
    prompt = f"""You are a senior macro strategist at a top-tier investment bank writing the morning note for institutional clients.

## 📊 Macro Sentiment Summary – {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC

### 🧾 Key Cross-Asset Dashboard

| Instrument | Current Price | Change | Trend | Key Macro Context |
|----|----|----|----|----|
{summary_table}

### 📅 Today's Key Economic Releases
{events_summary if events_summary else "No high-impact releases scheduled"}

Write a concise but comprehensive macro analysis following this structure:

1. **🔍 Market Movers & Macro Drivers** 
   - FX: Focus on DXY, EUR, GBP, JPY flows and central bank expectations
   - Rates: Yield curve dynamics and monetary policy implications  
   - Commodities: Energy and precious metals with geopolitical context
   - Risk: VIX and cross-asset volatility regime

2. **🧠 Key Themes Driving Markets**
   - Identify 3 main macro themes
   - Connect price action to fundamental drivers
   - Highlight any divergences or regime changes

3. **⚡ Trade Ideas & Risk Scenarios**
   - 2-3 actionable trade ideas with clear rationale
   - Key risk events and levels to watch
   - Correlation breaks or unusual flows

Use professional language, cite specific levels, and focus on what matters for trading today.
Be direct, quantitative, and actionable. No generic statements."""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Analysis generation error: {str(e)}"

# Economic calendar integration
def get_economic_calendar():
    """Fetch comprehensive economic calendar"""
    try:
        # Trading Economics API
        api_key = st.secrets.get("trading_economics_api_key", "1750867cdfc34c6:288nxdz64y932qq")
        countries = ["united states", "euro area", "united kingdom", "japan", "china", "germany"]
        
        all_events = []
        for country in countries:
            url = f"https://api.tradingeconomics.com/calendar/country/{country}?c={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                events = response.json()
                all_events.extend(events)
        
        # Convert to DataFrame
        if all_events:
            df = pd.DataFrame(all_events)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter for next 7 days
            today = pd.Timestamp.now()
            week_ahead = today + pd.Timedelta(days=7)
            df = df[(df['date'] >= today) & (df['date'] <= week_ahead)]
            
            # Sort by importance and date
            importance_map = {"Low": 1, "Medium": 2, "High": 3}
            df['importance_num'] = df['importance'].map(importance_map).fillna(1)
            df = df.sort_values(['date', 'importance_num'], ascending=[True, False])
            
            return df[['date', 'country', 'event', 'actual', 'forecast', 'previous', 'importance']]
        
    except Exception as e:
        st.error(f"Calendar fetch error: {str(e)}")
    
    return pd.DataFrame()

# Technical analysis overlay
def calculate_technical_indicators(ticker, period="1mo"):
    """Calculate professional technical indicators"""
    try:
        data = yf.download(ticker, period=period, interval="1h")
        if data.empty:
            return None
        
        # Calculate indicators
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal'] = data['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_upper'] = data['BB_middle'] + 2 * bb_std
        data['BB_lower'] = data['BB_middle'] - 2 * bb_std
        
        # Support/Resistance
        data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['R1'] = 2 * data['Pivot'] - data['Low']
        data['S1'] = 2 * data['Pivot'] - data['High']
        data['R2'] = data['Pivot'] + (data['High'] - data['Low'])
        data['S2'] = data['Pivot'] - (data['High'] - data['Low'])
        
        return data
        
    except Exception as e:
        return None

# Main dashboard
def main():
    st.title("🌍 Professional Macro Trading Dashboard")
    st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        
        # Asset selection
        selected_assets = st.multiselect(
            "Select Assets to Monitor",
            options=list(ENHANCED_DISPLAY_NAMES.values()),
            default=["DXY", "EUR/USD", "GBP/USD", "Gold", "WTI Crude", "US 10Y", "S&P 500", "VIX"]
        )
        
        # Refresh controls
        auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
        refresh_button = st.button("🔄 Refresh All Data", type="primary")
        
        # Theme selection
        theme = st.selectbox("Chart Theme", ["plotly_dark", "plotly_white", "seaborn"])
        
        st.divider()
        
        # Market hours indicator
        st.subheader("🕐 Market Hours")
        now = datetime.now()
        
        # Check market hours
        markets = {
            "London": (8, 16),  # 8 AM - 4 PM GMT
            "New York": (13, 21),  # 8 AM - 4 PM EST (13-21 GMT)
            "Tokyo": (0, 6),  # 9 AM - 3 PM JST (0-6 GMT)
            "Sydney": (22, 4)  # 10 AM - 4 PM AEDT (22-4 GMT)
        }
        
        for market, (open_hour, close_hour) in markets.items():
            if close_hour < open_hour:  # Handles Sydney crossing midnight
                is_open = now.hour >= open_hour or now.hour < close_hour
            else:
                is_open = open_hour <= now.hour < close_hour
            
            status = "🟢 OPEN" if is_open else "🔴 CLOSED"
            st.write(f"{market}: {status}")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Market Overview",
        "📈 Technical Analysis", 
        "📰 News & Sentiment",
        "📅 Economic Calendar",
        "⚠️ Risk Dashboard",
        "🔍 Deep Dive Analysis",
        "📋 Trade Ideas"
    ])
    
    # Fetch all data
    if refresh_button or auto_refresh:
        st.cache_data.clear()
    
    # Get prices with progress indication
    prices, price_errors = get_validated_prices(PROFESSIONAL_SYMBOL_MAP)
    
    # Filter based on selected assets
    filtered_prices = {}
    reverse_display_map = {v: k for k, v in ENHANCED_DISPLAY_NAMES.items()}
    for asset in selected_assets:
        key = reverse_display_map.get(asset)
        if key and key in prices:
            filtered_prices[key] = prices[key]
    
    # Tab 1: Market Overview
    with tab1:
        # Professional macro analysis
        st.header("🎯 Professional Macro Analysis")
        
        # Get comprehensive data
        news_data = get_comprehensive_news()
        econ_events = get_economic_calendar()
        
        # Generate analysis
        analysis = generate_professional_macro_analysis(filtered_prices, news_data, econ_events)
        
        # Display in professional format
        st.markdown('<div class="market-card">', unsafe_allow_html=True)
        st.markdown(analysis)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Market snapshot grid
        st.header("📊 Real-Time Market Snapshot")
        
        # Create responsive grid
        cols_per_row = 4
        asset_items = list(filtered_prices.items())
        
        for i in range(0, len(asset_items), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(asset_items):
                    key, data = asset_items[i + j]
                    display_name = ENHANCED_DISPLAY_NAMES.get(key, key)
                    
                    with col:
                        if isinstance(data, dict) and data.get("current") != "N/A":
                            current = data["current"]
                            change = data.get("change", 0)
                            pct = data.get("pct_change", 0)
                            
                            # Format based on asset type
                            if "quote" in key and ("usd" in key or "eur" in key or "gbp" in key):
                                current_str = f"{current:.5f}"
                            elif "10y" in key or "2y" in key:
                                current_str = f"{current:.3f}%"
                            else:
                                current_str = f"{current:,.2f}"
                            
                            # Color coding
                            if change > 0:
                                delta_color = "inverse"
                            elif change < 0:
                                delta_color = "normal" 
                            else:
                                delta_color = "off"
                            
                            st.metric(
                                label=display_name,
                                value=current_str,
                                delta=f"{change:+.2f} ({pct:+.1f}%)",
                                delta_color=delta_color
                            )
                            
                            # Mini sparkline
                            try:
                                ticker = PROFESSIONAL_SYMBOL_MAP.get(key)
                                if ticker and "10y" not in key:
                                    hist = yf.Ticker(ticker).history(period="1d", interval="5m")
                                    if not hist.empty:
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=hist.index,
                                            y=hist['Close'],
                                            mode='lines',
                                            line=dict(color='#00ff00' if change > 0 else '#ff0000', width=2),
                                            showlegend=False
                                        ))
                                        fig.update_layout(
                                            height=80,
                                            margin=dict(l=0, r=0, t=0, b=0),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            xaxis=dict(showgrid=False, showticklabels=False),
                                            yaxis=dict(showgrid=False, showticklabels=False)
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            except:
                                pass
                        else:
                            st.metric(display_name, "N/A", "No data")
    
    # Tab 2: Technical Analysis
    with tab2:
        st.header("📈 Advanced Technical Analysis")
        
        # Asset selector
        ta_col1, ta_col2 = st.columns([3, 1])
        with ta_col1:
            ta_asset = st.selectbox(
                "Select Asset for Technical Analysis",
                options=selected_assets
            )
        with ta_col2:
            ta_period = st.selectbox(
                "Time Period",
                options=["1d", "5d", "1mo", "3mo"],
                index=2
            )
        
        # Get the ticker
        ta_key = reverse_display_map.get(ta_asset)
        ta_ticker = PROFESSIONAL_SYMBOL_MAP.get(ta_key) if ta_key else None
        
        if ta_ticker and ta_ticker not in ["DGS10", "DGS2", "DGS30"]:  # Skip FRED series
            # Calculate indicators
            ta_data = calculate_technical_indicators(ta_ticker, ta_period)
            
            if ta_data is not None and not ta_data.empty:
                # Main price chart with indicators
                fig = go.Figure()
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=ta_data.index,
                    open=ta_data['Open'],
                    high=ta_data['High'],
                    low=ta_data['Low'],
                    close=ta_data['Close'],
                    name='Price'
                ))
                
                # Moving averages
                fig.add_trace(go.Scatter(
                    x=ta_data.index,
                    y=ta_data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=ta_data.index,
                    y=ta_data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ))
                
                # Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=ta_data.index,
                    y=ta_data['BB_upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=ta_data.index,
                    y=ta_data['BB_lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ))
                
                fig.update_layout(
                    title=f"{ta_asset} - Price Action & Indicators",
                    height=600,
                    template=theme,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators dashboard
                ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                
                latest = ta_data.iloc[-1]
                
                with ind_col1:
                    rsi = latest['RSI']
                    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("RSI", f"{rsi:.1f}", rsi_signal)
                
                with ind_col2:
                    macd_hist = latest['MACD'] - latest['Signal']
                    macd_signal = "Bullish" if macd_hist > 0 else "Bearish"
                    st.metric("MACD Histogram", f"{macd_hist:.3f}", macd_signal)
                
                with ind_col3:
                    bb_position = (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
                    bb_signal = "Upper Band" if bb_position > 0.8 else "Lower Band" if bb_position < 0.2 else "Middle"
                    st.metric("BB Position", f"{bb_position:.1%}", bb_signal)
                
                with ind_col4:
                    pivot = latest['Pivot']
                    current = latest['Close']
                    pivot_signal = "Above Pivot" if current > pivot else "Below Pivot"
                    st.metric("Pivot Point", f"{pivot:.2f}", pivot_signal)
                
                # Support/Resistance levels
                st.subheader("🎯 Key Levels")
                levels_col1, levels_col2 = st.columns(2)
                
                with levels_col1:
                    st.markdown("**Resistance Levels**")
                    st.write(f"R2: {latest['R2']:.2f}")
                    st.write(f"R1: {latest['R1']:.2f}")
                    st.write(f"Pivot: {latest['Pivot']:.2f}")
                
                with levels_col2:
                    st.markdown("**Support Levels**")
                    st.write(f"Pivot: {latest['Pivot']:.2f}")
                    st.write(f"S1: {latest['S1']:.2f}")
                    st.write(f"S2: {latest['S2']:.2f}")
    
    # Tab 3: News & Sentiment
    with tab3:
        st.header("📰 Market News & Sentiment Analysis")
        
        # Category selector
        news_categories = list(news_data.keys())
        selected_category = st.selectbox(
            "Select News Category",
            options=news_categories,
            format_func=lambda x: x.upper()
        )
        
        if selected_category in news_data:
            articles = news_data[selected_category]
            
            if articles:
                # Display articles in cards
                for article in articles[:10]:
                    with st.expander(f"📰 {article['headline'][:80]}...", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{article['headline']}**")
                            if article.get('summary'):
                                st.write(article['summary'][:200] + "...")
                            st.markdown(f"[Read Full Article]({article['url']})")
                        
                        with col2:
                            st.caption(f"Source: {article['source']}")
                            st.caption(f"Time: {article['datetime'].strftime('%H:%M')}")
                            
                            # Sentiment indicator (would need actual sentiment analysis)
                            sentiment = "Neutral"  # Placeholder
                            if sentiment == "Positive":
                                st.success("🟢 Positive")
                            elif sentiment == "Negative":
                                st.error("🔴 Negative")
                            else:
                                st.info("🟡 Neutral")
            else:
                st.info("No news available for this category")
        
        # Market sentiment summary
        st.subheader("🎭 Overall Market Sentiment")
        
        sent_col1, sent_col2, sent_col3 = st.columns(3)
        
with sent_col1:
            # VIX-based fear gauge
            vix_data = prices.get("vix_quote", {})
            if isinstance(vix_data, dict) and vix_data.get("current") != "N/A":
                vix_level = vix_data["current"]
                if vix_level < 15:
                    st.success("😊 Low Fear (Complacent)")
                elif vix_level < 25:
                    st.info("😐 Moderate Fear")
                elif vix_level < 30:
                    st.warning("😰 High Fear")
                else:
                    st.error("😱 Extreme Fear")
                st.metric("VIX Level", f"{vix_level:.2f}")
            else:
                st.info("VIX data unavailable")
                
        with sent_col2:
            # DXY strength indicator
            dxy_data = prices.get("dxy_quote", {})
            if isinstance(dxy_data, dict) and dxy_data.get("current") != "N/A":
                dxy_level = dxy_data["current"]
                dxy_change = dxy_data.get("percentChange", 0)
                if dxy_level > 106:
                    st.error("💪 USD Very Strong")
                elif dxy_level > 104:
                    st.warning("📈 USD Strong")
                elif dxy_level > 102:
                    st.info("➖ USD Neutral")
                else:
                    st.success("📉 USD Weak")
                st.metric("DXY Index", f"{dxy_level:.2f}", f"{dxy_change:.2f}%")
            else:
                st.info("DXY data unavailable")
                
        with sent_col3:
            # Gold as safe haven indicator
            gold_data = prices.get("gold_quote", {})
            if isinstance(gold_data, dict) and gold_data.get("current") != "N/A":
                gold_price = gold_data["current"]
                gold_change = gold_data.get("percentChange", 0)
                if gold_change > 1:
                    st.warning("🏃 Risk-Off Flow")
                elif gold_change < -1:
                    st.success("📊 Risk-On Flow")
                else:
                    st.info("➖ Neutral Flow")
                st.metric("Gold", f"${gold_price:.2f}", f"{gold_change:.2f}%")
            else:
                st.info("Gold data unavailable")
        
        # Enhanced news section with EUR/GBP focus
        st.header("📰 Market-Moving News")
        
        news_tabs = st.tabs(["All News", "USD", "EUR", "GBP", "JPY", "Commodities", "Bonds"])
        
        all_news = st.session_state.data.get('news', [])
        
        with news_tabs[0]:  # All News
            if all_news:
                for article in all_news[:10]:
                    with st.expander(f"📰 {article['headline'][:100]}..."):
                        st.write(f"**Source:** {article['source']}")
                        st.write(f"**Time:** {article['date']}")
                        st.write(f"**Summary:** {article['summary']}")
                        if article.get('link'):
                            st.write(f"[Read more]({article['link']})")
            else:
                st.info("No news available. Click 'Refresh Data' to fetch latest news.")
        
        # Currency-specific news tabs
        currency_keywords = {
            "USD": ["dollar", "usd", "fed", "fomc", "powell", "treasury", "us economy", "nfp", "cpi", "pce"],
            "EUR": ["euro", "eur", "ecb", "lagarde", "eurozone", "european", "germany", "france"],
            "GBP": ["pound", "gbp", "boe", "bailey", "uk economy", "britain", "sterling", "gilt", "brexit"],
            "JPY": ["yen", "jpy", "boj", "ueda", "japan", "nikkei", "japanese"],
            "Commodities": ["oil", "gold", "silver", "copper", "commodity", "wti", "brent", "metals"],
            "Bonds": ["yield", "bond", "treasury", "gilt", "bund", "10y", "2y", "curve"]
        }
        
        for i, (currency, keywords) in enumerate(currency_keywords.items(), 1):
            with news_tabs[i]:
                filtered_news = [
                    article for article in all_news
                    if any(keyword.lower() in article['headline'].lower() or 
                          keyword.lower() in article['summary'].lower() 
                          for keyword in keywords)
                ]
                
                if filtered_news:
                    for article in filtered_news[:5]:
                        with st.expander(f"📰 {article['headline'][:80]}..."):
                            st.write(f"**Source:** {article['source']}")
                            st.write(f"**Time:** {article['date']}")
                            st.write(f"**Summary:** {article['summary']}")
                            if article.get('link'):
                                st.write(f"[Read more]({article['link']})")
                else:
                    st.info(f"No {currency}-specific news found.")
        
        # Technical Analysis Section
        st.header("📊 Technical Levels & Analysis")
        
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        
        with tech_col1:
            st.subheader("🎯 Key Support/Resistance")
            # This would typically come from technical analysis
            st.write("**EUR/USD**")
            st.write("• Resistance: 1.0550, 1.0600")
            st.write("• Support: 1.0480, 1.0420")
            st.write("")
            st.write("**GBP/USD**")
            st.write("• Resistance: 1.2650, 1.2700")
            st.write("• Support: 1.2550, 1.2500")
            st.write("")
            st.write("**EUR/GBP**")
            st.write("• Resistance: 0.8350, 0.8400")
            st.write("• Support: 0.8280, 0.8250")
            
        with tech_col2:
            st.subheader("📈 Momentum Indicators")
            st.write("**DXY**: RSI 58 (Neutral)")
            st.write("**Gold**: RSI 45 (Oversold)")
            st.write("**Oil**: RSI 62 (Overbought)")
            st.write("**S&P 500**: RSI 55 (Neutral)")
            st.write("**UK Gilts**: RSI 48 (Neutral)")
            
        with tech_col3:
            st.subheader("🔄 Market Correlations")
            st.write("**Strong Positive**")
            st.write("• EUR/USD ↔ Gold")
            st.write("• Risk Assets ↔ Oil")
            st.write("")
            st.write("**Strong Negative**")
            st.write("• DXY ↔ EUR/USD")
            st.write("• VIX ↔ S&P 500")
            st.write("• GBP ↔ UK Gilt Yields")
        
        # Trading Ideas Section
        st.header("💡 Trading Ideas & Opportunities")
        
        ideas_data = st.session_state.data.get('ai_trading_ideas', [])
        if ideas_data:
            idea_cols = st.columns(2)
            for idx, idea in enumerate(ideas_data[:4]):
                with idea_cols[idx % 2]:
                    with st.container():
                        st.markdown(f"### {idea.get('title', 'Trading Idea')}")
                        st.write(f"**Asset:** {idea.get('asset', 'N/A')}")
                        st.write(f"**Direction:** {idea.get('direction', 'N/A')}")
                        st.write(f"**Rationale:** {idea.get('rationale', 'N/A')}")
                        st.write(f"**Risk:** {idea.get('risk', 'N/A')}")
                        st.divider()
        else:
            st.info("No trading ideas available. Refresh data for AI-generated ideas.")
        
        # Risk Calendar
        st.header("📅 Upcoming Risk Events")
        
        risk_events = [
            {"date": "Today 13:30", "event": "US CPI Release", "impact": "High", "forecast": "3.3%"},
            {"date": "Today 15:00", "event": "BoE Bailey Speech", "impact": "Medium", "forecast": "Hawkish"},
            {"date": "Tomorrow 08:00", "event": "UK GDP", "impact": "High", "forecast": "0.2%"},
            {"date": "Tomorrow 14:00", "event": "ECB Minutes", "impact": "Medium", "forecast": "Neutral"},
            {"date": "Friday 13:30", "event": "US NFP", "impact": "High", "forecast": "180K"},
        ]
        
        risk_df = pd.DataFrame(risk_events)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        # Portfolio Risk Metrics
        st.header("⚡ Portfolio Risk Metrics")
        
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        
        with risk_col1:
            st.metric("Portfolio VaR (1d)", "-2.35%", "-0.15%")
        with risk_col2:
            st.metric("Sharpe Ratio", "1.85", "+0.12")
        with risk_col3:
            st.metric("Max Drawdown", "-8.5%", "-1.2%")
        with risk_col4:
            st.metric("Correlation Risk", "Medium", "Stable")
        
        # Footer with last update time
        st.divider()
        
        if 'last_update' in st.session_state:
            st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("Data not yet loaded. Click 'Refresh Data' to start.")
            
        # Add custom CSS for better styling
        st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stExpander {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
        }
        div[data-testid="column"] {
            padding: 0 10px;
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()