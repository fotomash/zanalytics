# enhanced_smc_dashboard_realtime.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os
import glob
from collections import defaultdict
import requests
import time as time_module
from functools import lru_cache
import threading

# Custom CSS with live price styling
CUSTOM_CSS = """
<style>
    .main {
        padding: 0rem 1rem;
    }
    
    .dashboard-header {
        background: linear-gradient(135deg, #1e3d59 0%, #2e5266 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .dashboard-title {
        color: #ffc13b;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .dashboard-subtitle {
        color: #f5f0e1;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    .analysis-card {
        background: rgba(30, 61, 89, 0.3);
        border: 1px solid rgba(255, 193, 59, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .card-header {
        color: #ffc13b;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .card-icon {
        margin-right: 0.5rem;
        font-size: 1.5rem;
    }
    
    .metric-container {
        background: rgba(255, 193, 59, 0.1);
        border: 1px solid rgba(255, 193, 59, 0.3);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffc13b;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #f5f0e1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .pair-selector {
        background: linear-gradient(135deg, #1e3d59 0%, #2e5266 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .pair-category {
        background: rgba(255, 193, 59, 0.1);
        border: 1px solid rgba(255, 193, 59, 0.3);
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    .price-ticker {
        background: rgba(30, 61, 89, 0.8);
        border: 1px solid rgba(255, 193, 59, 0.5);
        border-radius: 8px;
        padding: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .price-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffc13b;
    }
    
    .price-change {
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .price-up { color: #00d084; }
    .price-down { color: #ff3860; }
    
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00d084;
        border-radius: 50%;
        margin-left: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    
    .data-quality-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-left: 5px;
    }
    
    .quality-good { background: #00d084; }
    .quality-warning { background: #ffdd57; }
    .quality-poor { background: #ff3860; }
    
    .level-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-bullish {
        background: rgba(0, 208, 132, 0.2);
        border: 1px solid #00d084;
        color: #00ff00;
    }
    
    .badge-bearish {
        background: rgba(255, 56, 96, 0.2);
        border: 1px solid #ff3860;
        color: #ff6b6b;
    }
    
    .badge-neutral {
        background: rgba(255, 221, 87, 0.2);
        border: 1px solid #ffdd57;
        color: #ffdd57;
    }
    
    .alert-poi {
        background: linear-gradient(135deg, rgba(255, 193, 59, 0.2) 0%, rgba(255, 193, 59, 0.1) 100%);
        border-left: 4px solid #ffc13b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
"""

@dataclass
class PointOfInterest:
    """Enhanced POI with visual properties"""
    price: float
    type: str
    timestamp: datetime
    strength: float
    touches: int
    description: str
    color: str
    icon: str
    mitigated: bool = False
    entry_zone: Tuple[float, float] = None

class RealTimePriceProvider:
    """Manages real-time price feeds"""
    
    def __init__(self):
        self.setup_api_clients()
        self.price_cache = {}
        
    def setup_api_clients(self):
        """Setup API clients"""
        try:
            import finnhub
            self.finnhub_client = finnhub.Client(api_key=st.secrets.get("finnhub_api_key", ""))
        except:
            self.finnhub_client = None
            
    @st.cache_data(ttl=5)  # Cache for 5 seconds
    def get_real_time_price(self, symbol: str) -> Dict:
        """Get real-time price for symbol"""
        try:
            # Try Finnhub first
            if self.finnhub_client:
                return self.get_finnhub_price(symbol)
        except:
            pass
            
        # Fallback to mock data for demo
        return self.get_mock_price(symbol)
    
    def get_finnhub_price(self, symbol: str) -> Dict:
        """Get price from Finnhub"""
        # Map symbol to Finnhub format
        symbol_mapping = {
            'XAUUSD': 'OANDA:XAU_USD',
            'EURUSD': 'OANDA:EUR_USD',
            'GBPUSD': 'OANDA:GBP_USD',
            'USDJPY': 'OANDA:USD_JPY',
            'BTCUSD': 'BINANCE:BTCUSDT',
            'ETHUSD': 'BINANCE:ETHUSDT'
        }
        
        finnhub_symbol = symbol_mapping.get(symbol, symbol)
        
        try:
            quote = self.finnhub_client.quote(finnhub_symbol)
            
            return {
                'symbol': symbol,
                'price': quote['c'],  # current price
                'bid': quote['c'] - 0.001,  # approximate bid
                'ask': quote['c'] + 0.001,  # approximate ask
                'change': quote['d'],  # change
                'change_percent': quote['dp'],  # change percent
                'timestamp': datetime.now(),
                'volume': quote.get('v', 0),
                'source': 'finnhub'
            }
        except Exception as e:
            st.error(f"Finnhub API error for {symbol}: {str(e)}")
            return self.get_mock_price(symbol)
    
    def get_mock_price(self, symbol: str) -> Dict:
        """Generate mock price data for demo"""
        base_prices = {
            'XAUUSD': 2650.00,
            'EURUSD': 1.0850,
            'GBPUSD': 1.2650,
            'USDJPY': 148.50,
            'BTCUSD': 43000.00,
            'ETHUSD': 2300.00
        }
        
        base_price = base_prices.get(symbol, 100.00)
        
        # Add some random movement
        change = np.random.normal(0, 0.001) * base_price
        current_price = base_price + change
        
        spread = base_price * 0.0001  # 1 pip spread
        
        return {
            'symbol': symbol,
            'price': current_price,
            'bid': current_price - spread/2,
            'ask': current_price + spread/2,
            'change': change,
            'change_percent': (change / base_price) * 100,
            'timestamp': datetime.now(),
            'volume': np.random.randint(1000, 10000),
            'source': 'mock'
        }

class PairDataManager:
    """Manages available trading pairs and their data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.available_pairs = {}
        self.data_quality = {}
        self.last_update = {}
        
    def scan_available_pairs(self) -> Dict[str, List[str]]:
        """Scan directories for available pair data"""
        pairs_by_category = defaultdict(list)
        
        for directory in self.config.get('data_management', {}).get('scan_directories', []):
            if not os.path.exists(directory):
                continue
                
            # Scan for tick files
            for pattern in self.config.get('data_management', {}).get('supported_formats', ['*_ticks.csv']):
                files = glob.glob(os.path.join(directory, pattern))
                
                for file in files:
                    # Extract pair name from filename
                    filename = os.path.basename(file)
                    pair = self.extract_pair_name(filename)
                    
                    if pair:
                        # Categorize the pair
                        category = self.categorize_pair(pair)
                        pairs_by_category[category].append(pair)
                        
                        # Store file info
                        self.available_pairs[pair] = {
                            'file': file,
                            'category': category,
                            'last_modified': datetime.fromtimestamp(os.path.getmtime(file)),
                            'size': os.path.getsize(file)
                        }
                        
                        # Check data quality
                        self.data_quality[pair] = self.check_data_quality(file)
        
        # Remove duplicates
        for category in pairs_by_category:
            pairs_by_category[category] = sorted(list(set(pairs_by_category[category])))
            
        return dict(pairs_by_category)
    
    def extract_pair_name(self, filename: str) -> Optional[str]:
        """Extract pair name from filename"""
        # Remove extension
        name = filename.replace('.csv', '').replace('.parquet', '')
        
        # Remove common suffixes
        for suffix in ['_ticks', '_tick_data', 'tick_']:
            name = name.replace(suffix, '')
            
        # Extract uppercase pair names (e.g., XAUUSD, BTCUSD)
        import re
        match = re.search(r'([A-Z]{6,7})', name.upper())
        if match:
            return match.group(1)
            
        return None
    
    def categorize_pair(self, pair: str) -> str:
        """Categorize pair based on configuration"""
        categories = self.config.get('data_management', {}).get('pair_categories', {})
        
        for category, pairs in categories.items():
            if pair in pairs:
                return category.replace('_', ' ').title()
                
        # Auto-categorization based on pair name
        if pair.endswith('USD') or pair.endswith('EUR') or pair.endswith('GBP'):
            if pair.startswith('XAU') or pair.startswith('XAG'):
                return 'Metals'
            elif pair[:3] in ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOG']:
                return 'Crypto'
            else:
                return 'Forex'
        
        return 'Other'
    
    def check_data_quality(self, file_path: str) -> str:
        """Quick check of data quality"""
        try:
            # Read first few rows
            df = pd.read_csv(file_path, nrows=100)
            
            # Check for required columns
            required_cols = ['timestamp', 'bid', 'ask']
            if not all(col in df.columns for col in required_cols):
                return 'poor'
                
            # Check for data freshness
            if 'timestamp' in df.columns:
                try:
                    last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
                    age = datetime.now() - last_timestamp
                    
                    if age.days > 7:
                        return 'warning'
                    elif age.days > 30:
                        return 'poor'
                except:
                    pass
                    
            # Check for data completeness
            null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if null_ratio > 0.1:
                return 'warning'
            elif null_ratio > 0.3:
                return 'poor'
                
            return 'good'
            
        except Exception as e:
            return 'poor'

class EnhancedSMCDashboardWithPairs:
    """Enhanced SMC Dashboard with pair selection and real-time prices"""
    
    def __init__(self):
        self.load_config()
        self.setup_state()
        self.pair_manager = PairDataManager(self.config)
        self.price_provider = RealTimePriceProvider()
        
    def load_config(self):
        """Load enhanced configuration with fallbacks"""
        config_path = Path(__file__).parent / "enhanced_smc_dashboard_config.yaml"
        
        # Default config
        self.config = {
            'visualization': {
                'theme': {'primary': '#1e3d59', 'accent': '#ffc13b'},
                'chart': {'height': 800, 'template': 'plotly_dark', 'background': 'rgba(17, 17, 17, 0.95)', 'grid_color': 'rgba(128, 128, 128, 0.1)'},
                'smc_colors': {
                    'bullish_ob': 'rgba(0, 255, 0, 0.3)',
                    'bullish_ob_border': '#00ff00',
                    'bearish_ob': 'rgba(255, 0, 0, 0.3)',
                    'bearish_ob_border': '#ff0000',
                    'fvg_bull': 'rgba(0, 255, 255, 0.2)',
                    'fvg_bear': 'rgba(255, 0, 255, 0.2)',
                    'liquidity': 'rgba(255, 255, 0, 0.3)',
                    'inducement': 'rgba(255, 165, 0, 0.4)',
                    'swept_level': 'rgba(128, 128, 128, 0.3)'
                },
                'markers': {'poi_size': 20, 'structure_size': 15, 'entry_size': 18}
            },
            'data_management': {
                'scan_directories': ['/Users/tom/Documents/_trade/_exports/_tick'],
                'supported_formats': ['*_ticks.csv'],
                'pair_categories': {
                    'forex_majors': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],
                    'metals': ['XAUUSD', 'XAGUSD'],
                    'crypto': ['BTCUSD', 'ETHUSD']
                }
            }
        }
        
        # Try to load config file
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    self.config.update(loaded_config)
        except Exception as e:
            st.warning(f"Could not load config file: {str(e)}. Using defaults.")
            
    def setup_state(self):
        """Initialize session state"""
        if 'selected_pair' not in st.session_state:
            st.session_state.selected_pair = 'XAUUSD'
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = '15min'
        if 'available_pairs' not in st.session_state:
            st.session_state.available_pairs = {}
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
            
    def run(self):
        """Main dashboard entry point"""
        st.set_page_config(
            page_title="Elite SMC Trading Dashboard",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        
        # Pair selector
        self.render_pair_selector()
        
        # Header with real-time price
        self.render_header()
        
        # Main content area
        if st.session_state.selected_pair:
            self.render_main_content()
        else:
            st.info("Please select a trading pair to begin analysis")
            
        # Auto-refresh
        if st.session_state.auto_refresh:
            time_module.sleep(5)
            st.rerun()
            
    def render_pair_selector(self):
        """Render the pair selection interface"""
        st.markdown("""
            <div class="pair-selector">
                <h3 style="color: #ffc13b; margin-bottom: 1rem;">
                    📊 Select Trading Pair
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Scan for available pairs
        with st.spinner("Scanning for available pairs..."):
            pairs_by_category = self.pair_manager.scan_available_pairs()
            st.session_state.available_pairs = pairs_by_category
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display pairs by category
            self.render_categorized_pairs(pairs_by_category)
                
        with col2:
            # Timeframe selector
            st.selectbox(
                "Timeframe",
                options=['tick', '1min', '5min', '15min', '30min', '1h', '4h', '1d'],
                index=3,  # Default to 15min
                key='selected_timeframe'
            )
            
            # Auto-refresh toggle
            st.checkbox(
                "🔄 Auto Refresh",
                value=True,
                key='auto_refresh',
                help="Auto-refresh prices every 5 seconds"
            )
            
    def render_categorized_pairs(self, pairs_by_category: Dict[str, List[str]]):
        """Render pairs organized by category"""
        for category, pairs in pairs_by_category.items():
            if not pairs:
                continue
                
            st.markdown(f"""
                <div class="pair-category">
                    <h4 style="color: #f5f0e1; margin-bottom: 0.5rem;">
                        {self.get_category_icon(category)} {category}
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Create a grid of pair buttons
            cols = st.columns(6)
            for idx, pair in enumerate(pairs):
                with cols[idx % 6]:
                    self.render_pair_button(pair)
                    
            st.markdown("<br>", unsafe_allow_html=True)
            
    def render_pair_button(self, pair: str):
        """Render individual pair button with real-time price"""
        # Get real-time price
        price_data = self.price_provider.get_real_time_price(pair)
        
        # Get data quality
        quality = self.pair_manager.data_quality.get(pair, 'unknown')
        quality_color = {
            'good': '#00d084',
            'warning': '#ffdd57',
            'poor': '#ff3860',
            'unknown': '#666666'
        }.get(quality, '#666666')
        
        # Create button
        selected = st.session_state.selected_pair == pair
        
        if st.button(
            f"{pair}",
            key=f"pair_{pair}",
            use_container_width=True,
            type="primary" if selected else "secondary"
        ):
            st.session_state.selected_pair = pair
            st.rerun()
            
        # Show price and quality indicator
        change_color = "price-up" if price_data['change'] >= 0 else "price-down"
        change_symbol = "+" if price_data['change'] >= 0 else ""
        
        st.markdown(f"""
            <div style="text-align: center; font-size: 0.8rem;">
                <div class="price-value" style="font-size: 1rem;">${price_data['price']:.2f}</div>
                <div class="{change_color}" style="font-size: 0.7rem;">
                    {change_symbol}{price_data['change_percent']:.2f}%
                    <span class="data-quality-indicator quality-{quality}" 
                          style="background: {quality_color};" 
                          title="Data quality: {quality}">
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    def get_category_icon(self, category: str) -> str:
        """Get icon for category"""
        icons = {
            'Forex Majors': '💱',
            'Forex Crosses': '🔄',
            'Metals': '🏆',
            'Crypto': '₿',
            'Indices': '📊',
            'Other': '📈'
        }
        return icons.get(category, '📊')
        
    def render_header(self):
        """Render header with selected pair info and real-time price"""
        if st.session_state.selected_pair:
            # Get real-time price
            price_data = self.price_provider.get_real_time_price(st.session_state.selected_pair)
            pair_info = self.pair_manager.available_pairs.get(st.session_state.selected_pair, {})
            
            change_color = "price-up" if price_data['change'] >= 0 else "price-down"
            change_symbol = "+" if price_data['change'] >= 0 else ""
            
            st.markdown(f"""
                <div class="dashboard-header">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h1 class="dashboard-title">
                                🎯 {st.session_state.selected_pair} Analysis
                            </h1>
                            <p class="dashboard-subtitle">
                                {pair_info.get('category', 'Unknown')} • 
                                {st.session_state.selected_timeframe} • 
                                Source: {price_data['source'].title()}
                            </p>
                        </div>
                        <div class="price-ticker">
                            <div>
                                <div class="price-value">${price_data['price']:.4f}</div>
                                <div class="price-change {change_color}">
                                    {change_symbol}{price_data['change_percent']:.2f}%
                                    <span class="live-indicator"></span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Quick stats for selected pair (removed win rate)
            self.render_pair_stats(price_data)
            
    def render_pair_stats(self, price_data: Dict):
        """Render statistics for selected pair"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            self.render_metric("Market Bias", "BULLISH", "badge-bullish")
        with col2:
            self.render_metric("Active POIs", "12", "badge-neutral")
        with col3:
            self.render_metric("Wyckoff Phase", "MARKUP", "badge-bullish")
        with col4:
            self.render_metric("Risk Score", "LOW", "badge-bullish")
        with col5:
            spread = price_data['ask'] - price_data['bid']
            self.render_metric("Spread", f"{spread:.4f}", "badge-neutral")
            
    def render_metric(self, label: str, value: str, badge_class: str):
        """Render a styled metric"""
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">{label}</div>
                <div class="metric-value {badge_class}">{value}</div>
            </div>
        """, unsafe_allow_html=True)
        
    def render_main_content(self):
        """Render main content area"""
        # Create tabs with icons
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Advanced Price Action",
            "🏦 Smart Money Concepts",
            "🎭 Wyckoff Analysis",
            "⚡ Points of Interest",
            "🎯 Trade Setups"
        ])
        
        with tab1:
            self.render_advanced_chart()
            
        with tab2:
            self.render_smc_analysis()
            
        with tab3:
            self.render_wyckoff_analysis()
            
        with tab4:
            self.render_poi_analysis()
            
        with tab5:
            self.render_trade_setups()
            
    def render_advanced_chart(self):
        """Render advanced price action chart with all markups"""
        st.markdown("""
            <div class="analysis-card">
                <div class="card-header">
                    <span class="card-icon">📈</span>
                    Advanced Price Action Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create sophisticated chart
        fig = self.create_advanced_chart()
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
    def create_advanced_chart(self) -> go.Figure:
        """Create advanced chart with all SMC markups"""
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.7, 0.15, 0.15],
            subplot_titles=('', '', ''),
            specs=[[{"secondary_y": True}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Generate sample data (replace with real data)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='15min')
        prices = 2650 + np.cumsum(np.random.randn(200) * 0.5)
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=prices - np.random.rand(200) * 0.3,
                high=prices + np.random.rand(200) * 0.5,
                low=prices - np.random.rand(200) * 0.5,
                close=prices + np.random.rand(200) * 0.3 - 0.15,
                name='Price',
                increasing_line_color='#00d084',
                decreasing_line_color='#ff3860'
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Add SMC markup elements
        self.add_order_blocks(fig, dates, prices)
        self.add_fair_value_gaps(fig, dates, prices)
        self.add_liquidity_zones(fig, dates, prices)
        self.add_points_of_interest(fig, dates, prices)
        self.add_market_structure(fig, dates, prices)
        
        # Volume with color coding
        volume = np.random.randint(1000, 5000, 200)
        colors = ['#00d084' if i % 2 == 0 else '#ff3860' for i in range(200)]
        
        fig.add_trace(
            go.Bar(
                x=dates,
                y=volume,
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # RSI indicator
        rsi = 50 + np.cumsum(np.random.randn(200) * 2)
        rsi = np.clip(rsi, 20, 80)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rsi,
                name='RSI',
                line=dict(color='#ffc13b', width=2)
            ),
            row=3, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            template=self.config['visualization']['chart']['template'],
            height=self.config['visualization']['chart']['height'],
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor=self.config['visualization']['chart']['background'],
            paper_bgcolor=self.config['visualization']['chart']['background'],
            font=dict(color='#f5f0e1')
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=self.config['visualization']['chart']['grid_color'],
            showgrid=True,
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor=self.config['visualization']['chart']['grid_color'],
            showgrid=True,
            zeroline=False
        )
        
        return fig
        
    def add_order_blocks(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add order blocks to chart"""
        # Bullish Order Block
        ob_start = 50
        ob_end = 70
        ob_high = prices[ob_start:ob_end].max() + 1
        ob_low = prices[ob_start:ob_end].min() - 0.5
        
        fig.add_shape(
            type="rect",
            x0=dates[ob_start], x1=dates[ob_end],
            y0=ob_low, y1=ob_high,
            fillcolor=self.config['visualization']['smc_colors']['bullish_ob'],
            line=dict(color=self.config['visualization']['smc_colors']['bullish_ob_border'], width=2),
            row=1, col=1
        )
        
        # Add annotation
        fig.add_annotation(
            x=dates[ob_start + 10],
            y=ob_high + 0.5,
            text="Bullish OB",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#00ff00",
            font=dict(color="#00ff00", size=12),
            row=1, col=1
        )
        
    def add_fair_value_gaps(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add Fair Value Gaps to chart"""
        # FVG example
        fvg_index = 100
        fvg_high = prices[fvg_index] + 0.8
        fvg_low = prices[fvg_index] - 0.8
        
        fig.add_shape(
            type="rect",
            x0=dates[fvg_index - 5], x1=dates[fvg_index + 5],
            y0=fvg_low, y1=fvg_high,
            fillcolor=self.config['visualization']['smc_colors']['fvg_bull'],
            line=dict(width=0),
            row=1, col=1
        )
        
        # FVG label
        fig.add_annotation(
            x=dates[fvg_index],
            y=fvg_high + 0.3,
            text="FVG",
            showarrow=False,
            font=dict(color="#00ffff", size=10),
            bgcolor="rgba(0,0,0,0.5)",
            row=1, col=1
        )
        
    def add_liquidity_zones(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add liquidity zones"""
        # Equal highs liquidity
        liq_level = prices.max() + 1
        
        fig.add_hline(
            y=liq_level,
            line_dash="dot",
            line_color="#ffff00",
            line_width=2,
            annotation_text="Liquidity Pool",
            annotation_position="right",
            row=1, col=1
        )
        
        # Add liquidity grab visualization
        grab_index = 150
        fig.add_trace(
            go.Scatter(
                x=[dates[grab_index]],
                y=[liq_level + 0.5],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=20,
                    color='#ffff00',
                    line=dict(width=2, color='#ffffff')
                ),
                name='Liquidity Grab',
                showlegend=True
            ),
            row=1, col=1
        )
        
    def add_points_of_interest(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add Points of Interest"""
        # Define POIs
        pois = [
            PointOfInterest(
                price=prices[30],
                type='Demand Zone',
                timestamp=dates[30],
                strength=0.9,
                touches=3,
                description='Strong Demand Zone',
                color='#00ff00',
                icon='⬆️'
            ),
            PointOfInterest(
                price=prices[120],
                type='Supply Zone',
                timestamp=dates[120],
                strength=0.85,
                touches=4,
                description='Key Supply Zone',
                color='#ff0000',
                icon='⬇️'
            )
        ]
        
        for poi in pois:
            # Add POI marker
            fig.add_trace(
                go.Scatter(
                    x=[poi.timestamp],
                    y=[poi.price],
                    mode='markers+text',
                    marker=dict(
                        symbol='hexagram',
                        size=self.config['visualization']['markers']['poi_size'],
                        color=poi.color,
                        line=dict(width=2, color='white')
                    ),
                    text=[poi.icon],
                    textposition='top center',
                    name=poi.type,
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add POI zone
            zone_width = 0.5
            fig.add_shape(
                type="rect",
                x0=dates[0], x1=dates[-1],
                y0=poi.price - zone_width, y1=poi.price + zone_width,
                fillcolor=poi.color,
                opacity=0.1,
                line=dict(width=0),
                row=1, col=1
            )
            
    def add_market_structure(self, fig: go.Figure, dates: pd.DatetimeIndex, prices: np.ndarray):
        """Add market structure elements"""
        # Add swing highs and lows
        swing_highs = [20, 60, 100, 140, 180]
        swing_lows = [10, 40, 80, 120, 160]
        
        # Plot swing highs
        for idx in swing_highs:
            if idx < len(dates):
                fig.add_trace(
                    go.Scatter(
                        x=[dates[idx]],
                        y=[prices[idx] + 1],
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='#ff3860'
                        ),
                        text=['HH' if idx > 60 else 'LH'],
                        textposition='top center',
                        textfont=dict(color='#ff3860', size=10),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
        # Plot swing lows
        for idx in swing_lows:
            if idx < len(dates):
                fig.add_trace(
                    go.Scatter(
                        x=[dates[idx]],
                        y=[prices[idx] - 1],
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='#00d084'
                        ),
                        text=['HL' if idx > 40 else 'LL'],
                        textposition='bottom center',
                        textfont=dict(color='#00d084', size=10),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
    def render_smc_analysis(self):
        """Render Smart Money Concepts analysis section"""
        st.markdown("""
            <div class="analysis-card">
                <div class="card-header">
                    <span class="card-icon">🏦</span>
                    Smart Money Concepts Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Sample SMC data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📦 Order Blocks")
            ob_data = {
                'Type': ['Bullish OB', 'Bearish OB', 'Bullish OB'],
                'Price': ['$2,650.50', '$2,680.20', '$2,635.80'],
                'Strength': ['Strong', 'Medium', 'Strong'],
                'Status': ['Active', 'Mitigated', 'Active']
            }
            st.dataframe(pd.DataFrame(ob_data), use_container_width=True, hide_index=True)
            
        with col2:
            st.markdown("### 🌊 Fair Value Gaps")
            fvg_data = {
                'Type': ['Bull FVG', 'Bear FVG'],
                'Range': ['$2,655-$2,657', '$2,670-$2,672'],
                'Status': ['Active', 'Filled']
            }
            st.dataframe(pd.DataFrame(fvg_data), use_container_width=True, hide_index=True)
    
    def render_wyckoff_analysis(self):
        """Render Wyckoff analysis"""
        st.markdown("### 🎭 Wyckoff Phase Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        phases = [
            ("Accumulation", "🟢", col1, "Phase B"),
            ("Markup", "🔵", col2, "Early"),
            ("Distribution", "🔴", col3, "Not Active"),
            ("Markdown", "🟠", col4, "Not Active")
        ]
        
        for phase, emoji, col, status in phases:
            with col:
                active = "Active" in status or "Phase" in status or "Early" in status
                bg_color = "rgba(0,208,132,0.2)" if active else "rgba(128,128,128,0.1)"
                
                st.markdown(f"""
                    <div style="background: {bg_color}; padding: 1rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 2rem;">{emoji}</div>
                        <div style="font-weight: bold; color: #ffc13b;">{phase}</div>
                        <div style="font-size: 0.9rem; color: #f5f0e1;">{status}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    def render_poi_analysis(self):
        """Render POI analysis"""
        st.markdown("### ⚡ Points of Interest")
        
        poi_data = {
            'Level': ['$2,680', '$2,672', '$2,658', '$2,645'],
            'Type': ['Resistance', 'Supply Zone', 'Demand Zone', 'Support'],
            'Touches': [5, 3, 4, 6],
            'Entry Zone': ['$2,678-2,682', '$2,670-2,674', '$2,656-2,660', '$2,643-2,647'],
            'Risk/Reward': ['1:3.5', '1:2.8', '1:4.2', '1:3.8']
        }
        
        st.dataframe(pd.DataFrame(poi_data), use_container_width=True, hide_index=True)
    
    def render_trade_setups(self):
        """Render trade setups"""
        st.markdown("### 🎯 Active Trade Setups")
        
        setups = [
            {
                "name": "Bullish OB Retest",
                "entry": "$2,656 - $2,658",
                "stop": "$2,652",
                "targets": ["$2,665", "$2,672", "$2,680"],
                "rr": "1:3.5",
                "confidence": "85%",
                "status": "Active"
            }
        ]
        
        for setup in setups:
            st.markdown(f"""
                <div class="analysis-card" style="border-left: 4px solid #00d084;">
                    <h4 style="color: #00d084; margin: 0;">{setup['name']}</h4>
                    <div style="margin-top: 1rem;">
                        <strong>Entry:</strong> {setup['entry']}<br>
                        <strong>Stop:</strong> {setup['stop']}<br>
                        <strong>Targets:</strong> {', '.join(setup['targets'])}<br>
                        <strong>R:R:</strong> {setup['rr']} | <strong>Confidence:</strong> {setup['confidence']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Run the dashboard
if __name__ == "__main__":
    dashboard = EnhancedSMCDashboardWithPairs()
    dashboard.run()