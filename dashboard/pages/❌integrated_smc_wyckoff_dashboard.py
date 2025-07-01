# integrated_smc_wyckoff_dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
from utils.analysis_engines import SMCAnalyzer, WyckoffAnalyzer

# Page config
st.set_page_config(page_title="🎯 SMC Wyckoff Dashboard", 
                   page_icon="🎯", 
                   layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .bullish { color: #00c851; }
    .bearish { color: #ff4444; }
    .neutral { color: #ffbb33; }
    .poi-active { 
        background-color: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
    }
    .fvg-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class IntegratedDashboard:
    def __init__(self):
        self.smc_analyzer = SMCAnalyzer()
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        if 'selected_pair' not in st.session_state:
            st.session_state.selected_pair = 'ETHUSD'
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = 'M1'
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
            
    def load_data(self, pair, timeframe):
        """Load data for selected pair and timeframe"""
        try:
            # Try multiple file patterns
            file_patterns = [
                f"{pair}_{timeframe}_bars.csv",
                f"{pair.lower()}_{timeframe}_bars.csv",
                f"{pair}_bars.csv"
            ]
            
            data = None
            for pattern in file_patterns:
                if os.path.exists(pattern):
                    data = pd.read_csv(pattern, delimiter='\t')
                    break
                    
            if data is None:
                # Use sample data if file not found
                dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
                price_base = 2500
                price_changes = np.cumsum(np.random.randn(500) * 2)
                
                data = pd.DataFrame({
                    'timestamp': dates,
                    'open': price_base + price_changes + np.random.randn(500) * 0.5,
                    'high': price_base + price_changes + np.random.randn(500) * 0.5 + 2,
                    'low': price_base + price_changes + np.random.randn(500) * 0.5 - 2,
                    'close': price_base + price_changes + np.random.randn(500) * 0.5,
                    'volume': np.random.randint(100, 1000, 500),
                    'tickvol': np.random.randint(50, 500, 500)
                })
                
            # Ensure timestamp is datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            return data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
            
    def analyze_data(self, data):
        """Run all analyses on the data"""
        results = {}
        
        # SMC Analysis
        smc_results = self.smc_analyzer.analyze(data)
        results['order_blocks'] = smc_results.get('order_blocks', pd.DataFrame())
        results['fair_value_gaps'] = smc_results.get('fair_value_gaps', pd.DataFrame())
        results['liquidity_zones'] = smc_results.get('liquidity_zones', pd.DataFrame())
        results['market_structure'] = smc_results.get('market_structure', pd.DataFrame())
        
        # Wyckoff Analysis
        wyckoff_results = self.wyckoff_analyzer.analyze(data)
        results['wyckoff_phase'] = wyckoff_results.get('current_phase', 'Unknown')
        results['wyckoff_events'] = wyckoff_results.get('events', pd.DataFrame())
        results['volume_patterns'] = wyckoff_results.get('volume_patterns', pd.DataFrame())
        
        # Market Bias
        recent_structure = results['market_structure'].tail(20)
        if not recent_structure.empty:
            bullish_count = len(recent_structure[recent_structure['type'] == 'HH']) + \
                          len(recent_structure[recent_structure['type'] == 'HL'])
            bearish_count = len(recent_structure[recent_structure['type'] == 'LH']) + \
                          len(recent_structure[recent_structure['type'] == 'LL'])
            
            if bullish_count > bearish_count * 1.5:
                results['market_bias'] = 'BULLISH'
            elif bearish_count > bullish_count * 1.5:
                results['market_bias'] = 'BEARISH'
            else:
                results['market_bias'] = 'NEUTRAL'
        else:
            results['market_bias'] = 'NEUTRAL'
            
        # POIs Count
        active_pois = 0
        if not results['order_blocks'].empty:
            active_pois += len(results['order_blocks'][results['order_blocks']['status'] == 'active'])
        if not results['fair_value_gaps'].empty:
            active_pois += len(results['fair_value_gaps'][results['fair_value_gaps']['status'] == 'unfilled'])
        results['active_pois'] = active_pois
        
        # Risk Score
        if results['market_bias'] == 'NEUTRAL':
            results['risk_score'] = 'MEDIUM'
        elif results['wyckoff_phase'] in ['Distribution', 'Markdown']:
            results['risk_score'] = 'HIGH'
        else:
            results['risk_score'] = 'LOW'
            
        return results
        
    def render_header_metrics(self):
        """Render the header with key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        bias = st.session_state.analysis_results.get('market_bias', 'NEUTRAL')
        bias_class = 'bullish' if bias == 'BULLISH' else 'bearish' if bias == 'BEARISH' else 'neutral'
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MARKET BIAS</div>
                <div class="metric-value {bias_class}">{bias}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            pois = st.session_state.analysis_results.get('active_pois', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ACTIVE POIS</div>
                <div class="metric-value">{pois}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            phase = st.session_state.analysis_results.get('wyckoff_phase', 'Unknown')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">WYCKOFF PHASE</div>
                <div class="metric-value">{phase.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            risk = st.session_state.analysis_results.get('risk_score', 'MEDIUM')
            risk_class = 'bullish' if risk == 'LOW' else 'bearish' if risk == 'HIGH' else 'neutral'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RISK SCORE</div>
                <div class="metric-value {risk_class}">{risk}</div>
            </div>
            """, unsafe_allow_html=True)
            
    def render_price_action_tab(self):
        """Render advanced price action analysis"""
        st.subheader("📊 Advanced Price Action Analysis")
        
        data = st.session_state.data
        results = st.session_state.analysis_results
        
        # Create main chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=('Price Action with SMC & Wyckoff', 'Volume Analysis'))
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ), row=1, col=1)
        
        # Add Order Blocks
        order_blocks = results.get('order_blocks', pd.DataFrame())
        if not order_blocks.empty:
            for _, ob in order_blocks.iterrows():
                if ob['status'] == 'active':
                    color = 'rgba(0,255,0,0.2)' if ob['type'] == 'bullish' else 'rgba(255,0,0,0.2)'
                    fig.add_shape(
                        type="rect",
                        x0=ob['start_time'], x1=data['timestamp'].iloc[-1],
                        y0=ob['low'], y1=ob['high'],
                        fillcolor=color,
                        line=dict(width=0),
                        row=1, col=1
                    )
                    
        # Add Fair Value Gaps
        fvgs = results.get('fair_value_gaps', pd.DataFrame())
        if not fvgs.empty:
            for _, fvg in fvgs.iterrows():
                if fvg['status'] == 'unfilled':
                    color = 'rgba(255,215,0,0.3)'
                    fig.add_shape(
                        type="rect",
                        x0=fvg['timestamp'], x1=data['timestamp'].iloc[-1],
                        y0=fvg['low'], y1=fvg['high'],
                        fillcolor=color,
                        line=dict(color='gold', width=1),
                        row=1, col=1
                    )
                    
        # Add Market Structure
        structure = results.get('market_structure', pd.DataFrame())
        if not structure.empty:
            for _, point in structure.iterrows():
                color = 'green' if point['type'] in ['HH', 'HL'] else 'red'
                fig.add_annotation(
                    x=point['timestamp'],
                    y=point['price'],
                    text=point['type'],
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    ax=0, ay=-20,
                    row=1, col=1
                )
                
        # Volume bars
        colors = ['green' if c >= o else 'red' 
                 for c, o in zip(data['close'], data['open'])]
        fig.add_trace(go.Bar(
            x=data['timestamp'],
            y=data['volume'],
            marker_color=colors,
            name='Volume'
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_smc_tab(self):
        """Render Smart Money Concepts analysis"""
        st.subheader("🏦 Smart Money Concepts Analysis")
        
        results = st.session_state.analysis_results
        
        # Order Blocks Section
        st.markdown("### 📦 Order Blocks")
        order_blocks = results.get('order_blocks', pd.DataFrame())
        
        if not order_blocks.empty:
            active_obs = order_blocks[order_blocks['status'] == 'active']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Active Order Blocks", len(active_obs))
            with col2:
                bullish_obs = len(active_obs[active_obs['type'] == 'bullish'])
                st.metric("Bullish OBs", bullish_obs)
            with col3:
                bearish_obs = len(active_obs[active_obs['type'] == 'bearish'])
                st.metric("Bearish OBs", bearish_obs)
                
            # Show active order blocks
            if not active_obs.empty:
                st.dataframe(
                    active_obs[['timestamp', 'type', 'high', 'low', 'strength']].tail(5),
                    use_container_width=True
                )
        else:
            st.info("No order blocks detected")
            
        # Fair Value Gaps Section
        st.markdown("### 🌊 Fair Value Gaps")
        fvgs = results.get('fair_value_gaps', pd.DataFrame())
        
        if not fvgs.empty:
            active_fvgs = fvgs[fvgs['status'] == 'unfilled']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Active FVGs", len(active_fvgs), 
                         f"{len(active_fvgs[active_fvgs['timestamp'] > datetime.now() - timedelta(hours=1)])} new")
            with col2:
                morning_fvgs = len(active_fvgs[pd.to_datetime(active_fvgs['timestamp']).dt.hour.between(8, 10)])
                st.metric("8AM FVGs", morning_fvgs)
            with col3:
                avg_size = active_fvgs['size'].mean() if not active_fvgs.empty else 0
                st.metric("Avg Size", f"{avg_size:.1f} pips", "+2.3")
            with col4:
                filled_today = len(fvgs[(fvgs['status'] == 'filled') & 
                                      (pd.to_datetime(fvgs['filled_time']) > datetime.now().replace(hour=0, minute=0))])
                st.metric("Filled Today", filled_today)
                
            # FVG Alert
            latest_fvg = active_fvgs.iloc[-1] if not active_fvgs.empty else None
            if latest_fvg is not None and pd.to_datetime(latest_fvg['timestamp']) > datetime.now() - timedelta(minutes=30):
                st.markdown(f"""
                <div class="fvg-alert">
                    <strong>🚨 New FVG Detected!</strong><br>
                    {latest_fvg['type'].capitalize()} FVG formed at ${latest_fvg['low']:.2f} - ${latest_fvg['high']:.2f}<br>
                    Created {pd.to_datetime(latest_fvg['timestamp']).strftime('%H:%M')}
                </div>
                """, unsafe_allow_html=True)
                
        # Liquidity Analysis
        st.markdown("### 💧 Liquidity Analysis")
        liquidity = results.get('liquidity_zones', pd.DataFrame())
        
        if not liquidity.empty:
            # Get latest price for reference
            latest_price = st.session_state.data['close'].iloc[-1]
            
            # Find nearest liquidity zones
            buy_side = liquidity[(liquidity['type'] == 'buy_side') & (liquidity['level'] > latest_price)]
            sell_side = liquidity[(liquidity['type'] == 'sell_side') & (liquidity['level'] < latest_price)]
            
            st.markdown("**🎯 Next Liquidity Targets:**")
            
            if not buy_side.empty:
                next_buy = buy_side.iloc[0]
                st.markdown(f"⬆️ **Buy Side:** ${next_buy['level']:.2f} ({next_buy['strength']} probability)")
                
            if not sell_side.empty:
                next_sell = sell_side.iloc[0]
                st.markdown(f"⬇️ **Sell Side:** ${next_sell['level']:.2f} ({next_sell['strength']} probability)")
                
    def render_wyckoff_tab(self):
        """Render Wyckoff analysis"""
        st.subheader("🎭 Wyckoff Analysis")
        
        results = st.session_state.analysis_results
        data = st.session_state.data
        
        # Current Phase Analysis
        col1, col2 = st.columns([1, 2])
        
        with col1:
            phase = results.get('wyckoff_phase', 'Unknown')
            phase_colors = {
                'Accumulation': 'green',
                'Markup': 'lightgreen',
                'Distribution': 'red',
                'Markdown': 'darkred',
                'Unknown': 'gray'
            }
            
            st.markdown(f"""
            <div style="background-color: {phase_colors.get(phase, 'gray')}; 
                        color: white; padding: 2rem; border-radius: 10px; text-align: center;">
                <h3>Current Phase</h3>
                <h1>{phase}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            events = results.get('wyckoff_events', pd.DataFrame())
            if not events.empty:
                st.markdown("### Recent Wyckoff Events")
                recent_events = events.tail(5)
                for _, event in recent_events.iterrows():
                    st.markdown(f"- **{event['event']}** at {pd.to_datetime(event['timestamp']).strftime('%H:%M')} "
                              f"(Price: ${event['price']:.2f})")
                              
        # Volume Pattern Analysis
        st.markdown("### 📊 Volume Pattern Analysis")
        volume_patterns = results.get('volume_patterns', pd.DataFrame())
        
        if not volume_patterns.empty:
            # Create volume analysis chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.03,
                              row_heights=[0.6, 0.4])
                              
            # Price with volume overlays
            fig.add_trace(go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ), row=1, col=1)
            
            # Mark high volume areas
            for _, pattern in volume_patterns.iterrows():
                if pattern['type'] == 'climactic':
                    fig.add_annotation(
                        x=pattern['timestamp'],
                        y=pattern['price'],
                        text="CV",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='purple',
                        bgcolor='purple',
                        bordercolor='purple',
                        font=dict(color='white'),
                        row=1, col=1
                    )
                    
            # Volume with patterns
            colors = ['purple' if v > data['volume'].mean() * 2 else 'gray' 
                     for v in data['volume']]
            fig.add_trace(go.Bar(
                x=data['timestamp'],
                y=data['volume'],
                marker_color=colors,
                name='Volume'
            ), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
    def render_poi_tab(self):
        """Render Points of Interest"""
        st.subheader("⚡ Points of Interest")
        
        results = st.session_state.analysis_results
        latest_price = st.session_state.data['close'].iloc[-1]
        
        # Collect all POIs
        pois = []
        
        # Add order blocks
        order_blocks = results.get('order_blocks', pd.DataFrame())
        if not order_blocks.empty:
            active_obs = order_blocks[order_blocks['status'] == 'active']
            for _, ob in active_obs.iterrows():
                distance = abs(ob['high'] - latest_price) / latest_price * 100
                pois.append({
                    'type': 'Order Block',
                    'subtype': ob['type'],
                    'level': (ob['high'] + ob['low']) / 2,
                    'range': f"${ob['low']:.2f} - ${ob['high']:.2f}",
                    'strength': ob['strength'],
                    'distance': distance
                })
                
        # Add FVGs
        fvgs = results.get('fair_value_gaps', pd.DataFrame())
        if not fvgs.empty:
            active_fvgs = fvgs[fvgs['status'] == 'unfilled']
            for _, fvg in active_fvgs.iterrows():
                distance = abs(fvg['high'] - latest_price) / latest_price * 100
                pois.append({
                    'type': 'Fair Value Gap',
                    'subtype': fvg['type'],
                    'level': (fvg['high'] + fvg['low']) / 2,
                    'range': f"${fvg['low']:.2f} - ${fvg['high']:.2f}",
                    'strength': 'High',
                    'distance': distance
                })
                
        # Add liquidity zones
        liquidity = results.get('liquidity_zones', pd.DataFrame())
        if not liquidity.empty:
            for _, liq in liquidity.iterrows():
                distance = abs(liq['level'] - latest_price) / latest_price * 100
                pois.append({
                    'type': 'Liquidity Zone',
                    'subtype': liq['type'],
                    'level': liq['level'],
                    'range': f"${liq['level']:.2f}",
                    'strength': liq['strength'],
                    'distance': distance
                })
                
        # Sort by distance
        pois_df = pd.DataFrame(pois)
        if not pois_df.empty:
            pois_df = pois_df.sort_values('distance')
            
            # Display nearest POIs
            st.markdown("### 🎯 Nearest Points of Interest")
            
            for _, poi in pois_df.head(10).iterrows():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    icon = {'Order Block': '📦', 'Fair Value Gap': '🌊', 'Liquidity Zone': '💧'}.get(poi['type'], '📍')
                    st.markdown(f"{icon} **{poi['type']}**")
                    
                with col2:
                    color = 'green' if poi['subtype'] in ['bullish', 'buy_side'] else 'red'
                    st.markdown(f"<span style='color: {color}'>{poi['subtype'].upper()}</span>", unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(poi['range'])
                    
                with col4:
                    st.markdown(f"{poi['distance']:.1f}%")
                    
                st.divider()
                
    def render_setups_tab(self):
        """Render trade setup opportunities"""
        st.subheader("🎯 Trade Setup Opportunities")
        
        results = st.session_state.analysis_results
        data = st.session_state.data
        latest_price = data['close'].iloc[-1]
        
        # Identify potential setups
        setups = []
        
        # Check for order block retests
        order_blocks = results.get('order_blocks', pd.DataFrame())
        if not order_blocks.empty:
            active_obs = order_blocks[order_blocks['status'] == 'active']
            for _, ob in active_obs.iterrows():
                if ob['type'] == 'bullish' and latest_price <= ob['high'] * 1.01:
                    setups.append({
                        'type': 'Bullish OB Retest',
                        'entry': ob['high'],
                        'stop': ob['low'] * 0.99,
                        'target': ob['high'] * 1.02,
                        'risk_reward': 2.0,
                        'confidence': 'High'
                    })
                    
        # Check for FVG fills
        fvgs = results.get('fair_value_gaps', pd.DataFrame())
        if not fvgs.empty:
            active_fvgs = fvgs[fvgs['status'] == 'unfilled']
            for _, fvg in active_fvgs.iterrows():
                if fvg['type'] == 'bullish' and latest_price <= fvg['high']:
                    setups.append({
                        'type': 'Bullish FVG Fill',
                        'entry': fvg['high'],
                        'stop': fvg['low'] * 0.99,
                        'target': fvg['high'] * 1.015,
                        'risk_reward': 1.5,
                        'confidence': 'Medium'
                    })
                    
        # Display setups
        if setups:
            for i, setup in enumerate(setups[:3]):  # Show top 3 setups
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### Setup #{i+1}: {setup['type']}")
                    
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Entry", f"${setup['entry']:.2f}")
                    with metric_cols[1]:
                        st.metric("Stop Loss", f"${setup['stop']:.2f}")
                    with metric_cols[2]:
                        st.metric("Target", f"${setup['target']:.2f}")
                    with metric_cols[3]:
                        st.metric("R:R", f"{setup['risk_reward']:.1f}")
                        
                with col2:
                    confidence_colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
                    st.markdown(f"""
                    <div style="background-color: {confidence_colors[setup['confidence']]}; 
                                color: white; padding: 1rem; border-radius: 5px; 
                                text-align: center; margin-top: 2rem;">
                        <strong>Confidence</strong><br>
                        {setup['confidence']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.divider()
        else:
            st.info("No high-probability setups identified at current levels")
            
    def run(self):
        # Header
        st.markdown('<h1 class="main-header">🎯 SMC Wyckoff Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced Market Structure & Wyckoff Analysis</p>', unsafe_allow_html=True)
        
        # Sidebar for pair and timeframe selection
        with st.sidebar:
            st.markdown("### 📊 Data Selection")
            
            # Pair selection
            pairs = {
                'Crypto': ['ETHUSD', 'BTCUSD', 'ETHUSDT', 'BTCUSDT'],
                'Forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
            }
            
            pair_category = st.selectbox("Category", list(pairs.keys()))
            selected_pair = st.selectbox("Pair", pairs[pair_category])
            
            # Timeframe selection
            timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
            selected_timeframe = st.selectbox("Timeframe", timeframes)
            
            # Update button
            if st.button("🔄 Update Analysis", type="primary", use_container_width=True):
                st.session_state.selected_pair = selected_pair
                st.session_state.selected_timeframe = selected_timeframe
                
                # Load and analyze data
                with st.spinner("Loading data..."):
                    data = self.load_data(selected_pair, selected_timeframe)
                    if data is not None:
                        st.session_state.data = data
                        
                        with st.spinner("Running analysis..."):
                            results = self.analyze_data(data)
                            st.session_state.analysis_results = results
                            
                        st.success("Analysis complete!")
                        
            st.divider()
            
            # Settings
            st.markdown("### ⚙️ Settings")
            st.checkbox("Auto-refresh", key="auto_refresh")
            st.slider("Lookback Period", 50, 500, 200, key="lookback")
            
        # Load initial data if needed
        if st.session_state.data is None:
            data = self.load_data(st.session_state.selected_pair, st.session_state.selected_timeframe)
            if data is not None:
                st.session_state.data = data
                results = self.analyze_data(data)
                st.session_state.analysis_results = results
                
        # Display header metrics
        if st.session_state.analysis_results:
            self.render_header_metrics()
            
        # Tabs
        tabs = st.tabs([
            "📊 Advanced Price Action",
            "🏦 Smart Money Concepts",
            "🎭 Wyckoff Analysis",
            "⚡ Points of Interest",
            "🎯 Trade Setups"
        ])
        
        with tabs[0]:
            self.render_price_action_tab()
            
        with tabs[1]:
            self.render_smc_tab()
            
        with tabs[2]:
            self.render_wyckoff_tab()
            
        with tabs[3]:
            self.render_poi_tab()
            
        with tabs[4]:
            self.render_setups_tab()

# Run the dashboard
if __name__ == "__main__":
    dashboard = IntegratedDashboard()
    dashboard.run()