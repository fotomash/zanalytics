# quantum_microstructure_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import yaml
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class QuantumMicrostructureAnalyzer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['quantum_analysis_config']
        
        self.session_state = {
            'inferred_volumes': [],
            'iceberg_events': [],
            'spoofing_events': [],
            'manipulation_score': 0,
            'market_regime': 'normal',
            'toxic_flow_periods': [],
            'hidden_liquidity_map': {}
        }
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist and are properly formatted"""
        # Convert spread from integer to decimal (divide by 1000 for most brokers)
        df['spread'] = df['spread'] / 1000
        
        # Calculate mid price
        df['price_mid'] = (df['bid'] + df['ask']) / 2
        
        # Calculate tick intervals
        df['tick_interval_ms'] = df['timestamp'].diff().dt.total_seconds() * 1000
        df['tick_interval_ms'] = df['tick_interval_ms'].fillna(0).clip(lower=0.1)  # Avoid division by zero
        
        # Calculate tick rate (ticks per second)
        df['tick_rate'] = 1000 / df['tick_interval_ms']
        df['tick_rate'] = df['tick_rate'].clip(upper=1000)  # Cap at 1000 ticks/sec
        
        return df
    
    def infer_volume_from_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer volume using tick patterns, spread dynamics, and price movements"""
        params = self.config['analysis_modules']['volume_inference_engine']['params']
        
        # Feature engineering for volume inference
        df['price_change'] = df['price_mid'].diff().abs()
        df['spread_change'] = df['spread'].diff().abs()
        
        # Tick density (more ticks = higher volume)
        df['tick_density'] = df['tick_rate']
        
        # Spread dynamics (tighter spreads during high volume)
        df['spread_normalized'] = df['spread'] / df['price_mid']
        df['spread_volatility'] = df['spread_normalized'].rolling(20).std()
        
        # Price impact (larger moves = larger volume)
        df['price_velocity'] = df['price_change'] / (df['tick_interval_ms'] + 1)
        
        # Composite volume score with safe normalization
        features = ['tick_density', 'spread_change', 'price_velocity']
        df['volume_score'] = 0
        
        for feature, weight in zip(features, [params['tick_density_weight'], 
                                             params['spread_change_weight'], 
                                             params['price_movement_weight']]):
            if df[feature].std() > 0:
                normalized = (df[feature] - df[feature].mean()) / df[feature].std()
                df['volume_score'] += weight * normalized
        
        # Convert to actual volume estimate (normalized to reasonable range)
        df['inferred_volume'] = np.exp(df['volume_score'].clip(-3, 3)) * 100
        df['inferred_volume'] = df['inferred_volume'].fillna(100).clip(lower=0)
        
        return df
    
    def detect_icebergs(self, df: pd.DataFrame) -> List[Dict]:
        """Detect iceberg orders through repeated executions at similar price levels"""
        params = self.config['analysis_modules']['iceberg_detector']['params']
        iceberg_events = []
        
        df['price_rounded'] = ((df['bid'] + df['ask']) / 2).round(0)
        time_window = timedelta(seconds=params['time_window_seconds'])
        
        # Group by price levels and analyze execution patterns
        for price_level in df['price_rounded'].unique():
            level_data = df[df['price_rounded'] == price_level].copy()
            
            if len(level_data) >= params['min_executions']:
                # Check time clustering
                time_spread = level_data['timestamp'].max() - level_data['timestamp'].min()
                
                if time_spread <= time_window:
                    # Calculate execution pattern metrics
                    execution_intervals = level_data['timestamp'].diff().dt.total_seconds().dropna()
                    
                    if len(execution_intervals) > 0 and execution_intervals.mean() > 0:
                        interval_consistency = 1 - (execution_intervals.std() / (execution_intervals.mean() + 1e-9))
                        
                        if interval_consistency > 0.5:  # Consistent intervals suggest algorithmic execution
                            iceberg_events.append({
                                'type': 'iceberg_order',
                                'price_level': price_level,
                                'start_time': level_data['timestamp'].min(),
                                'end_time': level_data['timestamp'].max(),
                                'execution_count': len(level_data),
                                'avg_spread': level_data['spread'].mean(),
                                'interval_consistency': interval_consistency,
                                'estimated_total_size': len(level_data) * level_data['inferred_volume'].mean(),
                                'confidence': min(interval_consistency * (len(level_data) / params['min_executions']), 1.0)
                            })
        
        return iceberg_events
    
    def detect_spoofing(self, df: pd.DataFrame) -> List[Dict]:
        """Detect spoofing through spread manipulation patterns"""
        params = self.config['analysis_modules']['spoofing_detector']['params']
        spoofing_events = []
        
        df['spread_ma'] = df['spread'].rolling(20).mean()
        df['spread_spike'] = df['spread'] / (df['spread_ma'] + 1e-9)
        
        # Find spread spikes
        spike_mask = df['spread_spike'] > params['spread_spike_threshold']
        spike_indices = df[spike_mask].index
        
        for idx in spike_indices:
            if idx + 10 < len(df):  # Need future data to confirm reversal
                future_window = df.loc[idx:idx+10]
                
                # Check for quick reversal
                min_future_spread = future_window['spread'].min()
                reversal_ratio = min_future_spread / df.loc[idx, 'spread']
                
                if reversal_ratio < params['price_recovery_threshold']:
                    reversal_idx = future_window[future_window['spread'] == min_future_spread].index[0]
                    reversal_time = (df.loc[reversal_idx, 'timestamp'] - 
                                   df.loc[idx, 'timestamp']).total_seconds() * 1000
                    
                    if reversal_time < params['reversal_time_ms']:
                        spoofing_events.append({
                            'type': 'spoofing',
                            'timestamp': df.loc[idx, 'timestamp'],
                            'spike_spread': df.loc[idx, 'spread'],
                            'normal_spread': df.loc[idx, 'spread_ma'],
                            'spike_ratio': df.loc[idx, 'spread_spike'],
                            'reversal_time_ms': reversal_time,
                            'reversal_ratio': reversal_ratio,
                            'confidence': min((df.loc[idx, 'spread_spike'] - params['spread_spike_threshold']) / 2, 1.0)
                        })
        
        return spoofing_events
    
    def detect_quote_stuffing(self, df: pd.DataFrame) -> List[Dict]:
        """Detect quote stuffing through excessive update rates with minimal price movement"""
        params = self.config['analysis_modules']['quote_stuffing_detector']['params']
        stuffing_events = []
        
        # Calculate rolling metrics
        window_size = 50
        df['update_rate'] = df['tick_rate']
        df['price_movement'] = df['price_mid'].pct_change().abs()
        
        for i in range(window_size, len(df)):
            window = df.iloc[i-window_size:i]
            time_span = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds()
            
            if time_span > 0:
                updates_per_second = len(window) / time_span
                avg_price_movement = window['price_movement'].mean()
                
                if (updates_per_second > params['update_rate_threshold'] and 
                    avg_price_movement < params['price_movement_threshold']):
                    
                    stuffing_events.append({
                        'type': 'quote_stuffing',
                        'start_time': window['timestamp'].iloc[0],
                        'end_time': window['timestamp'].iloc[-1],
                        'updates_per_second': updates_per_second,
                        'avg_price_movement': avg_price_movement,
                        'tick_count': len(window),
                        'confidence': min(updates_per_second / params['update_rate_threshold'] - 1, 1.0)
                    })
        
        return stuffing_events
    
    def detect_layering(self, df: pd.DataFrame) -> List[Dict]:
        """Detect layering through order book imbalance patterns"""
        params = self.config['analysis_modules']['layering_detector']['params']
        layering_events = []
        
        # Analyze bid-ask pressure
        df['bid_pressure'] = df['bid'].diff()
        df['ask_pressure'] = df['ask'].diff()
        df['pressure_imbalance'] = df['bid_pressure'] - df['ask_pressure']
        
        # Look for sustained one-sided pressure
        window = params['time_correlation_window']
        for i in range(window, len(df) - window):
            segment = df.iloc[i-window:i+window]
            
            # Check for consistent pressure in one direction
            if segment['pressure_imbalance'].std() > 0 and abs(segment['pressure_imbalance'].mean()) > segment['pressure_imbalance'].std() * 2:
                direction = 'buy' if segment['pressure_imbalance'].mean() > 0 else 'sell'
                
                layering_events.append({
                    'type': 'layering',
                    'timestamp': segment['timestamp'].iloc[len(segment)//2],
                    'direction': direction,
                    'pressure_imbalance': segment['pressure_imbalance'].mean(),
                    'consistency': 1 - (segment['pressure_imbalance'].std() / (abs(segment['pressure_imbalance'].mean()) + 1e-9)),
                    'affected_levels': len(segment['price_mid'].unique()),
                    'confidence': min(abs(segment['pressure_imbalance'].mean()) / segment['spread'].mean(), 1.0)
                })
        
        return layering_events
    
    def calculate_vpin(self, df: pd.DataFrame, bucket_size: int = 50) -> pd.DataFrame:
        """Calculate Volume-Synchronized Probability of Informed Trading (VPIN)"""
        df['price_direction'] = np.sign(df['price_mid'].diff())
        df['buy_volume'] = df['inferred_volume'] * (df['price_direction'] > 0)
        df['sell_volume'] = df['inferred_volume'] * (df['price_direction'] < 0)
        
        # Calculate VPIN in buckets
        vpin_values = []
        for i in range(bucket_size, len(df), bucket_size):
            bucket = df.iloc[i-bucket_size:i]
            total_volume = bucket['inferred_volume'].sum()
            if total_volume > 0:
                vpin = abs(bucket['buy_volume'].sum() - bucket['sell_volume'].sum()) / total_volume
            else:
                vpin = 0
            vpin_values.extend([vpin] * bucket_size)
        
        # Pad the end
        if len(vpin_values) < len(df):
            vpin_values.extend([vpin_values[-1] if vpin_values else 0] * (len(df) - len(vpin_values)))
        
        df['vpin'] = vpin_values[:len(df)]
        
        return df
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime based on microstructure patterns"""
        # Calculate regime indicators
        spread_volatility = df['spread'].tail(100).std() / (df['spread'].mean() + 1e-9)
        tick_rate_variance = df['tick_interval_ms'].tail(100).var()
        price_volatility = df['price_mid'].tail(100).pct_change().std()
        avg_vpin = df['vpin'].tail(100).mean()
        
        # Regime classification
        if spread_volatility > 2 and tick_rate_variance > 10000:
            regime = 'breakdown'
        elif avg_vpin > 0.7 or len(self.session_state['spoofing_events']) > 5:
            regime = 'manipulated'
        elif price_volatility > df['price_mid'].pct_change().std() * 2:
            regime = 'stressed'
        else:
            regime = 'normal'
        
        return regime
    
    def create_advanced_dashboard(self, df: pd.DataFrame, selected_file: str):
        """Create comprehensive dashboard with all advanced analytics"""
        st.set_page_config(page_title="Quantum Microstructure Analyzer", layout="wide")
        
        # Header
        st.title("🧬 Quantum Microstructure Analysis System")
        asset = selected_file.split('_')[0] if '_' in selected_file else 'Unknown'
        st.caption(f"Asset: {asset} | File: {selected_file} | Regime: {self.session_state['market_regime'].upper()}")
        
        # Key Metrics Row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Iceberg Orders", len(self.session_state['iceberg_events']))
        col2.metric("Spoofing Events", len(self.session_state['spoofing_events']))
        col3.metric("Manipulation Score", f"{self.session_state['manipulation_score']:.2f}")
        col4.metric("Avg VPIN", f"{df['vpin'].mean():.3f}")
        col5.metric("Toxic Flow %", f"{len(self.session_state['toxic_flow_periods'])/len(df)*100:.1f}%")
        col6.metric("Hidden Liquidity", len(self.session_state['hidden_liquidity_map']))
        
        # Create comprehensive visualizations
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Price & Inferred Volume', 'Spread Dynamics & Anomalies', 'Tick Rate Heatmap',
                'VPIN (Toxicity)', 'Manipulation Timeline', 'Iceberg Detection Map',
                'Order Flow Imbalance', 'Microstructure 3D Surface', 'Hidden Liquidity Zones',
                'Quote Stuffing Density', 'Spoofing Patterns', 'Regime Transitions'
            ),
            specs=[
                [{'secondary_y': True}, {}, {}],
                [{}, {}, {}],
                [{}, {'type': 'surface'}, {}],
                [{}, {}, {}]
            ],
            row_heights=[0.25, 0.25, 0.25, 0.25],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Price & Volume
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['price_mid'], name='Price', line=dict(color='#1f77b4')),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['inferred_volume'], name='Volume', marker=dict(color='#ff7f0e', opacity=0.3)),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Spread Dynamics
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['spread'], name='Spread', line=dict(color='#2ca02c')),
            row=1, col=2
        )
        # Add spoofing markers
        for event in self.session_state['spoofing_events']:
            fig.add_trace(
                go.Scatter(
                    x=[event['timestamp']], y=[event['spike_spread']],
                    mode='markers', marker=dict(color='red', size=10, symbol='x'),
                    name='Spoofing', showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Tick Rate Heatmap
        # Create 2D histogram for tick density
        tick_heatmap = go.Histogram2d(
            x=df['timestamp'],
            y=df['tick_rate'],
            colorscale='Hot',
            nbinsx=100,
            nbinsy=50
        )
        fig.add_trace(tick_heatmap, row=1, col=3)
        
        # 4. VPIN
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['vpin'], name='VPIN', line=dict(color='#d62728')),
            row=2, col=1
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)
        
        # 5. Manipulation Timeline
        all_events = (
            [{'time': e['timestamp'], 'type': e['type'], 'conf': e['confidence']} for e in self.session_state['spoofing_events']] +
            [{'time': e['start_time'], 'type': e['type'], 'conf': e['confidence']} for e in self.session_state['iceberg_events']]
        )
        
        if all_events:
            event_df = pd.DataFrame(all_events)
            for event_type in event_df['type'].unique():
                type_events = event_df[event_df['type'] == event_type]
                fig.add_trace(
                    go.Scatter(
                        x=type_events['time'], y=type_events['conf'],
                        mode='markers', name=event_type,
                        marker=dict(size=10)
                    ),
                    row=2, col=2
                )
        
        # 6. Iceberg Map
        if self.session_state['iceberg_events']:
            iceberg_df = pd.DataFrame(self.session_state['iceberg_events'])
            fig.add_trace(
                go.Scatter(
                    x=iceberg_df['start_time'],
                    y=iceberg_df['price_level'],
                    mode='markers',
                    marker=dict(
                        size=iceberg_df['estimated_total_size']/100,
                        color=iceberg_df['confidence'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"Size: {s:.0f}" for s in iceberg_df['estimated_total_size']],
                    name='Icebergs'
                ),
                row=2, col=3
            )
        
        # 7. Order Flow Imbalance
        if 'bid_pressure' in df.columns and 'ask_pressure' in df.columns:
            df['order_flow_imbalance'] = (df['bid_pressure'] - df['ask_pressure']).rolling(50).mean()
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['order_flow_imbalance'], name='Order Flow Imbalance',
                          fill='tozeroy', line=dict(color='#9467bd')),
                row=3, col=1
            )
        
        # 8. 3D Microstructure Surface
        # Create meshgrid for 3D visualization
        try:
            time_bins = pd.cut(df.index, bins=50, labels=False)
            price_bins = pd.cut(df['price_mid'], bins=50, labels=False)
            
            surface_data = pd.pivot_table(
                df, values='inferred_volume', 
                index=price_bins, columns=time_bins, 
                aggfunc='sum', fill_value=0
            )
            
            if not surface_data.empty:
                fig.add_trace(
                    go.Surface(
                        z=surface_data.values,
                        colorscale='Viridis',
                        name='Volume Surface'
                    ),
                    row=3, col=2
                )
        except:
            pass  # Skip 3D surface if not enough data
        
        # 9. Hidden Liquidity Zones
        support_levels = df.groupby(df['price_mid'].round(0))['inferred_volume'].sum().sort_values(ascending=False).head(10)
        if not support_levels.empty:
            fig.add_trace(
                go.Bar(
                    x=support_levels.values,
                    y=support_levels.index,
                    orientation='h',
                    name='Hidden Liquidity',
                    marker=dict(color='#8c564b')
                ),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(height=1600, showlegend=False)
        fig.update_xaxes(title_text="Time", row=4, col=1)
        fig.update_xaxes(title_text="Time", row=4, col=2)
        fig.update_xaxes(title_text="Time", row=4, col=3)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Analysis Sections
        with st.expander("🔍 Manipulation Event Details", expanded=True):
            tab1, tab2, tab3, tab4 = st.tabs(["Icebergs", "Spoofing", "Layering", "Quote Stuffing"])
            
            with tab1:
                if self.session_state['iceberg_events']:
                    iceberg_df = pd.DataFrame(self.session_state['iceberg_events'])
                    st.dataframe(iceberg_df[['price_level', 'start_time', 'execution_count', 
                                           'estimated_total_size', 'confidence']].round(2))
                else:
                    st.info("No iceberg orders detected")
            
            with tab2:
                if self.session_state['spoofing_events']:
                    spoof_df = pd.DataFrame(self.session_state['spoofing_events'])
                    st.dataframe(spoof_df[['timestamp', 'spike_spread', 'normal_spread', 
                                         'reversal_time_ms', 'confidence']].round(2))
                else:
                    st.info("No spoofing events detected")
        
        # Market Quality Metrics
        with st.expander("📊 Market Quality Metrics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Spread", f"{df['spread'].mean():.4f}")
                st.metric("Spread Volatility", f"{df['spread'].std():.4f}")
                st.metric("Price Efficiency", f"{1 - df['spread'].mean()/df['price_mid'].mean():.6f}")
            
            with col2:
                st.metric("Avg Tick Rate", f"{df['tick_rate'].mean():.1f}/s")
                st.metric("Max Tick Burst", f"{df['tick_rate'].max():.0f}/s")
                st.metric("Tick Clustering", f"{df['tick_interval_ms'].std()/df['tick_interval_ms'].mean():.2f}")
            
            with col3:
                st.metric("Inferred Volume", f"{df['inferred_volume'].sum():,.0f}")
                st.metric("Volume Concentration", f"{df['inferred_volume'].std()/df['inferred_volume'].mean():.2f}")
                st.metric("Price Impact", f"{df['price_velocity'].mean():.8f}")

# Main execution
if __name__ == "__main__":
    st.sidebar.title("🧬 Quantum Configuration")
    
    try:
        tick_files_directory = st.secrets["raw_data_directory"]
        st.sidebar.success(f"Data Source:\n{tick_files_directory}")
    except KeyError:
        st.sidebar.error("`raw_data_directory` not found in secrets.toml")
        st.stop()
    
    # FIXED: Only show files with "ticks" in the name
    valid_files = [f for f in os.listdir(tick_files_directory) 
                   if 'ticks' in f.lower() and f.endswith(('.csv', '.txt')) and not f.startswith('._')]
    
    if not valid_files:
        st.sidebar.error("No tick files found")
        st.stop()
    
    selected_file = st.sidebar.selectbox("Select Tick Data", sorted(valid_files))
    file_path = os.path.join(tick_files_directory, selected_file)
    
    # Advanced options
    st.sidebar.markdown("### Advanced Settings")
    sensitivity = st.sidebar.slider("Detection Sensitivity", 0.5, 2.0, 1.0, 0.1)
    
    # Initialize analyzer
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'quantum_microstructure_config.yaml')
    
    analyzer = QuantumMicrostructureAnalyzer(config_path)
    
    # Load and analyze data
    with st.spinner("🧬 Running quantum analysis..."):
        df = pd.read_csv(file_path, delimiter='\t', encoding_errors='ignore')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # FIXED: Preprocess data to ensure all columns exist
        df = analyzer.preprocess_data(df)
        
        # Core analysis pipeline
        df = analyzer.infer_volume_from_ticks(df)
        df = analyzer.calculate_vpin(df)
        
        # Detect all manipulation patterns
        analyzer.session_state['iceberg_events'] = analyzer.detect_icebergs(df)
        analyzer.session_state['spoofing_events'] = analyzer.detect_spoofing(df)
        quote_stuffing = analyzer.detect_quote_stuffing(df)
        layering = analyzer.detect_layering(df)
        
        # Calculate manipulation score
        total_events = (len(analyzer.session_state['iceberg_events']) + 
                       len(analyzer.session_state['spoofing_events']) +
                       len(quote_stuffing) + len(layering))
        analyzer.session_state['manipulation_score'] = min(total_events / 10, 10)
        
        # Detect toxic flow periods
        analyzer.session_state['toxic_flow_periods'] = df[df['vpin'] > 0.7].index.tolist()
        
        # Detect market regime
        analyzer.session_state['market_regime'] = analyzer.detect_market_regime(df)
        
        # Create dashboard
        analyzer.create_advanced_dashboard(df, selected_file)
    
    # Add session state to sidebar
    with st.sidebar.expander("Session State"):
        st.json({
            'total_ticks': len(df),
            'regime': analyzer.session_state['market_regime'],
            'manipulation_score': round(analyzer.session_state['manipulation_score'], 2),
            'events_detected': total_events
        })