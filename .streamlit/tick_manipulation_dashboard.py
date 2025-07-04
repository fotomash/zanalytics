# tick_manipulation_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yaml
import os # MODIFIED: Import the 'os' module

# --- Main Dashboard Class ---
class TickManipulationDashboard:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['dashboard_config']
        
        self.session_state = {
            'vectors': {},
            'alerts': [],
            'manipulation_events': [],
            'tick_embeddings': np.zeros((0, 1536))
        }
        
    def analyze_tick_data(self, df: pd.DataFrame) -> dict:
        """Core analysis pipeline for tick data."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # The 'spread' column in your sample is large, assuming it's in basis points or similar
        # If it's already in pips, you can remove this division.
        df['spread'] = df['spread'] / 1000 
        
        df['tick_interval_ms'] = df['timestamp'].diff().dt.total_seconds() * 1000
        df['tick_interval_ms'] = df['tick_interval_ms'].fillna(0).clip(lower=0)
        
        spread_vectors = self._compute_spread_vectors(df)
        tick_bursts = self._detect_tick_bursts(df)
        manipulation_scores = self._detect_manipulation_patterns(df)
        volume_profile = self._analyze_volume_profile(df)
        
        self.session_state['vectors']['spread'] = spread_vectors
        self.session_state['vectors']['tick_bursts'] = tick_bursts
        self.session_state['manipulation_events'] = manipulation_scores['events']
        
        return {
            'raw_df': df,
            'spread_analysis': spread_vectors,
            'tick_bursts': tick_bursts,
            'manipulation': manipulation_scores,
            'volume_profile': volume_profile
        }
    
    def _compute_spread_vectors(self, df: pd.DataFrame) -> np.ndarray:
        window = self.config['analysis_modules']['spread_dynamics']['params']['window_size']
        threshold = self.config['analysis_modules']['spread_dynamics']['params']['anomaly_threshold']
        
        df['spread_mean'] = df['spread'].rolling(window=window, min_periods=1).mean()
        df['spread_std'] = df['spread'].rolling(window=window, min_periods=1).std().fillna(0)
        df['spread_zscore'] = (df['spread'] - df['spread_mean']) / (df['spread_std'] + 1e-9)
        df['spread_anomaly'] = np.abs(df['spread_zscore']) > threshold
        
        return df[['spread', 'spread_zscore', 'spread_anomaly']].values

    def _detect_tick_bursts(self, df: pd.DataFrame) -> dict:
        window = self.config['analysis_modules']['tick_rate_analyzer']['params']['burst_detection_window']
        threshold = self.config['analysis_modules']['tick_rate_analyzer']['params']['rate_change_threshold']
        
        df['tick_rate'] = 1000 / (df['tick_interval_ms'] + 1)
        df['tick_rate_ma'] = df['tick_rate'].rolling(window=window, min_periods=1).mean()
        df['tick_rate_std'] = df['tick_rate'].rolling(window=window, min_periods=1).std().fillna(0)
        df['tick_burst'] = df['tick_rate'] > (df['tick_rate_ma'] + threshold * df['tick_rate_std'])
        
        return df[['tick_rate', 'tick_burst']].to_dict('list')

    def _detect_manipulation_patterns(self, df: pd.DataFrame) -> dict:
        manipulation_events = []
        
        # Pattern: Rapid spread widening (potential spoofing/layering indicator)
        df['spread_change_rate'] = df['spread'].diff().abs().fillna(0)
        rapid_spread_changes = df[df['spread_change_rate'] > df['spread_change_rate'].quantile(0.98)]
        
        for _, row in rapid_spread_changes.iterrows():
            manipulation_events.append({
                'type': 'rapid_spread_change',
                'timestamp': row['timestamp'],
                'value': row['spread_change_rate'],
                'confidence': min(row['spread_change_rate'] / df['spread_change_rate'].quantile(0.99), 1.0)
            })
            
        return {'events': manipulation_events}

    def _analyze_volume_profile(self, df: pd.DataFrame) -> dict:
        df['price_level'] = ((df['bid'] + df['ask']) / 2).round(0)
        
        volume_profile = df.groupby('price_level').agg(
            tick_count=('timestamp', 'count')
        ).reset_index()
        
        return volume_profile.to_dict('list')

    def create_dashboard(self, analysis_results: dict):
        st.set_page_config(page_title="Tick Manipulation Analyzer", layout="wide")
        
        st.title("ZANFLOW: Tick-Level Market Structure & Manipulation Dashboard")
        st.caption(f"Asset: {self.config['asset']} | Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        df = analysis_results['raw_df']
        timestamps = df['timestamp']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ticks Analyzed", f"{len(df):,}")
        col2.metric("Manipulation Alerts", len(analysis_results['manipulation']['events']))
        col3.metric("Detected Tick Bursts", int(np.sum(analysis_results['tick_bursts']['tick_burst'])))

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Spread Dynamics & Anomalies', 'Tick Rate & Burst Events', 'Manipulation Timeline', 'Volume Profile'),
            specs=[[{}, {}], [{}, {"type": "bar"}]],
            vertical_spacing=0.15
        )

        # 1. Spread dynamics
        spreads = analysis_results['spread_analysis'][:, 0]
        anomalies = analysis_results['spread_analysis'][:, 2].astype(bool)
        fig.add_trace(go.Scatter(x=timestamps, y=spreads, mode='lines', name='Spread', line=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=timestamps[anomalies], y=spreads[anomalies], mode='markers', name='Spread Anomaly', marker=dict(color='#d62728', size=8, symbol='x')), row=1, col=1)

        # 2. Tick rate analysis
        tick_rates = analysis_results['tick_bursts']['tick_rate']
        bursts = np.array(analysis_results['tick_bursts']['tick_burst']).astype(bool)
        fig.add_trace(go.Scatter(x=timestamps, y=tick_rates, mode='lines', name='Tick Rate (ticks/sec)', line=dict(color='#2ca02c')), row=1, col=2)
        fig.add_trace(go.Scatter(x=timestamps[bursts], y=np.array(tick_rates)[bursts], mode='markers', name='Tick Burst', marker=dict(color='#ff7f0e', size=8)), row=1, col=2)

        # 3. Manipulation timeline
        events = analysis_results['manipulation']['events']
        if events:
            event_times = [e['timestamp'] for e in events]
            event_conf = [e['confidence'] for e in events]
            fig.add_trace(go.Scatter(x=event_times, y=event_conf, mode='markers', name='Manipulation Signal', marker=dict(color='#9467bd', size=12, line=dict(width=2, color='DarkSlateGrey'))), row=2, col=1)
        fig.update_yaxes(title_text="Confidence", range=[0, 1.1], row=2, col=1)

        # 4. Volume profile
        profile = analysis_results['volume_profile']
        fig.add_trace(go.Bar(y=profile['price_level'], x=profile['tick_count'], orientation='h', name='Tick Count by Price', marker_color='#8c564b'), row=2, col=2)
        fig.update_yaxes(title_text="Price Level", row=2, col=2)

        fig.update_layout(height=800, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# --- Main Execution Logic ---
if __name__ == "__main__":
    st.sidebar.title("Configuration")
    
    # MODIFIED: Load data path from secrets.toml
    try:
        data_dir = st.secrets["data_directory"]
        st.sidebar.success(f"Data directory loaded from secrets:\n{data_dir}")
    except (KeyError, FileNotFoundError):
        st.sidebar.error("`data_directory` not found in .streamlit/secrets.toml. Please check your configuration.")
        st.stop()

    # For this example, we'll hardcode the filename. You could make this a dropdown.
    file_name = "BTCUSD_ticks.csv"
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(file_path):
        st.error(f"Data file not found at: {file_path}")
        st.stop()

    # Initialize dashboard
    dashboard = TickManipulationDashboard('microstructure_dashboard_config.yaml')
    
    # Load and analyze data
    try:
        df = pd.read_csv(file_path, delimiter='\t')
        results = dashboard.analyze_tick_data(df)
        
        # Create dashboard
        dashboard.create_dashboard(results)
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")