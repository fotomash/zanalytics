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

    def detect_mock_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
        """
        Heuristically decide whether the loaded tick data looks synthetic / mocked.

        Returns
        -------
        Tuple[bool, Dict[str, float]]
            * bool  -> True if the data is suspicious (likely mock)
            * dict  -> diagnostic metrics that triggered the suspicion
        """
        diagnostics: Dict[str, float] = {}

        # 1) Are price‑increments overly uniform?
        price_steps = df['price_mid'].diff().abs().round(8).dropna()
        diagnostics['unique_price_steps'] = price_steps.nunique()

        # 2) Is the spread almost always the same?
        diagnostics['unique_spread_values'] = df['spread'].nunique()

        # 3) Are tick intervals constant (i.e. generated on a timer)?
        diagnostics['tick_interval_std_ms'] = df['tick_interval_ms'].std()

        # 4) Make a simple judgement
        suspicious = (
            diagnostics['unique_price_steps'] < 3  # too few distinct price changes
            or diagnostics['unique_spread_values'] < 3  # spread doesn’t vary
            or diagnostics['tick_interval_std_ms'] < 1.0  # intervals virtually constant
        )

        return suspicious, diagnostics

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
        """Create comprehensive dashboard with all advanced analytics (each chart as its own figure)"""
        st.set_page_config(page_title="Quantum Microstructure Analyzer", layout="wide")

        # Header
        st.title("🧬 Quantum Microstructure Analysis System")
        asset = selected_file.split('_')[0] if '_' in selected_file else 'Unknown'
        st.caption(f"Asset: {asset} | File: {selected_file} | Regime: {self.session_state['market_regime'].upper()}")

        # ——— Research Commentary ———
        with st.expander("📚 Research Commentary (methodology & caveats)", expanded=False):
            st.markdown(
                """
**MetaTrader data caveat:**  
We analyse `<TICKVOL>` and `<SPREAD>` from MetaTrader exports. These lack true Level II order-book depth, so Quote-Stuffing and Spoofing signals are **heuristic** rather than definitive.

**ZANFLOW microstructure heuristics**

| Signal | What we look for in tick data | Heuristic trigger |
| --- | --- | --- |
| **Quote Stuffing Density** | Extreme burst of tick updates with <1 bp price change | `updates / sec` > threshold **and** `avg_price_movement` below threshold |
| **Spoofing Patterns** | Spread spikes that reverse within ≤ 250 ms plus no follow-through in price | `spread_spike` > 3×MA **then** quick reversion |
| **Regime Transitions** | VPIN > 0.7 ⇒ *manipulated* • VPIN 0.5-0.7 ⇒ *stressed* • else *normal* | VPIN 100-tick rolling |

**Further reading**

* [Early Market Signals and Confirmations](sandbox:/mnt/data/Early Market Signals and Confirmations.md)  
* [Market Trading Techniques and Anomalies](sandbox:/mnt/data/Market Trading Techniques and Anomalies.md)  
* [Trap Reversal Tick-Validated JSON](sandbox:/mnt/data/SMC_EngineeredLiquidity_TrapReversal_TickValidated_v12.json)  
* [Tick Data Analysis Notes](sandbox:/mnt/data/tick_data_analisys.json)

These documents expand on engineered-liquidity traps, Wyckoff sweeps, and the VPIN-driven regime model used here.
"""
            )

        # Key Metrics Row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Iceberg Orders", len(self.session_state['iceberg_events']))
        col2.metric("Spoofing Events", len(self.session_state['spoofing_events']))
        col3.metric("Manipulation Score", f"{self.session_state['manipulation_score']:.2f}")
        col4.metric("Avg VPIN", f"{df['vpin'].mean():.3f}")
        col5.metric("Toxic Flow %", f"{len(self.session_state['toxic_flow_periods'])/len(df)*100:.1f}%")
        col6.metric("Hidden Liquidity", len(self.session_state['hidden_liquidity_map']))

        # 1. Price & Inferred Volume
        try:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['price_mid'], name='Price', line=dict(color='#1f77b4')))
            fig1.add_trace(go.Bar(x=df['timestamp'], y=df['inferred_volume'], name='Volume', marker=dict(color='#ff7f0e', opacity=0.3)))
            fig1.update_layout(title="Price & Inferred Volume", xaxis_title="Time", yaxis_title="Price / Volume", showlegend=True)
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Price vs Inferred Volume: Shows the evolution of market price alongside estimated trading volume. The line represents price, and the bars represent inferred volume at each tick.")
            # Dynamic commentary
            price_change = df['price_mid'].iloc[-1] - df['price_mid'].iloc[0]
            total_vol = df['inferred_volume'].sum()
            st.markdown(f"**Quick take:** Price moved {price_change:+.4f} points over the displayed period with an estimated {total_vol:,.0f} units traded.")
        except Exception:
            pass

        # 2. Spread Dynamics & Anomalies
        try:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['spread'], name='Spread', line=dict(color='#2ca02c')))
            # Add spoofing markers
            for event in self.session_state['spoofing_events']:
                fig2.add_trace(
                    go.Scatter(
                        x=[event['timestamp']], y=[event['spike_spread']],
                        mode='markers', marker=dict(color='red', size=10, symbol='x'),
                        name='Spoofing', showlegend=False
                    )
                )
            fig2.update_layout(title="Spread Dynamics & Anomalies", xaxis_title="Time", yaxis_title="Spread", showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Spread Dynamics & Anomalies: Visualizes spread changes over time and highlights detected spoofing events with red X markers.")
            # Dynamic commentary
            avg_spread = df['spread'].mean()
            st.markdown(f"**Spread check:** Average spread {avg_spread:.4f} • Spoofing spikes detected: {len(self.session_state['spoofing_events'])}.")
        except Exception:
            pass

        # 3. Tick Rate Heatmap
        try:
            fig3 = go.Figure()
            heatmap = go.Histogram2d(
                x=df['timestamp'],
                y=df['tick_rate'],
                colorscale='Hot',
                nbinsx=100,
                nbinsy=50
            )
            fig3.add_trace(heatmap)
            fig3.update_layout(title="Tick Rate Heatmap", xaxis_title="Time", yaxis_title="Tick Rate")
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Tick Rate Heatmap: Displays the density of tick updates over time and tick rate. Brighter areas indicate higher density of updates.")
            # Dynamic commentary
            peak_tick_rate = df['tick_rate'].max()
            st.markdown(f"**Note:** Peak tick rate hit {peak_tick_rate:.1f} updates / sec in this window.")
        except Exception:
            pass

        # 4. VPIN (Toxicity)
        try:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df['timestamp'], y=df['vpin'], name='VPIN', line=dict(color='#d62728')))
            fig4.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig4.update_layout(title="VPIN (Toxicity)", xaxis_title="Time", yaxis_title="VPIN", showlegend=True)
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("VPIN (Toxicity): Quantifies order flow toxicity and potential for informed trading. High VPIN values (above 0.5) may indicate toxic, informed, or predatory trading.")
            # Dynamic commentary
            toxic_pct = (df['vpin'] > 0.5).mean() * 100
            st.markdown(f"**Flow toxicity:** {toxic_pct:.1f}% of ticks flagged as high‑risk (VPIN > 0.5).")
        except Exception:
            pass

        # 5. Manipulation Timeline
        try:
            all_events = (
                [{'time': e['timestamp'], 'type': e['type'], 'conf': e['confidence']} for e in self.session_state['spoofing_events']] +
                [{'time': e['start_time'], 'type': e['type'], 'conf': e['confidence']} for e in self.session_state['iceberg_events']]
            )
            if all_events:
                event_df = pd.DataFrame(all_events)
                fig5 = go.Figure()
                for event_type in event_df['type'].unique():
                    type_events = event_df[event_df['type'] == event_type]
                    fig5.add_trace(
                        go.Scatter(
                            x=type_events['time'], y=type_events['conf'],
                            mode='markers', name=event_type,
                            marker=dict(size=10)
                        )
                    )
                fig5.update_layout(title="Manipulation Timeline", xaxis_title="Time", yaxis_title="Confidence", showlegend=True)
                st.plotly_chart(fig5, use_container_width=True)
                st.caption("Manipulation Timeline: Plots the timing and confidence of detected spoofing and iceberg events. Each marker represents a detected event, with its confidence score.")
                # Dynamic commentary
                st.markdown(f"**Summary:** {len(self.session_state['spoofing_events'])} spoofing and {len(self.session_state['iceberg_events'])} iceberg events recorded in view.")
        except Exception:
            pass

        # 6. Iceberg Detection Map
        try:
            if self.session_state['iceberg_events']:
                iceberg_df = pd.DataFrame(self.session_state['iceberg_events'])
                fig6 = go.Figure()
                fig6.add_trace(
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
                    )
                )
                fig6.update_layout(title="Iceberg Detection Map", xaxis_title="Time", yaxis_title="Price Level", showlegend=False)
                st.plotly_chart(fig6, use_container_width=True)
                st.caption("Iceberg Detection Map: Shows locations and confidence of detected iceberg orders. Marker size is proportional to estimated order size; color indicates detection confidence.")
                # Dynamic commentary
                total_iceberg_vol = iceberg_df['estimated_total_size'].sum()
                st.markdown(f"**Aggregate:** Total inferred iceberg volume ≈ {total_iceberg_vol:,.0f} units.")
        except Exception:
            pass

        # 7. Order Flow Imbalance
        try:
            if 'bid_pressure' in df.columns and 'ask_pressure' in df.columns:
                df['order_flow_imbalance'] = (df['bid_pressure'] - df['ask_pressure']).rolling(50).mean()
                fig7 = go.Figure()
                fig7.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['order_flow_imbalance'],
                        name='Order Flow Imbalance',
                        fill='tozeroy',
                        line=dict(color='#9467bd')
                    )
                )
                fig7.update_layout(title="Order Flow Imbalance", xaxis_title="Time", yaxis_title="Imbalance", showlegend=True)
                st.plotly_chart(fig7, use_container_width=True)
                st.caption("Order Flow Imbalance: Indicates the net pressure from bids vs. asks over time. Positive values suggest buying pressure; negative values indicate selling pressure.")
                # Dynamic commentary
                peak_imb = df['order_flow_imbalance'].abs().max()
                st.markdown(f"**Highlight:** Max absolute imbalance reached {peak_imb:.2f}.")
        except Exception:
            pass

        # 8. Microstructure 3D Surface
        try:
            time_bins = pd.cut(df.index, bins=50, labels=False)
            price_bins = pd.cut(df['price_mid'], bins=50, labels=False)
            surface_data = pd.pivot_table(
                df, values='inferred_volume',
                index=price_bins, columns=time_bins,
                aggfunc='sum', fill_value=0
            )
            if not surface_data.empty:
                fig8 = go.Figure()
                fig8.add_trace(
                    go.Surface(
                        z=surface_data.values,
                        colorscale='Viridis',
                        name='Volume Surface'
                    )
                )
                fig8.update_layout(title="Microstructure 3D Surface", scene=dict(
                    xaxis_title="Time Bin",
                    yaxis_title="Price Bin",
                    zaxis_title="Volume"
                ))
                st.plotly_chart(fig8, use_container_width=True)
                st.caption("Microstructure 3D Surface: 3D visualization of volume distribution across price and time. Peaks indicate where large volumes concentrate in the price-time grid.")
                # Dynamic commentary
                st.markdown("**Interpretation:** Volume peaks mark hotspots where large orders clustered at specific price‑time zones.")
        except Exception:
            pass

        # 9. Hidden Liquidity Zones
        try:
            support_levels = df.groupby(df['price_mid'].round(0))['inferred_volume'].sum().sort_values(ascending=False).head(10)
            if not support_levels.empty:
                fig9 = go.Figure()
                fig9.add_trace(
                    go.Bar(
                        x=support_levels.values,
                        y=support_levels.index,
                        orientation='h',
                        name='Hidden Liquidity',
                        marker=dict(color='#8c564b')
                    )
                )
                fig9.update_layout(title="Hidden Liquidity Zones", xaxis_title="Inferred Volume", yaxis_title="Price Level", showlegend=False)
                st.plotly_chart(fig9, use_container_width=True)
                st.caption("Hidden Liquidity Zones: Highlights price levels with concentrated inferred volume. Bars show the top 10 price levels where large hidden orders may reside.")
                # Dynamic commentary
                top_level = support_levels.index[0]
                top_vol = support_levels.iloc[0]
                st.markdown(f"**Takeaway:** Level {top_level:.2f} holds the densest hidden liquidity (~{top_vol:,.0f} units).")
        except Exception:
            pass

        # 10. Quote Stuffing Density
        try:
            st.markdown("#### Quote Stuffing Density")
            if self.session_state.get('quote_stuffing_events'):
                qs_df = pd.DataFrame(self.session_state['quote_stuffing_events'])
                qs_df['duration_s'] = (qs_df['end_time'] - qs_df['start_time']).dt.total_seconds()
                fig_qs = go.Figure()
                fig_qs.add_trace(go.Bar(
                    x=qs_df['start_time'],
                    y=qs_df['updates_per_second'],
                    name='Updates / s',
                    marker=dict(color='#e377c2')
                ))
                fig_qs.update_layout(title="Quote Stuffing Density",
                                     xaxis_title="Time",
                                     yaxis_title="Updates per Second")
                st.plotly_chart(fig_qs, use_container_width=True)
                st.caption("Quote Stuffing Density: Bars represent detected stuffing periods; height indicates quote‑update rate during each burst.")
                # Dynamic commentary
                max_rate = qs_df['updates_per_second'].max()
                st.markdown(f"**Observation:** Highest burst reached {max_rate:.1f} quotes / sec across {len(qs_df)} stuffing episodes.")
            else:
                st.info("No quote stuffing episodes detected.")
        except Exception:
            pass

        # 11. Spoofing Patterns
        try:
            st.markdown("#### Spoofing Patterns")
            if self.session_state['spoofing_events']:
                sp_df = pd.DataFrame(self.session_state['spoofing_events'])
                sp_df['minute'] = sp_df['timestamp'].dt.floor('min')
                counts = sp_df.groupby('minute').size()
                fig_sp = go.Figure()
                fig_sp.add_trace(go.Bar(
                    x=counts.index,
                    y=counts.values,
                    name='Spoofing Events',
                    marker=dict(color='#17becf')
                ))
                fig_sp.update_layout(title="Spoofing Pattern Frequency",
                                     xaxis_title="Minute",
                                     yaxis_title="Number of Spoofing Events")
                st.plotly_chart(fig_sp, use_container_width=True)
                st.caption("Spoofing Patterns: Bars show how many spoofing spikes occurred in each minute.")
                # Dynamic commentary
                total_spoof = len(self.session_state['spoofing_events'])
                peak_minute = counts.max() if not counts.empty else 0
                st.markdown(f"**Insight:** {total_spoof} spoofing events total; busiest minute saw {peak_minute} events.")
            else:
                st.info("No spoofing events detected.")
        except Exception:
            pass

        # 12. Regime Transitions
        try:
            st.markdown("#### Regime Transitions")
            # Simple regime classification based on VPIN thresholds
            df['regime_simple'] = np.where(df['vpin'] > 0.7, 'manipulated',
                                   np.where(df['vpin'] > 0.5, 'stressed', 'normal'))
            regime_map = {'normal': 0, 'stressed': 1, 'manipulated': 2}
            df['regime_code'] = df['regime_simple'].map(regime_map)
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['regime_code'],
                mode='lines',
                line=dict(shape='hv'),
                name='Regime'
            ))
            fig_reg.update_layout(
                title="Regime Transitions",
                xaxis_title="Time",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(regime_map.values()),
                    ticktext=list(regime_map.keys())
                ),
                showlegend=False
            )
            st.plotly_chart(fig_reg, use_container_width=True)
            st.caption("Regime Transitions: The step‑line shows shifts between market regimes derived from VPIN thresholds (>0.7 = manipulated, >0.5 = stressed, else normal).")
            # Dynamic commentary
            regime_counts = df['regime_simple'].value_counts(normalize=True) * 100
            breakdown = ", ".join([f"{k}: {v:.1f}%" for k, v in regime_counts.items()])
            st.markdown(f"**Breakdown:** {breakdown}")
        except Exception:
            pass

        # Detailed Analysis Sections (unchanged)
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

        # Market Quality Metrics (unchanged)
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
    # DEBUG: show the resolved file path so we know exactly which file is being ingested
    st.sidebar.info(f"📂 Reading tick file: {file_path}")

    # Advanced options
    st.sidebar.markdown("### Advanced Settings")
    sensitivity = st.sidebar.slider("Detection Sensitivity", 0.5, 2.0, 1.0, 0.1)

    # Initialize analyzer
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'quantum_microstructure_config.yaml')

    analyzer = QuantumMicrostructureAnalyzer(config_path)

    # Load and analyze data
    with st.spinner("🧬 Running quantum analysis..."):
        # Now ingesting standard comma‑separated CSV files
        df = pd.read_csv(file_path, encoding_errors='ignore')  # default sep=','
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ensure data is in chronological order (oldest → newest) so all rolling/diff
        # calculations operate on the most‑recent bars instead of the earliest ones.
        df = df.sort_values('timestamp').reset_index(drop=True)
        # DEBUG: quick peek at the data to verify timestamps and latest candles
        with st.expander("🔎 Data preview (head & tail)", expanded=False):
            st.write("Head of data:")
            st.write(df.head())
            st.write("Tail of data:")
            st.write(df.tail())
        max_bars = st.sidebar.slider("Bars to Display", 50, len(df), min(1000, len(df)))
        df = df.tail(max_bars)
        st.caption(
            "This dashboard visualizes real-time tick data with derived microstructure metrics. Use it to detect spoofing, iceberg orders, quote stuffing, and assess trading toxicity via VPIN.")

        # FIXED: Preprocess data to ensure all columns exist
        df = analyzer.preprocess_data(df)

        # Heuristic sanity‑check: does this look like real tick data?
        is_mock, diag = analyzer.detect_mock_data(df)
        if is_mock:
            st.warning(
                "⚠️  The loaded file exhibits characteristics of synthetic/mock data. "
                f"Diagnostics: {diag}"
            )

        # Core analysis pipeline
        df = analyzer.infer_volume_from_ticks(df)
        df = analyzer.calculate_vpin(df)

        # Detect all manipulation patterns
        analyzer.session_state['iceberg_events'] = analyzer.detect_icebergs(df)
        analyzer.session_state['spoofing_events'] = analyzer.detect_spoofing(df)
        quote_stuffing = analyzer.detect_quote_stuffing(df)
        analyzer.session_state['quote_stuffing_events'] = quote_stuffing
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