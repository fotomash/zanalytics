# dashboard_smc_ultimate.py - Complete version with proper indentation

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="ncOS SMC Ultimate Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultimate SMC styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    .smc-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    .order-block-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .liquidity-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .wyckoff-phase {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2c3e50;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .signal-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 25px rgba(0,0,0,0.3);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .data-status {
        background: #2c3e50;
        color: #ecf0f1;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
class UltimateSMCConfig:
    def __init__(self):
        self.data_dir = "./data"
        self.max_bars = 1500
        self.pairs = ["XAUUSD", "BTCUSD", "GBPUSD"]
        self.timeframes = ["1T", "5T", "15T", "30T", "1H"]
        self.timeframe_map = {
            "1T": "1 Minute",
            "5T": "5 Minutes",
            "15T": "15 Minutes", 
            "30T": "30 Minutes",
            "1H": "1 Hour"
        }
        self.colors = {
            'bullish': '#26de81',
            'bearish': '#fc5c65',
            'bullish_ob': 'rgba(38, 222, 129, 0.3)',
            'bearish_ob': 'rgba(252, 92, 101, 0.3)',
            'supply': 'rgba(255, 107, 107, 0.2)',
            'demand': 'rgba(46, 213, 115, 0.2)',
            'liquidity': '#45aaf2',
            'fvg_bull': 'rgba(162, 155, 254, 0.2)',
            'fvg_bear': 'rgba(253, 121, 168, 0.2)',
            'poc': '#f39c12',
            'value_area': 'rgba(243, 156, 18, 0.1)',
            'harmonic': '#9b59b6',
            'wyckoff_acc': '#00cec9',
            'wyckoff_dist': '#e17055'
        }

# Ultimate Data Loader with ALL SMC features
class UltimateSMCDataLoader:
    def __init__(self, config: UltimateSMCConfig):
        self.config = config
        
    def scan_complete_data(self) -> Dict:
        """Scan for ALL SMC data including COMPREHENSIVE files"""
        data_map = {}
        
        for pair in self.config.pairs:
            pair_path = os.path.join(self.config.data_dir, pair)
            if not os.path.exists(pair_path):
                continue
                
            data_map[pair] = {
                "comprehensive_files": {},
                "summary_files": {},
                "analysis_report": None,
                "available_timeframes": []
            }
            
            # Find COMPREHENSIVE CSV files (these have ALL SMC data)
            for tf in self.config.timeframes:
                comp_file = os.path.join(pair_path, f"{pair}_M1_bars_COMPREHENSIVE_{tf}.csv")
                if os.path.exists(comp_file):
                    data_map[pair]["comprehensive_files"][tf] = comp_file
                    data_map[pair]["available_timeframes"].append(tf)
                    
                # Find corresponding SUMMARY JSON
                summary_file = os.path.join(pair_path, f"{pair}_M1_bars_SUMMARY_{tf}.json")
                if os.path.exists(summary_file):
                    data_map[pair]["summary_files"][tf] = summary_file
            
            # Find ANALYSIS REPORT
            analysis_report = os.path.join(pair_path, f"{pair}_M1_bars_ANALYSIS_REPORT.json")
            if os.path.exists(analysis_report):
                data_map[pair]["analysis_report"] = analysis_report
        
        return data_map
    
    def load_comprehensive_data(self, file_path: str, max_bars: int = None) -> pd.DataFrame:
        """Load COMPREHENSIVE CSV with ALL SMC indicators"""
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Sort and get latest bars
            df = df.sort_index()
            if max_bars:
                df = df.tail(max_bars)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading comprehensive data: {str(e)}")
            return None
    
    def load_smc_summary(self, file_path: str) -> Dict:
        """Load SMC summary JSON with all analysis results"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading summary: {str(e)}")
            return {}
    
    def extract_all_smc_features(self, df: pd.DataFrame, summary: Dict) -> Dict:
        """Extract ALL SMC features from comprehensive data and summary"""
        smc_features = {
            "order_blocks": self._extract_order_blocks(df, summary),
            "liquidity_zones": self._extract_liquidity_zones(df, summary),
            "fair_value_gaps": self._extract_fvg(df, summary),
            "supply_demand_zones": self._extract_supply_demand(df, summary),
            "bos_choch_points": self._extract_structure_breaks(df, summary),
            "pivot_points": self._extract_pivots(df, summary),
            "market_structure": self._extract_market_structure(df, summary),
            "volume_profile": self._extract_volume_profile(df, summary),
            "harmonic_patterns": self._extract_harmonics(summary),
            "wyckoff_analysis": self._extract_wyckoff(summary),
            "signals": self._extract_trading_signals(df, summary)
        }
        
        return smc_features
    
    def _extract_order_blocks(self, df: pd.DataFrame, summary: Dict) -> List[Dict]:
        """Extract order blocks from data"""
        order_blocks = []
        
        # From summary JSON
        if 'smc_analysis' in summary and 'order_blocks' in summary['smc_analysis']:
            order_blocks.extend(summary['smc_analysis']['order_blocks'])
        
        # From DataFrame columns if they exist
        if 'order_block_high' in df.columns and 'order_block_low' in df.columns:
            ob_mask = df['order_block_high'].notna() | df['order_block_low'].notna()
            for idx in df[ob_mask].index[-10:]:  # Last 10 order blocks
                order_blocks.append({
                    'timestamp': idx,
                    'high': df.loc[idx, 'order_block_high'] if pd.notna(df.loc[idx, 'order_block_high']) else df.loc[idx, 'high'],
                    'low': df.loc[idx, 'order_block_low'] if pd.notna(df.loc[idx, 'order_block_low']) else df.loc[idx, 'low'],
                    'type': 'bullish' if df.loc[idx, 'close'] > df.loc[idx, 'open'] else 'bearish'
                })
        
        return order_blocks
    
    def _extract_liquidity_zones(self, df: pd.DataFrame, summary: Dict) -> List[Dict]:
        """Extract liquidity zones"""
        zones = []
        
        # From summary
        if 'liquidity_analysis' in summary:
            if 'zones' in summary['liquidity_analysis']:
                zones.extend(summary['liquidity_analysis']['zones'])
            if 'levels' in summary['liquidity_analysis']:
                for level in summary['liquidity_analysis']['levels']:
                    zones.append({'level': level, 'type': 'liquidity'})
        
        # From price action - find swing highs/lows
        if len(df) > 20:
            highs = df['high'].rolling(10).max()
            lows = df['low'].rolling(10).min()
            
            # Recent swing highs as sell-side liquidity
            recent_highs = df[df['high'] == highs]['high'].tail(5)
            for idx, high in recent_highs.items():
                zones.append({
                    'timestamp': idx,
                    'level': high,
                    'type': 'sell_side_liquidity'
                })
            
            # Recent swing lows as buy-side liquidity
            recent_lows = df[df['low'] == lows]['low'].tail(5)
            for idx, low in recent_lows.items():
                zones.append({
                    'timestamp': idx,
                    'level': low,
                    'type': 'buy_side_liquidity'
                })
        
        return zones
    
    def _extract_fvg(self, df: pd.DataFrame, summary: Dict) -> List[Dict]:
        """Extract Fair Value Gaps"""
        fvgs = []
        
        # From summary
        if 'microstructure_analysis' in summary and 'fair_value_gaps' in summary['microstructure_analysis']:
            fvgs.extend(summary['microstructure_analysis']['fair_value_gaps'])
        
        # Calculate from price action
        if len(df) > 3:
            for i in range(2, len(df)):
                # Bullish FVG: Current low > Previous high
                if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                    fvgs.append({
                        'timestamp': df.index[i],
                        'high': df.iloc[i]['low'],
                        'low': df.iloc[i-2]['high'],
                        'type': 'bullish_fvg'
                    })
                
                # Bearish FVG: Current high < Previous low
                elif df.iloc[i]['high'] < df.iloc[i-2]['low']:
                    fvgs.append({
                        'timestamp': df.index[i],
                        'high': df.iloc[i-2]['low'],
                        'low': df.iloc[i]['high'],
                        'type': 'bearish_fvg'
                    })
        
        return fvgs[-20:]  # Last 20 FVGs
    
    def _extract_supply_demand(self, df: pd.DataFrame, summary: Dict) -> Dict:
        """Extract supply and demand zones"""
        zones = {'supply': [], 'demand': []}
        
        # From summary
        if 'supply_demand_analysis' in summary:
            if 'supply_zones' in summary['supply_demand_analysis']:
                zones['supply'] = summary['supply_demand_analysis']['supply_zones']
            if 'demand_zones' in summary['supply_demand_analysis']:
                zones['demand'] = summary['supply_demand_analysis']['demand_zones']
        
        # From DataFrame - find strong moves
        if 'volume' in df.columns and len(df) > 10:
            # High volume + big red candle = Supply zone
            vol_mean = df['volume'].mean()
            for i in range(10, len(df)):
                if df.iloc[i]['volume'] > vol_mean * 2:
                    if df.iloc[i]['close'] < df.iloc[i]['open']:  # Red candle
                        body = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
                        if body > df['close'].pct_change().std() * df.iloc[i]['close'] * 2:
                            zones['supply'].append({
                                'timestamp': df.index[i],
                                'high': df.iloc[i]['high'],
                                'low': df.iloc[i]['low'],
                                'strength': df.iloc[i]['volume'] / vol_mean
                            })
                    else:  # Green candle = Demand zone
                        body = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
                        if body > df['close'].pct_change().std() * df.iloc[i]['close'] * 2:
                            zones['demand'].append({
                                'timestamp': df.index[i],
                                'high': df.iloc[i]['high'],
                                'low': df.iloc[i]['low'],
                                'strength': df.iloc[i]['volume'] / vol_mean
                            })
        
        return zones
    
    def _extract_structure_breaks(self, df: pd.DataFrame, summary: Dict) -> Dict:
        """Extract BOS and CHoCH points"""
        structure = {'bos': [], 'choch': []}
        
        # From summary
        if 'market_structure' in summary:
            if 'break_of_structure' in summary['market_structure']:
                structure['bos'] = summary['market_structure']['break_of_structure']
            if 'change_of_character' in summary['market_structure']:
                structure['choch'] = summary['market_structure']['change_of_character']
        
        return structure
    
    def _extract_pivots(self, df: pd.DataFrame, summary: Dict) -> Dict:
        """Extract pivot points"""
        pivots = {'highs': [], 'lows': []}
        
        # From DataFrame
        if 'pivot_high' in df.columns:
            pivot_highs = df[df['pivot_high'].notna()]
            for idx in pivot_highs.index[-10:]:
                pivots['highs'].append({
                    'timestamp': idx,
                    'price': df.loc[idx, 'pivot_high']
                })
        
        if 'pivot_low' in df.columns:
            pivot_lows = df[df['pivot_low'].notna()]
            for idx in pivot_lows.index[-10:]:
                pivots['lows'].append({
                    'timestamp': idx,
                    'price': df.loc[idx, 'pivot_low']
                })
        
        return pivots
    
    def _extract_market_structure(self, df: pd.DataFrame, summary: Dict) -> Dict:
        """Extract overall market structure"""
        structure = {
            'trend': 'neutral',
            'strength': 0,
            'phase': 'ranging'
        }
        
        # From summary
        if 'market_analysis' in summary:
            structure.update(summary['market_analysis'])
        
        # Calculate from price
        if len(df) > 50:
            sma20 = df['close'].rolling(20).mean()
            sma50 = df['close'].rolling(50).mean()
            
            if df['close'].iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1]:
                structure['trend'] = 'bullish'
                structure['strength'] = min((df['close'].iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1] * 100, 100)
            elif df['close'].iloc[-1] < sma20.iloc[-1] < sma50.iloc[-1]:
                structure['trend'] = 'bearish' 
                structure['strength'] = min((sma50.iloc[-1] - df['close'].iloc[-1]) / sma50.iloc[-1] * 100, 100)
        
        return structure
    
    def _extract_volume_profile(self, df: pd.DataFrame, summary: Dict) -> Dict:
        """Extract volume profile data"""
        profile = {
            'poc': None,  # Point of Control
            'vah': None,  # Value Area High
            'val': None,  # Value Area Low
            'levels': []
        }
        
        if 'volume_profile' in summary:
            profile.update(summary['volume_profile'])
        
        # Calculate from data
        if 'volume' in df.columns and len(df) > 20:
            price_bins = pd.cut(df['close'], bins=20)
            volume_profile = df.groupby(price_bins)['volume'].sum()
            
            if len(volume_profile) > 0:
                poc_idx = volume_profile.idxmax()
                if poc_idx is not None:
                    profile['poc'] = poc_idx.mid
                
                # Value area (70% of volume)
                total_vol = volume_profile.sum()
                cumsum = 0
                va_indices = []
                
                sorted_profile = volume_profile.sort_values(ascending=False)
                for idx, vol in sorted_profile.items():
                    cumsum += vol
                    va_indices.append(idx)
                    if cumsum >= total_vol * 0.7:
                        break
                
                if va_indices:
                    all_mids = [idx.mid for idx in va_indices]
                    profile['vah'] = max(all_mids)
                    profile['val'] = min(all_mids)
        
        return profile
    
    def _extract_harmonics(self, summary: Dict) -> List[Dict]:
        """Extract harmonic patterns"""
        patterns = []
        
        if 'harmonic_patterns' in summary:
            patterns = summary['harmonic_patterns']
        
        return patterns
    
    def _extract_wyckoff(self, summary: Dict) -> Dict:
        """Extract Wyckoff analysis"""
        wyckoff = {
            'phase': 'Unknown',
            'events': [],
            'accumulation_zones': [],
            'distribution_zones': []
        }
        
        if 'wyckoff_analysis' in summary:
            wyckoff.update(summary['wyckoff_analysis'])
        
        return wyckoff
    
    def _extract_trading_signals(self, df: pd.DataFrame, summary: Dict) -> List[Dict]:
        """Extract trading signals"""
        signals = []
        
        if 'signals' in summary:
            signals = summary['signals']
        
        # Generate signals from indicators if available
        if 'signal' in df.columns:
            signal_df = df[df['signal'] != 0].tail(10)
            for idx in signal_df.index:
                signals.append({
                    'timestamp': idx,
                    'type': 'buy' if signal_df.loc[idx, 'signal'] > 0 else 'sell',
                    'price': signal_df.loc[idx, 'close'],
                    'strength': abs(signal_df.loc[idx, 'signal'])
                })
        
        return signals

# Ultimate Chart Generator with ALL SMC features  
class UltimateSMCChartGenerator:
    def __init__(self, config: UltimateSMCConfig):
        self.colors = config.colors
    
    def create_ultimate_smc_chart(self, df: pd.DataFrame, smc_features: Dict, pair: str, timeframe: str) -> go.Figure:
        """Create the ULTIMATE SMC chart with ALL features"""
        
        # Create figure with subplots
        fig = make_subplots(
            rows=5, cols=1,
            row_heights=[0.5, 0.15, 0.1, 0.1, 0.15],
            subplot_titles=[
                f"{pair} {timeframe} - Ultimate SMC Analysis",
                "Volume & Order Flow",
                "Market Structure",
                "RSI & Momentum", 
                "Signal Strength"
            ],
            vertical_spacing=0.02,
            shared_xaxes=True
        )
        
        # 1. Main Price Chart with ALL SMC overlays
        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )
        
        # Add ALL SMC features
        self._add_all_order_blocks(fig, df, smc_features['order_blocks'])
        self._add_all_liquidity_zones(fig, smc_features['liquidity_zones'])
        self._add_all_supply_demand_zones(fig, smc_features['supply_demand_zones'])
        self._add_all_fair_value_gaps(fig, smc_features['fair_value_gaps'])
        self._add_structure_breaks(fig, smc_features['bos_choch_points'])
        self._add_pivot_points(fig, smc_features['pivot_points'])
        self._add_volume_profile_levels(fig, smc_features['volume_profile'])
        self._add_harmonic_patterns(fig, smc_features['harmonic_patterns'])
        self._add_wyckoff_zones(fig, smc_features['wyckoff_analysis'])
        self._add_trading_signals(fig, smc_features['signals'])
        
        # Add technical indicators if available
        if 'ema_9' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['ema_9'], name="EMA 9", 
                          line=dict(color='orange', width=1)),
                row=1, col=1
            )
        
        if 'ema_21' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['ema_21'], name="EMA 21",
                          line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_upper'], name="BB Upper",
                          line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['bb_lower'], name="BB Lower",
                          line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
        
        # 2. Volume with delta
        if 'volume' in df.columns:
            colors = [self.colors['bullish'] if df['close'].iloc[i] > df['open'].iloc[i] 
                     else self.colors['bearish'] for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name="Volume",
                       marker_color=colors, opacity=0.7),
                row=2, col=1
            )
            
            # Add volume moving average
            if len(df) > 20:
                vol_ma = df['volume'].rolling(20).mean()
                fig.add_trace(
                    go.Scatter(x=df.index, y=vol_ma, name="Vol MA(20)",
                              line=dict(color='yellow', width=2)),
                    row=2, col=1
                )
        
        # 3. Market Structure Indicator
        if 'atr' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['atr'], name="ATR",
                          line=dict(color='purple', width=2)),
                row=3, col=1
            )
        
        # 4. RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], name="RSI",
                          line=dict(color='green', width=2)),
                row=4, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         row=4, col=1, annotation_text="OB")
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         row=4, col=1, annotation_text="OS")
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=4, col=1)
        
        # 5. Signal Strength
        if smc_features['signals']:
            signal_times = [s['timestamp'] for s in smc_features['signals']]
            signal_strengths = [s.get('strength', 1) for s in smc_features['signals']]
            signal_types = [s['type'] for s in smc_features['signals']]
            
            colors = [self.colors['bullish'] if t == 'buy' else self.colors['bearish'] 
                     for t in signal_types]
            
            fig.add_trace(
                go.Bar(x=signal_times, y=signal_strengths, name="Signals",
                       marker_color=colors),
                row=5, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{pair} {timeframe} - ncOS Ultimate SMC Analysis",
                font=dict(size=24, color='white')
            ),
            height=1200,
            showlegend=True,
            template="plotly_dark",
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="ATR", row=3, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)
        fig.update_yaxes(title_text="Signal", row=5, col=1)
        fig.update_xaxes(title_text="Time", row=5, col=1)
        
        return fig
    
    def _add_all_order_blocks(self, fig, df, order_blocks):
        """Add ALL order blocks with proper visualization"""
        for ob in order_blocks[-20:]:  # Last 20 OBs
            if 'timestamp' in ob:
                # Find the zone
                try:
                    idx_pos = df.index.get_loc(ob['timestamp'])
                    start_idx = max(0, idx_pos - 5)
                    end_idx = min(len(df) - 1, idx_pos + 20)
                    
                    color = self.colors['bullish_ob'] if ob.get('type') == 'bullish' else self.colors['bearish_ob']
                    
                    fig.add_shape(
                        type="rect",
                        x0=df.index[start_idx],
                        x1=df.index[end_idx],
                        y0=ob.get('low', df.iloc[idx_pos]['low']),
                        y1=ob.get('high', df.iloc[idx_pos]['high']),
                        fillcolor=color,
                        line=dict(color=color.replace('0.3', '1'), width=2),
                        row=1, col=1
                    )
                    
                    # Add label
                    fig.add_annotation(
                        x=ob['timestamp'],
                        y=ob.get('high', df.iloc[idx_pos]['high']),
                        text=f"OB-{ob.get('type', 'N/A').upper()}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=color.replace('0.3', '1'),
                        row=1, col=1
                    )
                except:
                    pass
    
    def _add_all_liquidity_zones(self, fig, zones):
        """Add ALL liquidity zones"""
        for zone in zones[-15:]:  # Last 15 zones
            if 'level' in zone:
                zone_type = zone.get('type', 'liquidity')
                color = self.colors['liquidity']
                
                if 'sell_side' in zone_type:
                    color = self.colors['bearish']
                    annotation = "SSL"
                elif 'buy_side' in zone_type:
                    color = self.colors['bullish']
                    annotation = "BSL"
                else:
                    annotation = "LIQ"
                
                fig.add_hline(
                    y=zone['level'],
                    line_dash="dot",
                    line_color=color,
                    annotation_text=annotation,
                    annotation_position="right",
                    row=1, col=1
                )
    
    def _add_all_supply_demand_zones(self, fig, zones):
        """Add ALL supply and demand zones"""
        # Supply zones
        for zone in zones.get('supply', [])[-10:]:
            if 'timestamp' in zone:
                try:
                    fig.add_shape(
                        type="rect",
                        x0=zone['timestamp'],
                        x1=zone['timestamp'] + pd.Timedelta(hours=1),
                        y0=zone.get('low', 0),
                        y1=zone.get('high', 0),
                        fillcolor=self.colors['supply'],
                        line=dict(color='red', width=1, dash='dash'),
                        row=1, col=1
                    )
                except:
                    pass
        
        # Demand zones
        for zone in zones.get('demand', [])[-10:]:
            if 'timestamp' in zone:
                try:
                    fig.add_shape(
                        type="rect",
                        x0=zone['timestamp'],
                        x1=zone['timestamp'] + pd.Timedelta(hours=1),
                        y0=zone.get('low', 0),
                        y1=zone.get('high', 0),
                        fillcolor=self.colors['demand'],
                        line=dict(color='green', width=1, dash='dash'),
                        row=1, col=1
                    )
                except:
                    pass
    
    def _add_all_fair_value_gaps(self, fig, fvgs):
        """Add ALL Fair Value Gaps"""
        for fvg in fvgs:
            if 'timestamp' in fvg:
                color = self.colors['fvg_bull'] if 'bullish' in fvg.get('type', '') else self.colors['fvg_bear']
                
                try:
                    fig.add_shape(
                        type="rect",
                        x0=fvg['timestamp'],
                        x1=fvg['timestamp'] + pd.Timedelta(minutes=30),
                        y0=fvg.get('low', 0),
                        y1=fvg.get('high', 0),
                        fillcolor=color,
                        line=dict(color=color.replace('0.2', '1'), width=1),
                        row=1, col=1
                    )
                    
                    # Add FVG label
                    fig.add_annotation(
                        x=fvg['timestamp'],
                        y=(fvg.get('high', 0) + fvg.get('low', 0)) / 2,
                        text="FVG",
                        font=dict(size=8, color='white'),
                        bgcolor=color.replace('0.2', '0.8'),
                        row=1, col=1
                    )
                except:
                    pass
    
    def _add_structure_breaks(self, fig, structure):
        """Add BOS and CHoCH points"""
        # BOS
        for bos in structure.get('bos', [])[-10:]:
            if 'timestamp' in bos:
                fig.add_annotation(
                    x=bos['timestamp'],
                    y=bos.get('price', 0),
                    text="BOS",
                    showarrow=True,
                    arrowhead=4,
                    arrowcolor='yellow',
                    bgcolor='yellow',
                    font=dict(color='black', size=10),
                    row=1, col=1
                )
        
        # CHoCH
        for choch in structure.get('choch', [])[-10:]:
            if 'timestamp' in choch:
                fig.add_annotation(
                    x=choch['timestamp'],
                    y=choch.get('price', 0),
                    text="CHoCH",
                    showarrow=True,
                    arrowhead=3,
                    arrowcolor='orange',
                    bgcolor='orange',
                    font=dict(color='black', size=10),
                    row=1, col=1
                )
    
    def _add_pivot_points(self, fig, pivots):
        """Add pivot highs and lows"""
        # Pivot highs
        for ph in pivots.get('highs', []):
            if 'timestamp' in ph:
                fig.add_annotation(
                    x=ph['timestamp'],
                    y=ph['price'],
                    text="PH",
                    showarrow=False,
                    font=dict(size=8, color='red'),
                    row=1, col=1
                )
        
        # Pivot lows
        for pl in pivots.get('lows', []):
            if 'timestamp' in pl:
                fig.add_annotation(
                    x=pl['timestamp'],
                    y=pl['price'],
                    text="PL",
                    showarrow=False,
                    font=dict(size=8, color='green'),
                    row=1, col=1
                )
    
    def _add_volume_profile_levels(self, fig, profile):
        """Add volume profile levels (POC, VAH, VAL)"""
        if profile['poc']:
            fig.add_hline(
                y=profile['poc'],
                line_dash="solid",
                line_color=self.colors['poc'],
                line_width=3,
                annotation_text="POC",
                annotation_position="right",
                row=1, col=1
            )
        
        if profile['vah'] and profile['val']:
            # Value area rectangle
            fig.add_hrect(
                y0=profile['val'],
                y1=profile['vah'],
                fillcolor=self.colors['value_area'],
                line_width=0,
                row=1, col=1
            )
            
            # VAH line
            fig.add_hline(
                y=profile['vah'],
                line_dash="dash",
                line_color=self.colors['poc'],
                annotation_text="VAH",
                annotation_position="right",
                row=1, col=1
            )
            
            # VAL line
            fig.add_hline(
                y=profile['val'],
                line_dash="dash",
                line_color=self.colors['poc'],
                annotation_text="VAL",
                annotation_position="right",
                row=1, col=1
            )
    
    def _add_harmonic_patterns(self, fig, patterns):
        """Add harmonic patterns"""
        for pattern in patterns[-5:]:  # Last 5 patterns
            if 'points' in pattern:
                # Draw pattern lines
                points = pattern['points']
                if len(points) >= 4:
                    # XABCD pattern
                    x_vals = [p.get('timestamp') for p in points if 'timestamp' in p]
                    y_vals = [p.get('price', 0) for p in points]
                    
                    if x_vals:
                        fig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=y_vals,
                                mode='lines+markers',
                                line=dict(color=self.colors['harmonic'], width=2, dash='dash'),
                                marker=dict(size=8, color=self.colors['harmonic']),
                                name=f"Harmonic-{pattern.get('type', 'Pattern')}",
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        
                        # Add pattern label
                        if x_vals and y_vals:
                            fig.add_annotation(
                                x=x_vals[-1],
                                y=y_vals[-1],
                                text=pattern.get('type', 'Harmonic'),
                                bgcolor=self.colors['harmonic'],
                                font=dict(color='white', size=10),
                                row=1, col=1
                            )
    
    def _add_wyckoff_zones(self, fig, wyckoff):
        """Add Wyckoff accumulation/distribution zones"""
        # Current phase background
        if wyckoff['phase'] != 'Unknown':
            phase_color = self.colors['wyckoff_acc'] if 'accumulation' in wyckoff['phase'].lower() else self.colors['wyckoff_dist']
            
            # Add phase annotation
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                text=f"Wyckoff: {wyckoff['phase']}",
                showarrow=False,
                bgcolor=phase_color,
                font=dict(color='white', size=12),
                row=1, col=1
            )
        
        # Accumulation zones
        for zone in wyckoff.get('accumulation_zones', [])[-5:]:
            if 'start' in zone and 'end' in zone:
                try:
                    fig.add_vrect(
                        x0=zone['start'],
                        x1=zone['end'],
                        fillcolor=self.colors['wyckoff_acc'],
                        opacity=0.2,
                        annotation_text="Accumulation",
                        annotation_position="top left",
                        row=1, col=1
                    )
                except:
                    pass
        
        # Distribution zones
        for zone in wyckoff.get('distribution_zones', [])[-5:]:
            if 'start' in zone and 'end' in zone:
                try:
                    fig.add_vrect(
                        x0=zone['start'],
                        x1=zone['end'],
                        fillcolor=self.colors['wyckoff_dist'],
                        opacity=0.2,
                        annotation_text="Distribution",
                        annotation_position="top left",
                        row=1, col=1
                    )
                except:
                    pass
    
    def _add_trading_signals(self, fig, signals):
        """Add trading signals with arrows"""
        for signal in signals[-20:]:  # Last 20 signals
            if 'timestamp' in signal and 'price' in signal:
                arrow_color = self.colors['bullish'] if signal['type'] == 'buy' else self.colors['bearish']
                arrow_symbol = 5 if signal['type'] == 'buy' else 6  # Up or down arrow
                y_shift = -50 if signal['type'] == 'buy' else 50
                
                fig.add_annotation(
                    x=signal['timestamp'],
                    y=signal['price'],
                    text=signal['type'].upper(),
                    showarrow=True,
                    arrowhead=arrow_symbol,
                    arrowsize=2,
                    arrowwidth=2,
                    arrowcolor=arrow_color,
                    ax=0,
                    ay=y_shift,
                    bgcolor=arrow_color,
                    font=dict(color='white', size=10),
                    row=1, col=1
                )

# Main Ultimate Dashboard
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 ncOS Ultimate SMC Intelligence Dashboard</h1>
        <p>Complete Smart Money Concepts Analysis with ALL Features</p>
        <p><em>Processing COMPREHENSIVE data with full SMC suite</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize
    config = UltimateSMCConfig()
    loader = UltimateSMCDataLoader(config)
    chart_gen = UltimateSMCChartGenerator(config)
    
    # Scan data
    data_map = loader.scan_complete_data()
    
    if not data_map:
        st.error("❌ No data found in ./data directory")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Ultimate SMC Controls")
    
    # Data status
    st.sidebar.markdown("### 📊 Data Status")
    for pair, info in data_map.items():
        if info['available_timeframes']:
            st.sidebar.markdown(f"""
            <span class="data-status">✅ {pair}: {len(info['available_timeframes'])} TFs</span>
            """, unsafe_allow_html=True)
    
    # Pair selection
    st.sidebar.markdown("### 💱 Currency Pair")
    available_pairs = [p for p, info in data_map.items() if info['available_timeframes']]
    
    if not available_pairs:
        st.error("❌ No pairs with COMPREHENSIVE data found")
        return
    
    selected_pair = st.sidebar.selectbox(
        "Select Pair:",
        options=available_pairs,
        key="ultimate_pair"
    )
    
    # Timeframe selection
    pair_info = data_map[selected_pair]
    st.sidebar.markdown("### ⏰ Timeframe")
    
    selected_tf = st.sidebar.selectbox(
        "Select Timeframe:",
        options=pair_info['available_timeframes'],
        format_func=lambda x: config.timeframe_map.get(x, x),
        key="ultimate_tf"
    )
    
    # Bar limit
    max_bars = st.sidebar.number_input(
        "📊 Max Bars (Latest First)",
        min_value=100,
        max_value=5000,
        value=config.max_bars,
        step=100,
        key="ultimate_bars"
    )
    
    # Feature toggles
    with st.sidebar.expander("🎯 SMC Features", expanded=True):
        show_order_blocks = st.checkbox("Order Blocks", value=True)
        show_liquidity = st.checkbox("Liquidity Zones", value=True)
        show_supply_demand = st.checkbox("Supply/Demand Zones", value=True)
        show_fvg = st.checkbox("Fair Value Gaps", value=True)
        show_structure = st.checkbox("BOS/CHoCH", value=True)
        show_pivots = st.checkbox("Pivot Points", value=True)
        show_volume_profile = st.checkbox("Volume Profile", value=True)
        show_harmonics = st.checkbox("Harmonic Patterns", value=True)
        show_wyckoff = st.checkbox("Wyckoff Analysis", value=True)
        show_signals = st.checkbox("Trading Signals", value=True)
    
    # Load and process data
    if selected_pair and selected_tf:
# Load COMPREHENSIVE data
        comp_file = pair_info['comprehensive_files'].get(selected_tf)
        summary_file = pair_info['summary_files'].get(selected_tf)
        
        if not comp_file:
            st.error(f"❌ No COMPREHENSIVE data for {selected_pair} {selected_tf}")
            return
        
        with st.spinner(f"🔄 Loading {selected_pair} {selected_tf} COMPREHENSIVE data..."):
            # Load data
            df = loader.load_comprehensive_data(comp_file, max_bars)
            
            if df is None:
                st.error("❌ Failed to load data")
                return
            
            # Load summary
            summary = {}
            if summary_file:
                summary = loader.load_smc_summary(summary_file)
            
            # Extract ALL SMC features
            smc_features = loader.extract_all_smc_features(df, summary)
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            current_price = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close != 0 else 0
            
            st.markdown(f"""
            <div class="smc-metric">
                <h3>💰 Price</h3>
                <h2>{current_price:.5f}</h2>
                <p>{change:+.5f} ({change_pct:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            market_structure = smc_features['market_structure']
            trend_color = "#00ff88" if market_structure['trend'] == 'bullish' else "#ff4757"
            
            st.markdown(f"""
            <div class="smc-metric">
                <h3>📈 Trend</h3>
                <h2 style="color: {trend_color}">{market_structure['trend'].upper()}</h2>
                <p>Strength: {market_structure.get('strength', 0):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            order_blocks_count = len(smc_features['order_blocks'])
            liquidity_count = len(smc_features['liquidity_zones'])
            
            st.markdown(f"""
            <div class="smc-metric">
                <h3>🎯 SMC Levels</h3>
                <h2>{order_blocks_count + liquidity_count}</h2>
                <p>OB: {order_blocks_count} | LIQ: {liquidity_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            wyckoff_phase = smc_features['wyckoff_analysis']['phase']
            phase_color = "#00cec9" if 'accumulation' in wyckoff_phase.lower() else "#e17055"
            
            st.markdown(f"""
            <div class="smc-metric">
                <h3>📊 Wyckoff</h3>
                <h2 style="color: {phase_color}; font-size: 1.2rem">{wyckoff_phase}</h2>
                <p>Events: {len(smc_features['wyckoff_analysis'].get('events', []))}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            signals_count = len(smc_features['signals'])
            buy_signals = sum(1 for s in smc_features['signals'] if s.get('type') == 'buy')
            sell_signals = signals_count - buy_signals
            
            st.markdown(f"""
            <div class="smc-metric">
                <h3>🚀 Signals</h3>
                <h2>{signals_count}</h2>
                <p>Buy: {buy_signals} | Sell: {sell_signals}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key insights
        st.markdown("### 🔍 Key SMC Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if smc_features['order_blocks']:
                latest_ob = smc_features['order_blocks'][-1]
                ob_type = latest_ob.get('type', 'unknown')
                
                st.markdown(f"""
                <div class="order-block-card">
                    <h4>📦 Latest Order Block</h4>
                    <p>Type: <strong>{ob_type.upper()}</strong></p>
                    <p>Level: {latest_ob.get('high', 0):.5f} - {latest_ob.get('low', 0):.5f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if smc_features['liquidity_zones']:
                latest_liq = smc_features['liquidity_zones'][-1]
                liq_type = latest_liq.get('type', 'liquidity')
                
                st.markdown(f"""
                <div class="liquidity-card">
                    <h4>💧 Latest Liquidity</h4>
                    <p>Type: <strong>{liq_type.upper()}</strong></p>
                    <p>Level: {latest_liq.get('level', 0):.5f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if smc_features['volume_profile']['poc']:
                st.markdown(f"""
                <div class="wyckoff-phase">
                    <h4>🎯 Volume Profile</h4>
                    <p>POC: {smc_features['volume_profile']['poc']:.5f}</p>
                    <p>VAH: {smc_features['volume_profile'].get('vah', 0):.5f}</p>
                    <p>VAL: {smc_features['volume_profile'].get('val', 0):.5f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Latest signal alert
        if smc_features['signals']:
            latest_signal = smc_features['signals'][-1]
            signal_color = "#00ff88" if latest_signal['type'] == 'buy' else "#ff4757"
            
            st.markdown(f"""
            <div class="signal-alert">
                <h3>⚡ LATEST SIGNAL: {latest_signal['type'].upper()}</h3>
                <p>Price: {latest_signal.get('price', 0):.5f} | Strength: {latest_signal.get('strength', 0):.1f}</p>
                <p>Time: {latest_signal.get('timestamp', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create the ULTIMATE chart
        st.markdown("### 📊 Ultimate SMC Analysis Chart")
        
        # Filter features based on toggles
        filtered_features = smc_features.copy()
        if not show_order_blocks:
            filtered_features['order_blocks'] = []
        if not show_liquidity:
            filtered_features['liquidity_zones'] = []
        if not show_supply_demand:
            filtered_features['supply_demand_zones'] = {'supply': [], 'demand': []}
        if not show_fvg:
            filtered_features['fair_value_gaps'] = []
        if not show_structure:
            filtered_features['bos_choch_points'] = {'bos': [], 'choch': []}
        if not show_pivots:
            filtered_features['pivot_points'] = {'highs': [], 'lows': []}
        if not show_volume_profile:
            filtered_features['volume_profile'] = {'poc': None, 'vah': None, 'val': None}
        if not show_harmonics:
            filtered_features['harmonic_patterns'] = []
        if not show_wyckoff:
            filtered_features['wyckoff_analysis'] = {'phase': 'Unknown', 'events': [], 
                                                     'accumulation_zones': [], 'distribution_zones': []}
        if not show_signals:
            filtered_features['signals'] = []
        
        # Generate chart
        chart = chart_gen.create_ultimate_smc_chart(
            df, 
            filtered_features,
            selected_pair,
            config.timeframe_map.get(selected_tf, selected_tf)
        )
        
        st.plotly_chart(chart, use_container_width=True)
        
        # Detailed analysis sections
        with st.expander("📊 Detailed SMC Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📦 Order Blocks Analysis")
                if smc_features['order_blocks']:
                    ob_df = pd.DataFrame(smc_features['order_blocks'][-10:])
                    st.dataframe(ob_df, use_container_width=True)
                else:
                    st.info("No order blocks detected")
                
                st.markdown("#### 💧 Liquidity Zones")
                if smc_features['liquidity_zones']:
                    liq_df = pd.DataFrame(smc_features['liquidity_zones'][-10:])
                    st.dataframe(liq_df, use_container_width=True)
                else:
                    st.info("No liquidity zones detected")
            
            with col2:
                st.markdown("#### 🎯 Fair Value Gaps")
                if smc_features['fair_value_gaps']:
                    fvg_df = pd.DataFrame(smc_features['fair_value_gaps'][-10:])
                    st.dataframe(fvg_df, use_container_width=True)
                else:
                    st.info("No FVGs detected")
                
                st.markdown("#### 📊 Market Structure")
                st.json(smc_features['market_structure'])
        
        # Data preview
        with st.expander("📋 Raw Data Preview (Latest First)", expanded=False):
            # Show comprehensive data columns
            st.markdown(f"**Available Indicators:** {', '.join(df.columns.tolist())}")
            
            # Display latest data
            display_df = df.iloc[::-1].head(50)  # Reverse for latest first
            st.dataframe(display_df, use_container_width=True, height=400)
        
        # Summary JSON preview
        if summary:
            with st.expander("🔍 Complete Analysis Summary", expanded=False):
                st.json(summary)
        
        # Export section
        st.markdown("### 💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Chart as HTML", key="export_chart"):
                html_str = chart.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="Download Chart HTML",
                    data=html_str,
                    file_name=f"{selected_pair}_{selected_tf}_smc_chart.html",
                    mime="text/html"
                )
        
        with col2:
            if st.button("📋 Export Analysis Report", key="export_analysis"):
                analysis_report = {
                    "pair": selected_pair,
                    "timeframe": selected_tf,
                    "timestamp": datetime.now().isoformat(),
                    "bars_analyzed": len(df),
                    "latest_price": float(df['close'].iloc[-1]),
                    "smc_features": {
                        "order_blocks": len(smc_features['order_blocks']),
                        "liquidity_zones": len(smc_features['liquidity_zones']),
                        "fair_value_gaps": len(smc_features['fair_value_gaps']),
                        "signals": len(smc_features['signals'])
                    },
                    "market_structure": smc_features['market_structure'],
                    "wyckoff_phase": smc_features['wyckoff_analysis']['phase'],
                    "volume_profile": smc_features['volume_profile']
                }
                
                st.download_button(
                    label="Download Analysis JSON",
                    data=json.dumps(analysis_report, indent=2),
                    file_name=f"{selected_pair}_{selected_tf}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📊 Export Filtered Data", key="export_data"):
                export_df = df.copy()
                export_df.index = export_df.index.strftime('%Y-%m-%d %H:%M:%S')
                
                csv_data = export_df.to_csv()
                st.download_button(
                    label="Download CSV Data",
                    data=csv_data,
                    file_name=f"{selected_pair}_{selected_tf}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Performance metrics
        st.markdown("### 📈 Performance Metrics")
        
        if len(df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Calculate returns
                returns = df['close'].pct_change().dropna()
                sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col2:
                # Max drawdown
                cumret = (1 + returns).cumprod()
                running_max = cumret.expanding().max()
                drawdown = (cumret - running_max) / running_max
                max_dd = drawdown.min()
                st.metric("Max Drawdown", f"{max_dd:.2%}")
            
            with col3:
                # Win rate from signals
                if smc_features['signals']:
                    # Simple win rate calculation
                    wins = sum(1 for i, s in enumerate(smc_features['signals'][:-1]) 
                              if s['type'] == 'buy' and df['close'].iloc[-1] > s.get('price', 0))
                    win_rate = (wins / len(smc_features['signals']) * 100) if smc_features['signals'] else 0
                    st.metric("Signal Win Rate", f"{win_rate:.1f}%")
                else:
                    st.metric("Signal Win Rate", "N/A")
            
            with col4:
                # Volatility
                volatility = returns.std() * np.sqrt(252) * 100
                st.metric("Annual Volatility", f"{volatility:.1f}%")
        
        # Footer with refresh option
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔄 Refresh Data", key="refresh_data", use_container_width=True):
                st.rerun()
        
        # About section
        with st.expander("ℹ️ About ncOS Ultimate SMC Dashboard", expanded=False):
            st.markdown("""
            ### 🎯 Features
            - **Complete SMC Analysis**: Order Blocks, Liquidity Zones, FVG, Supply/Demand
            - **Market Structure**: BOS/CHoCH detection, Pivot Points
            - **Volume Profile**: POC, VAH, VAL visualization
            - **Harmonic Patterns**: Advanced pattern recognition
            - **Wyckoff Analysis**: Accumulation/Distribution phases
            - **Trading Signals**: Entry/Exit points with strength indicators
            
            ### 📊 Data Processing
            - Reads COMPREHENSIVE CSV files with all technical indicators
            - Parses SUMMARY JSON files for complete analysis
            - Configurable bar limits for performance
            - Real-time feature toggling
            
            ### 🚀 Performance
            - Optimized for large datasets
            - Efficient memory usage
            - Fast chart rendering
            """)

if __name__ == "__main__":
    main()