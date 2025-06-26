
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import requests
import ta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ZANFLOW v13 ULTIMATE MEGA DASHBOARD - COMPLETE IMPLEMENTATION
# Integrating Wyckoff, SMC, Liquidity Analysis, and Economic Data

class WyckoffAnalyzer:
    """Complete Wyckoff Analysis with proper phase detection"""
    
    def __init__(self):
        self.phases = {
            'accumulation': {
                'PS': 'Preliminary Support',
                'SC': 'Selling Climax',
                'AR': 'Automatic Rally',
                'ST': 'Secondary Test',
                'SPRING': 'Spring/Shakeout',
                'SOS': 'Sign of Strength',
                'LPS': 'Last Point of Support',
                'BU': 'Back-Up'
            },
            'distribution': {
                'PSY': 'Preliminary Supply',
                'BC': 'Buying Climax',
                'AR': 'Automatic Reaction',
                'ST': 'Secondary Test',
                'SOW': 'Sign of Weakness',
                'LPSY': 'Last Point of Supply',
                'UTAD': 'Upthrust After Distribution'
            }
        }
        
    def detect_trading_range(self, df, lookback=50):
        """Detect Trading Range boundaries"""
        high_range = df['High'].rolling(lookback).max()
        low_range = df['Low'].rolling(lookback).min()
        
        # Calculate TR metrics
        tr_height = (high_range - low_range) / low_range * 100
        volume_profile = df['Volume'].rolling(lookback).mean()
        
        return {
            'upper_boundary': high_range,
            'lower_boundary': low_range,
            'tr_height_pct': tr_height,
            'avg_volume': volume_profile,
            'is_tr': tr_height < 15  # Less than 15% range = TR
        }
    
    def detect_springs_utad(self, df, tr_data):
        """Detect Springs and UTAD with volume confirmation"""
        springs = []
        utads = []
        
        for i in range(20, len(df)-1):
            # Spring detection - break below support with snapback
            if df['Low'].iloc[i] < tr_data['lower_boundary'].iloc[i]:
                if df['Close'].iloc[i] > df['Low'].iloc[i] + (df['High'].iloc[i] - df['Low'].iloc[i]) * 0.5:
                    # Volume confirmation
                    if df['Volume'].iloc[i] > df['Volume'].rolling(20).mean().iloc[i] * 1.5:
                        springs.append({
                            'index': i,
                            'date': df.index[i],
                            'low': df['Low'].iloc[i],
                            'snapback_strength': (df['Close'].iloc[i] - df['Low'].iloc[i]) / (df['High'].iloc[i] - df['Low'].iloc[i])
                        })
            
            # UTAD detection - break above resistance with failure
            if df['High'].iloc[i] > tr_data['upper_boundary'].iloc[i]:
                if df['Close'].iloc[i] < df['High'].iloc[i] - (df['High'].iloc[i] - df['Low'].iloc[i]) * 0.5:
                    if df['Volume'].iloc[i] > df['Volume'].rolling(20).mean().iloc[i] * 1.5:
                        utads.append({
                            'index': i,
                            'date': df.index[i],
                            'high': df['High'].iloc[i],
                            'rejection_strength': (df['High'].iloc[i] - df['Close'].iloc[i]) / (df['High'].iloc[i] - df['Low'].iloc[i])
                        })
        
        return springs, utads
    
    def detect_phases(self, df, tr_data, springs, utads):
        """Detect Wyckoff phases with proper logic"""
        phases = []
        
        # Phase detection logic based on document specifications
        for i in range(50, len(df)):
            # Check for accumulation phases
            if tr_data['is_tr'].iloc[i]:
                # Phase A - Stopping action
                if self._is_selling_climax(df, i):
                    phases.append({
                        'type': 'Phase A - SC',
                        'index': i,
                        'phase': 'accumulation'
                    })
                
                # Phase C - Spring
                for spring in springs:
                    if abs(spring['index'] - i) < 5:
                        phases.append({
                            'type': 'Phase C - Spring',
                            'index': i,
                            'phase': 'accumulation',
                            'strength': spring['snapback_strength']
                        })
                
                # Phase D - SOS
                if self._is_sign_of_strength(df, i, tr_data):
                    phases.append({
                        'type': 'Phase D - SOS',
                        'index': i,
                        'phase': 'accumulation'
                    })
        
        return phases
    
    def _is_selling_climax(self, df, i):
        """Detect selling climax pattern"""
        return (df['Volume'].iloc[i] > df['Volume'].rolling(20).mean().iloc[i] * 2 and
                df['Close'].iloc[i] > df['Low'].iloc[i] + (df['High'].iloc[i] - df['Low'].iloc[i]) * 0.6)
    
    def _is_sign_of_strength(self, df, i, tr_data):
        """Detect SOS - price advance on increasing volume"""
        return (df['Close'].iloc[i] > tr_data['upper_boundary'].iloc[i-5] and
                df['Volume'].iloc[i] > df['Volume'].rolling(10).mean().iloc[i])

class SMCAnalyzer:
    """Smart Money Concepts Analysis"""
    
    def __init__(self):
        self.liquidity_levels = []
        self.order_blocks = []
        self.fvgs = []
        
    def detect_liquidity_sweeps(self, df, lookback=20):
        """Detect liquidity sweeps with snapback"""
        sweeps = []
        
        for i in range(lookback, len(df)-1):
            # Previous high/low levels
            prev_high = df['High'].iloc[i-lookback:i].max()
            prev_low = df['Low'].iloc[i-lookback:i].min()
            
            # Bullish sweep - spike below low with snapback
            if df['Low'].iloc[i] < prev_low:
                wick_ratio = (df['Close'].iloc[i] - df['Low'].iloc[i]) / (df['High'].iloc[i] - df['Low'].iloc[i])
                if wick_ratio > 0.7:  # Strong rejection
                    sweeps.append({
                        'type': 'bullish_sweep',
                        'index': i,
                        'level': prev_low,
                        'sweep_low': df['Low'].iloc[i],
                        'rejection_strength': wick_ratio,
                        'volume_spike': df['Volume'].iloc[i] / df['Volume'].rolling(20).mean().iloc[i]
                    })
            
            # Bearish sweep - spike above high with snapback
            if df['High'].iloc[i] > prev_high:
                wick_ratio = (df['High'].iloc[i] - df['Close'].iloc[i]) / (df['High'].iloc[i] - df['Low'].iloc[i])
                if wick_ratio > 0.7:
                    sweeps.append({
                        'type': 'bearish_sweep',
                        'index': i,
                        'level': prev_high,
                        'sweep_high': df['High'].iloc[i],
                        'rejection_strength': wick_ratio,
                        'volume_spike': df['Volume'].iloc[i] / df['Volume'].rolling(20).mean().iloc[i]
                    })
        
        return sweeps
    
    def detect_fvgs(self, df):
        """Detect Fair Value Gaps (imbalances)"""
        fvgs = []
        
        for i in range(2, len(df)):
            # Bullish FVG
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                fvgs.append({
                    'type': 'bullish_fvg',
                    'index': i,
                    'upper': df['Low'].iloc[i],
                    'lower': df['High'].iloc[i-2],
                    'midpoint': (df['Low'].iloc[i] + df['High'].iloc[i-2]) / 2
                })
            
            # Bearish FVG
            if df['High'].iloc[i] < df['Low'].iloc[i-2]:
                fvgs.append({
                    'type': 'bearish_fvg',
                    'index': i,
                    'upper': df['Low'].iloc[i-2],
                    'lower': df['High'].iloc[i],
                    'midpoint': (df['Low'].iloc[i-2] + df['High'].iloc[i]) / 2
                })
        
        return fvgs
    
    def detect_order_blocks(self, df, sweeps):
        """Detect Order Blocks at sweep origins"""
        order_blocks = []
        
        for sweep in sweeps:
            idx = sweep['index']
            
            # Find the last opposing candle before sweep
            if sweep['type'] == 'bullish_sweep':
                # Find last bearish candle before sweep
                for j in range(idx-1, max(0, idx-10), -1):
                    if df['Close'].iloc[j] < df['Open'].iloc[j]:
                        order_blocks.append({
                            'type': 'bullish_ob',
                            'index': j,
                            'upper': df['High'].iloc[j],
                            'lower': df['Low'].iloc[j],
                            'sweep_index': idx
                        })
                        break
            
            elif sweep['type'] == 'bearish_sweep':
                # Find last bullish candle before sweep
                for j in range(idx-1, max(0, idx-10), -1):
                    if df['Close'].iloc[j] > df['Open'].iloc[j]:
                        order_blocks.append({
                            'type': 'bearish_ob',
                            'index': j,
                            'upper': df['High'].iloc[j],
                            'lower': df['Low'].iloc[j],
                            'sweep_index': idx
                        })
                        break
        
        return order_blocks
    
    def detect_choch_bos(self, df):
        """Detect Change of Character and Break of Structure"""
        structure_breaks = []
        
        # Calculate swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df)-2):
            # Swing high
            if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i-2] and \
               df['High'].iloc[i] > df['High'].iloc[i+1] and df['High'].iloc[i] > df['High'].iloc[i+2]:
                swing_highs.append({'index': i, 'price': df['High'].iloc[i]})
            
            # Swing low
            if df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i-2] and \
               df['Low'].iloc[i] < df['Low'].iloc[i+1] and df['Low'].iloc[i] < df['Low'].iloc[i+2]:
                swing_lows.append({'index': i, 'price': df['Low'].iloc[i]})
        
        # Detect structure breaks
        for i in range(10, len(df)):
            # CHoCH Bullish - Break above previous lower high in downtrend
            recent_highs = [sh for sh in swing_highs if sh['index'] < i and sh['index'] > i-50]
            if len(recent_highs) >= 2:
                if df['Close'].iloc[i] > recent_highs[-1]['price'] and recent_highs[-1]['price'] < recent_highs[-2]['price']:
                    structure_breaks.append({
                        'type': 'bullish_choch',
                        'index': i,
                        'break_level': recent_highs[-1]['price']
                    })
            
            # BOS Bullish - Break above previous high in uptrend
            if len(recent_highs) >= 1:
                if df['Close'].iloc[i] > recent_highs[-1]['price']:
                    structure_breaks.append({
                        'type': 'bullish_bos',
                        'index': i,
                        'break_level': recent_highs[-1]['price']
                    })
        
        return structure_breaks

class LiquidityEngine:
    """ZANFLOW v12 Liquidity Analysis Engine"""
    
    def __init__(self):
        self.liquidity_pools = []
        self.engineered_liquidity = []
        
    def identify_liquidity_pools(self, df):
        """Identify major liquidity pools"""
        pools = []
        
        # Session highs/lows
        for i in range(0, len(df), 24):  # Daily sessions
            if i + 24 < len(df):
                session_data = df.iloc[i:i+24]
                pools.append({
                    'type': 'session_high',
                    'level': session_data['High'].max(),
                    'strength': 'high',
                    'session_start': df.index[i]
                })
                pools.append({
                    'type': 'session_low',
                    'level': session_data['Low'].min(),
                    'strength': 'high',
                    'session_start': df.index[i]
                })
        
        # Equal highs/lows
        self._detect_equal_levels(df, pools)
        
        return pools
    
    def _detect_equal_levels(self, df, pools):
        """Detect equal highs and lows (engineered liquidity)"""
        tolerance = 0.0002  # 2 pips tolerance
        
        for i in range(20, len(df)-20):
            # Check for equal highs
            current_high = df['High'].iloc[i]
            for j in range(max(0, i-50), i):
                if abs(df['High'].iloc[j] - current_high) / current_high < tolerance:
                    pools.append({
                        'type': 'equal_highs',
                        'level': current_high,
                        'indices': [j, i],
                        'strength': 'medium'
                    })
            
            # Check for equal lows
            current_low = df['Low'].iloc[i]
            for j in range(max(0, i-50), i):
                if abs(df['Low'].iloc[j] - current_low) / current_low < tolerance:
                    pools.append({
                        'type': 'equal_lows',
                        'level': current_low,
                        'indices': [j, i],
                        'strength': 'medium'
                    })
        
        return pools

class MicrostructureAnalyzer:
    """Tick and microstructure analysis"""
    
    def compute_spread_instability(self, df):
        """Compute spread instability metric"""
        spreads = df['High'] - df['Low']
        spread_volatility = spreads.rolling(20).std()
        spread_mean = spreads.rolling(20).mean()
        
        instability = spread_volatility / spread_mean
        instability = instability.fillna(0)
        
        return instability
    
    def detect_absorption_patterns(self, df):
        """Detect volume absorption patterns"""
        patterns = []
        
        for i in range(20, len(df)-1):
            # High volume with small spread = absorption
            volume_ratio = df['Volume'].iloc[i] / df['Volume'].rolling(20).mean().iloc[i]
            spread = df['High'].iloc[i] - df['Low'].iloc[i]
            avg_spread = (df['High'] - df['Low']).rolling(20).mean().iloc[i]
            
            if volume_ratio > 1.5 and spread < avg_spread * 0.7:
                patterns.append({
                    'index': i,
                    'type': 'absorption',
                    'volume_ratio': volume_ratio,
                    'spread_compression': spread / avg_spread
                })
        
        return patterns

class EconomicDataManager:
    """Manage economic data integration"""
    
    def __init__(self):
        self.economic_data = {}
        
    def fetch_macro_indicators(self):
        """Fetch macro indicators (simulated for demo)"""
        # In production, use actual API calls
        return {
            'DXY': {'value': 103.5, 'change': 0.25},
            'US10Y': {'value': 4.25, 'change': 0.05},
            'VIX': {'value': 15.3, 'change': -0.8},
            'GOLD': {'value': 2050, 'change': 10},
            'OIL': {'value': 78.5, 'change': -1.2}
        }
    
    def calculate_risk_metrics(self, macro_data):
        """Calculate risk-on/risk-off metrics"""
        risk_score = 0
        
        # VIX inverse correlation
    if macro_data['VIX']['value']:

        st.markdown("### 📊 Market Regime Analysis")
            
    if 'risk_metrics' in results:
                risk = results['risk_metrics']
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk['risk_score'],
                    title={'text': "Risk Sentiment"},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [-2, 2]},
                        'bar': {'color': "green" if risk['risk_score'] > 0 else "red"},
                        'steps': [
                            {'range': [-2, -1], 'color': "lightgray"},
                            {'range': [-1, 1], 'color': "gray"},
                            {'range': [1, 2], 'color': "lightgray"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 0
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Regime details
                st.info(f"**Market Regime:** {risk['market_regime'].upper()}")
                st.info(f"**Confidence:** {risk['confidence']:.1%}")

class TradeSignalGenerator:
    """Generate trade signals based on confluence"""
    
    def __init__(self):
        self.signals = []
        
    def generate_signals(self, df, results):
        """Generate trade signals from analysis results"""
        signals = []
        
        # Wyckoff Spring signals
        if 'springs' in results:
            for spring in results['springs']:
                signals.append({
                    'type': 'LONG',
                    'reason': 'Wyckoff Spring',
                    'index': spring['index'],
                    'entry': df['Close'].iloc[spring['index']],
                    'sl': spring['low'] - df['ATR'].iloc[spring['index']],
                    'tp': df['Close'].iloc[spring['index']] + 3 * df['ATR'].iloc[spring['index']],
                    'confidence': spring['snapback_strength']
                })
        
        # SMC Sweep signals
        if 'sweeps' in results:
            for sweep in results['sweeps']:
                if sweep['type'] == 'bullish_sweep' and sweep['rejection_strength'] > 0.8:
                    signals.append({
                        'type': 'LONG',
                        'reason': 'Bullish Liquidity Sweep',
                        'index': sweep['index'],
                        'entry': df['Close'].iloc[sweep['index']],
                        'sl': sweep['sweep_low'] - df['ATR'].iloc[sweep['index']] * 0.5,
                        'tp': df['Close'].iloc[sweep['index']] + 2 * df['ATR'].iloc[sweep['index']],
                        'confidence': sweep['rejection_strength']
                    })
        
        # Order Block signals
        if 'order_blocks' in results:
            for ob in results['order_blocks'][-5:]:  # Last 5 OBs
                if ob['type'] == 'bullish_ob':
                    # Check if price is at OB
                    current_price = df['Close'].iloc[-1]
                    if ob['lower'] <= current_price <= ob['upper']:
                        signals.append({
                            'type': 'LONG',
                            'reason': 'Order Block Reaction',
                            'index': len(df) - 1,
                            'entry': current_price,
                            'sl': ob['lower'] - df['ATR'].iloc[-1] * 0.5,
                            'tp': current_price + 2.5 * df['ATR'].iloc[-1],
                            'confidence': 0.7
                        })
        
        return signals

class AdvancedRiskManager:
    """Advanced risk management with dynamic position sizing"""
    
    def __init__(self, account_balance=10000, risk_percent=1.0):
        self.account_balance = account_balance
        self.risk_percent = risk_percent
        
    def calculate_position_size(self, entry, stop_loss, confidence=1.0):
        """Calculate position size based on risk"""
        risk_amount = self.account_balance * (self.risk_percent / 100) * confidence
        stop_distance = abs(entry - stop_loss)
        
        if stop_distance == 0:
            return 0
            
        position_size = risk_amount / stop_distance
        
        # Apply max position size limit (10% of account)
        max_position = self.account_balance * 0.1 / entry
        position_size = min(position_size, max_position)
        
        return round(position_size, 2)
    
    def calculate_risk_reward(self, entry, stop_loss, take_profit):
        """Calculate risk:reward ratio"""
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return 0
            
        return round(reward / risk, 2)

class BacktestEngine:
    """Simple backtesting engine for strategy validation"""
    
    def __init__(self):
        self.trades = []
        
    def backtest_signals(self, df, signals, risk_manager):
        """Backtest generated signals"""
        results = []
        
        for signal in signals:
            if signal['index'] >= len(df) - 1:
                continue
                
            # Simulate trade
            entry_idx = signal['index']
            entry_price = signal['entry']
            stop_loss = signal['sl']
            take_profit = signal['tp']
            
            # Find exit
            for i in range(entry_idx + 1, len(df)):
                # Check stop loss
                if df['Low'].iloc[i] <= stop_loss:
                    exit_price = stop_loss
                    exit_idx = i
                    pnl = exit_price - entry_price if signal['type'] == 'LONG' else entry_price - exit_price
                    result = 'LOSS'
                    break
                    
                # Check take profit
                if df['High'].iloc[i] >= take_profit:
                    exit_price = take_profit
                    exit_idx = i
                    pnl = exit_price - entry_price if signal['type'] == 'LONG' else entry_price - exit_price
                    result = 'WIN'
                    break
            else:
                # Trade still open
                exit_price = df['Close'].iloc[-1]
                exit_idx = len(df) - 1
                pnl = exit_price - entry_price if signal['type'] == 'LONG' else entry_price - exit_price
                result = 'OPEN'
            
            # Calculate position size and actual PnL
            position_size = risk_manager.calculate_position_size(entry_price, stop_loss, signal['confidence'])
            actual_pnl = pnl * position_size
            
            results.append({
                'entry_date': df.index[entry_idx],
                'exit_date': df.index[exit_idx],
                'type': signal['type'],
                'reason': signal['reason'],
                'entry': entry_price,
                'exit': exit_price,
                'sl': stop_loss,
                'tp': take_profit,
                'pnl': pnl,
                'pnl_pct': (pnl / entry_price) * 100,
                'actual_pnl': actual_pnl,
                'result': result,
                'rr': risk_manager.calculate_risk_reward(entry_price, stop_loss, take_profit)
            })
        
        return pd.DataFrame(results)

def add_trade_management_section(dashboard):
    """Add trade management section to dashboard"""
    st.markdown("---")
    st.header("💼 Trade Management & Signals")
    
    # Initialize components
    signal_gen = TradeSignalGenerator()
    risk_mgr = AdvancedRiskManager(
        account_balance=st.session_state.get('account_balance', 10000),
        risk_percent=st.session_state.get('risk_percent', 1.0)
    )
    backtest = BacktestEngine()
    
    # Generate signals
    signals = signal_gen.generate_signals(
        st.session_state.data, 
        st.session_state.analysis_results
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Active Trade Signals")
        
        if signals:
            # Display signals in a table
            signal_df = pd.DataFrame(signals)
            signal_df['R:R'] = signal_df.apply(
                lambda x: risk_mgr.calculate_risk_reward(x['entry'], x['sl'], x['tp']), 
                axis=1
            )
            signal_df['Position Size'] = signal_df.apply(
                lambda x: risk_mgr.calculate_position_size(x['entry'], x['sl'], x['confidence']),
                axis=1
            )
            
            # Format for display
            display_df = signal_df[['type', 'reason', 'entry', 'sl', 'tp', 'R:R', 'Position Size', 'confidence']]
            display_df.columns = ['Direction', 'Reason', 'Entry', 'Stop Loss', 'Take Profit', 'R:R', 'Size', 'Confidence']
            
            st.dataframe(
                display_df.style.format({
                    'Entry': '{:.5f}',
                    'Stop Loss': '{:.5f}',
                    'Take Profit': '{:.5f}',
                    'R:R': '{:.1f}',
                    'Size': '{:.2f}',
                    'Confidence': '{:.1%}'
                }).background_gradient(subset=['Confidence'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Backtest results
            if st.button("🔄 Run Backtest"):
                backtest_results = backtest.backtest_signals(
                    st.session_state.data,
                    signals,
                    risk_mgr
                )
                
                if not backtest_results.empty:
                    st.subheader("📈 Backtest Results")
                    
                    # Summary metrics
                    total_trades = len(backtest_results)
                    winning_trades = len(backtest_results[backtest_results['result'] == 'WIN'])
                    losing_trades = len(backtest_results[backtest_results['result'] == 'LOSS'])
                    
                    if total_trades > 0:
                        win_rate = winning_trades / total_trades
                        avg_win = backtest_results[backtest_results['result'] == 'WIN']['pnl_pct'].mean() if winning_trades > 0 else 0
                        avg_loss = backtest_results[backtest_results['result'] == 'LOSS']['pnl_pct'].mean() if losing_trades > 0 else 0
                        total_pnl = backtest_results['actual_pnl'].sum()
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        metric_col1.metric("Win Rate", f"{win_rate:.1%}")
                        metric_col2.metric("Avg Win", f"{avg_win:.2f}%")
                        metric_col3.metric("Avg Loss", f"{avg_loss:.2f}%")
                        metric_col4.metric("Total P&L", f"${total_pnl:,.2f}")
                        
                        # Trade details
                        st.dataframe(
                            backtest_results[['entry_date', 'type', 'reason', 'entry', 'exit', 'pnl_pct', 'actual_pnl', 'result']].style.format({
                                'entry': '{:.5f}',
                                'exit': '{:.5f}',
                                'pnl_pct': '{:.2f}%',
                                'actual_pnl': '${:,.2f}'
                            }).background_gradient(subset=['pnl_pct'], cmap='RdYlGn'),
                            use_container_width=True
                        )
        else:
            st.info("No trade signals generated. Adjust analysis parameters or wait for market conditions to align.")
    
    with col2:
        st.subheader("⚙️ Risk Settings")
        
        account_balance = st.number_input(
            "Account Balance ($)",
            min_value=1000,
            max_value=1000000,
            value=st.session_state.get('account_balance', 10000),
            step=1000,
            key='account_balance_input'
        )
        
        if account_balance != st.session_state.get('account_balance', 10000):
            st.session_state.account_balance = account_balance
        
        st.markdown("### 📊 Position Calculator")
        
        calc_entry = st.number_input("Entry Price", value=st.session_state.data['Close'].iloc[-1], format="%.5f")
        calc_sl = st.number_input("Stop Loss", value=calc_entry * 0.99, format="%.5f")
        calc_tp = str.number_input("Take Profit", value=calc_entry * 1.02, format="%.5f")
        
        if st.button("Calculate Position"):
            pos_size = risk_mgr.calculate_position_size(calc_entry, calc_sl)
            rr_ratio = risk_mgr.calculate_risk_reward(calc_entry, calc_sl, calc_tp)
            
            st.success(f"**Position Size:** {pos_size:.2f} units")
            st.info(f"**Risk:Reward:** {rr_ratio:.1f}:1")
            st.info(f"**Risk Amount:** ${abs(calc_entry - calc_sl) * pos_size:.2f}")
            st.info(f"**Potential Profit:** ${abs(calc_tp - calc_entry) * pos_size:.2f}")

def main():
    """Main execution function"""
    # Initialize dashboard
    dashboard = ZANFLOWDashboard()
    
    # Run dashboard
    dashboard.run()
    
    # Add trade management section if data is loaded
    if st.session_state.get('data') is not None:
        add_trade_management_section(dashboard)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>ZANFLOW v13 ULTIMATE MEGA DASHBOARD | Institutional-Grade Trading System</p>
            <p>Integrating Wyckoff, SMC, Liquidity Analysis, and Advanced Risk Management</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()