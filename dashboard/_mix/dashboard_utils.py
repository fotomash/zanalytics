# Dashboard Utilities
# dashboard_utils.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import plotly.graph_objects as go
from datetime import datetime, timedelta

class PerformanceTracker:
    """Track and display trading performance metrics"""

    @staticmethod
    def calculate_performance_metrics(signals: List[Dict[str, Any]], 
                                    market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics from signals"""
        metrics = {
            'total_signals': len(signals),
            'win_rate': 0.0,
            'avg_risk_reward': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0
        }

        if not signals:
            return metrics

        # Calculate average risk-reward
        risk_rewards = [s.get('risk_reward_ratio', 0) for s in signals]
        metrics['avg_risk_reward'] = np.mean(risk_rewards) if risk_rewards else 0

        # Estimate win rate based on confidence
        confidences = [s.get('confidence', 0.5) for s in signals]
        metrics['estimated_win_rate'] = np.mean(confidences) if confidences else 0.5

        return metrics

    @staticmethod
    def create_performance_chart(metrics: Dict[str, Any]) -> go.Figure:
        """Create performance metrics visualization"""
        fig = go.Figure()

        # Create gauge charts for key metrics
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['estimated_win_rate'] * 100,
            title={'text': "Win Rate %"},
            domain={'x': [0, 0.5], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ]
            }
        ))

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['avg_risk_reward'],
            title={'text': "Avg Risk/Reward"},
            domain={'x': [0.5, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 1], 'color': "lightcoral"},
                    {'range': [1, 2], 'color': "lightyellow"},
                    {'range': [2, 5], 'color': "lightgreen"}
                ]
            }
        ))

        fig.update_layout(height=250)
        return fig

class MarketScanner:
    """Scan multiple markets for opportunities"""

    @staticmethod
    def scan_markets(symbols: List[str], data_loader) -> pd.DataFrame:
        """Scan multiple symbols and return summary"""
        scan_results = []

        for symbol in symbols:
            try:
                # Get latest analysis
                analysis = data_loader.load_integrated_analysis(symbol, "1h")
                if not analysis:
                    continue

                consensus = analysis.get('consensus', {})

                scan_results.append({
                    'Symbol': symbol,
                    'Sentiment': consensus.get('overall_sentiment', 'neutral'),
                    'Confidence': consensus.get('confidence', 0),
                    'Signals': len(consensus.get('signals', [])),
                    'Last Update': analysis.get('timestamp', '')
                })
            except:
                continue

        return pd.DataFrame(scan_results)

class AlertManager:
    """Manage and display trading alerts"""

    @staticmethod
    def check_alerts(signals: List[Dict[str, Any]], 
                    market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        current_price = market_data['close'].iloc[-1] if not market_data.empty else 0

        for signal in signals:
            # Check if signal is high priority
            if signal.get('priority', 0) >= 3:
                alerts.append({
                    'type': 'high_priority_signal',
                    'message': f"High priority {signal.get('signal_type', '')} signal detected",
                    'severity': 'high',
                    'timestamp': datetime.now()
                })

            # Check if price near entry
            entry_price = signal.get('entry_price', 0)
            if entry_price and abs(current_price - entry_price) / entry_price < 0.002:
                alerts.append({
                    'type': 'price_near_entry',
                    'message': f"Price near signal entry: ${entry_price:.4f}",
                    'severity': 'medium',
                    'timestamp': datetime.now()
                })

        return alerts

# Portfolio Analysis Component
def display_portfolio_analysis(st_container, portfolio_data: Dict[str, Any]):
    """Display portfolio analysis in streamlit container"""
    with st_container:
        st.subheader("💼 Portfolio Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Value", f"${portfolio_data.get('total_value', 0):,.2f}")

        with col2:
            pnl = portfolio_data.get('unrealized_pnl', 0)
            st.metric("Unrealized P&L", f"${pnl:,.2f}", 
                     f"{pnl/portfolio_data.get('total_value', 1)*100:.2f}%")

        with col3:
            st.metric("Open Positions", portfolio_data.get('open_positions', 0))

        with col4:
            st.metric("Available Balance", f"${portfolio_data.get('available_balance', 0):,.2f}")

# Multi-timeframe Analysis Component
def create_multi_timeframe_view(symbol: str, timeframes: List[str], data_loader) -> go.Figure:
    """Create multi-timeframe analysis view"""
    fig = make_subplots(
        rows=len(timeframes), 
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{tf} Chart" for tf in timeframes]
    )

    for i, tf in enumerate(timeframes):
        df = data_loader.load_market_data(symbol, tf)
        if df is not None and not df.empty:
            # Add candlestick for each timeframe
            fig.add_trace(
                go.Candlestick(
                    x=df.index[-100:],  # Last 100 candles
                    open=df['open'].iloc[-100:],
                    high=df['high'].iloc[-100:],
                    low=df['low'].iloc[-100:],
                    close=df['close'].iloc[-100:],
                    name=f"{tf}",
                    showlegend=False
                ),
                row=i+1, col=1
            )

    fig.update_layout(height=300 * len(timeframes))
    return fig
