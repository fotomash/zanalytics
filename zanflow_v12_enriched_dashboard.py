import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import warnings
import os
import glob

warnings.filterwarnings('ignore')

class EnrichedZanflowDashboard:
    def __init__(self):
        """Initialize the enriched dashboard with dynamic data scanning."""
        self.all_data = {}
        self.scan_results = {}

    def load_all_data(self):
        """
        Scan all subdirectories for data files and load them into a structured dictionary.
        """
        st.write("Scanning for data files...")
        json_files = glob.glob('**/*.json', recursive=True)
        csv_files = glob.glob('**/*_COMPREHENSIVE_*.csv', recursive=True)
        
        self.scan_results = {'json': json_files, 'csv': csv_files}
        
        # Load CSV data
        for f in csv_files:
            try:
                path = Path(f)
                pair = path.parent.name
                timeframe = path.stem.split('_COMPREHENSIVE_')[-1]
                
                if pair not in self.all_data:
                    self.all_data[pair] = {'csv': {}, 'json': {}}
                
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                self.all_data[pair]['csv'][timeframe] = df
            except Exception as e:
                continue

        # Load JSON analysis reports
        for f in json_files:
            try:
                path = Path(f)
                # Heuristic to find pair name from path
                pair_match = re.search(r'(XAUUSD|EURUSD|GBPUSD)', str(path), re.IGNORECASE)
                if not pair_match:
                    # Fallback to parent directory if no pair in name
                    pair = path.parent.name
                    if len(pair) > 6 or len(pair) < 3: # Simple check if it's a valid pair name
                        continue
                else:
                    pair = pair_match.group(0).upper()

                if pair not in self.all_data:
                    self.all_data[pair] = {'csv': {}, 'json': {}}
                
                with open(f, 'r') as file:
                    report_data = json.load(f)
                
                # Store all JSON reports, keyed by filename for uniqueness
                self.all_data[pair]['json'][path.stem] = report_data
            except Exception as e:
                continue

    def get_strategic_summary(self, pair):
        """
        Generate a high-level strategic summary based on the latest analysis,
        incorporating ISPTS and Wyckoff principles.
        """
        # Find the most relevant JSON analysis for the summary
        latest_analysis = None
        if pair in self.all_data and self.all_data[pair]['json']:
            # Prioritize microstructure reports
            for name, data in self.all_data[pair]['json'].items():
                if 'microstructure' in name.lower():
                    latest_analysis = data
                    break
            if not latest_analysis:
                # Fallback to any other report
                latest_analysis = next(iter(self.all_data[pair]['json'].values()))

        if not latest_analysis:
            return "🟡 **Neutral**: No detailed analysis report found. Awaiting data.", "NEUTRAL"

        # Extract key metrics using .get() for safety
        manipulation = latest_analysis.get('manipulation_detection', {}).get('manipulation_score', 0)
        smc_bias = latest_analysis.get('smc_analysis', {}).get('bias', 'NEUTRAL').upper()
        wyckoff_phase = latest_analysis.get('wyckoff_analysis', {}).get('dominant_phase', 'Unknown')
        inducement_rate = latest_analysis.get('inducement_analysis', {}).get('inducement_rate', 0)

        summary = []
        
        # Interpret the data through the ISPTS/Wyckoff lens
        if manipulation > 40:
            summary.append(f"**High Institutional Activity** ({manipulation:.1f}% manip. score).")
        
        if smc_bias == 'BULLISH' and wyckoff_phase == 'Accumulation':
            summary.append("Bias is **Bullish** within an **Accumulation** structure.")
            if inducement_rate > 5:
                summary.append("High inducement suggests a **liquidity sweep** may precede a markup.")
            else:
                summary.append("Look for entries in **discount POIs** after confirmation.")
        
        elif smc_bias == 'BEARISH' and wyckoff_phase == 'Distribution':
            summary.append("Bias is **Bearish** within a **Distribution** structure.")
            if inducement_rate > 5:
                summary.append("High inducement suggests a **liquidity sweep** may precede a markdown.")
            else:
                summary.append("Look for entries in **premium POIs** after confirmation.")
        
        else:
            summary.append(f"Market is in a **{wyckoff_phase}** phase with a **{smc_bias}** SMC bias.")

        if not summary:
            return "🟡 **Observational**: Market conditions are mixed. Awaiting clearer signals.", "NEUTRAL"
            
        final_summary = " ".join(summary)
        overall_bias = f"{'🟢' if smc_bias == 'BULLISH' else '🔴' if smc_bias == 'BEARISH' else '🟡'} {smc_bias}"
        
        return final_summary, overall_bias

    def create_main_dashboard(self):
        st.set_page_config(
            page_title="ZANFLOW v12 Enriched", 
            page_icon="🧠", 
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Refined dark theme CSS
        st.markdown("""
        <style>
            .main { background-color: #0E1117; color: #FAFAFA; }
            .stApp { background: linear-gradient(135deg, #0E1117 0%, #1E2332 100%); }
            .main-header {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            }
            .metric-card {
                background: linear-gradient(135deg, #262730 0%, #2F3349 100%);
                padding: 1rem; border-radius: 10px; border: 1px solid #404552;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            }
            .stMetric { color: #FAFAFA; }
            .stMetric > div:nth-child(2) { font-size: 2em; }
            .strategic-summary {
                background: rgba(47, 51, 73, 0.7); padding: 1rem; border-radius: 10px;
                border-left: 5px solid #667eea; margin-bottom: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="main-header">
            <h1>🚀 ZANFLOW v12 Ultimate Dashboard (Enriched)</h1>
            <p>Dynamic Data Scanning • Real-Time Analysis • ISPTS & Wyckoff Insights</p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🧠 Initializing ZANFLOW Engine... Scanning for all data..."):
            self.load_all_data()

        if not self.all_data:
            st.error("❌ No data found. Ensure your data files (CSV/JSON) are in this directory or subdirectories.")
            st.info(f"Scanned for files, found: {len(self.scan_results['csv'])} CSVs, {len(self.scan_results['json'])} JSONs.")
            return

        self.create_sidebar_controls()

        if st.session_state.get('selected_pair') and st.session_state.get('selected_timeframe'):
            self.display_enriched_analysis()
        else:
            st.warning("Please select a currency pair and timeframe from the sidebar to begin analysis.")

    def create_sidebar_controls(self):
        st.sidebar.title("🎛️ Analysis Control Center")
        available_pairs = list(self.all_data.keys())
        if not available_pairs:
            st.sidebar.error("No pairs loaded.")
            return

        selected_pair = st.sidebar.selectbox("📈 Select Currency Pair", available_pairs, key="selected_pair")
        
        if selected_pair and self.all_data[selected_pair]['csv']:
            available_timeframes = sorted(list(self.all_data[selected_pair]['csv'].keys()))
            st.sidebar.selectbox("⏱️ Select Timeframe", available_timeframes, key="selected_timeframe")
        else:
            st.sidebar.warning(f"No CSV data found for {selected_pair}")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Chart Settings")
        st.session_state['lookback_bars'] = st.sidebar.slider("Lookback Period (Bars)", 100, 2000, 500)
        st.session_state['chart_theme'] = 'plotly_dark' # Hardcode to dark theme

    def display_enriched_analysis(self):
        pair = st.session_state['selected_pair']
        timeframe = st.session_state['selected_timeframe']

        if pair not in self.all_data or timeframe not in self.all_data[pair]['csv']:
            st.error("Selected data not available.")
            return

        df = self.all_data[pair]['csv'][timeframe]
        lookback = st.session_state.get('lookback_bars', 500)
        df_display = df.tail(lookback).copy()

        st.markdown(f"## 🚀 {pair} {timeframe} - Enriched Analysis")

        # Strategic Summary Panel
        summary_text, overall_bias = self.get_strategic_summary(pair)
        st.markdown("### 🧠 Strategic Summary")
        st.markdown(f"<div class='strategic-summary'>{summary_text}</div>", unsafe_allow_html=True)

        # Market Status Row
        self.display_market_status(df_display, overall_bias)

        # Main Chart
        self.create_enriched_price_chart(df_display, pair, timeframe)
        
        # Data Explorer
        with st.expander("🔍 Data Explorer"):
            st.dataframe(df_display.tail(100).iloc[:, :20]) # Show sample data
            st.json(self.all_data[pair]['json'])


    def display_market_status(self, df, overall_bias):
        cols = st.columns(5)
        with cols[0]:
            st.metric("Current Price", f"{df['close'].iloc[-1]:.4f}")
        with cols[1]:
            st.metric("SMC/Wyckoff Bias", overall_bias)
        with cols[2]:
            atr = df.get('atr_14', pd.Series([0])).iloc[-1]
            st.metric("ATR (14)", f"{atr:.4f}")
        with cols[3]:
            rsi = df.get('rsi_14', pd.Series([50])).iloc[-1]
            st.metric("RSI (14)", f"{rsi:.1f}")
        with cols[4]:
            vol_regime = "🔥 HIGH" if df['close'].pct_change().std() > df['close'].pct_change().quantile(0.8) else "❄️ LOW"
            st.metric("Volatility", vol_regime)

    def create_enriched_price_chart(self, df, pair, timeframe):
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f"{pair} {timeframe} - Price Action & Key Levels",
                "Volume & Momentum",
                "Oscillators (RSI/MACD)"
            ],
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2],
            shared_xaxes=True
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name="Price", increasing_line_color='#27AE60', decreasing_line_color='#E74C3C'
        ), row=1, col=1)

        # Moving Averages
        ma_colors = {'ema_8': '#3498DB', 'ema_21': '#F1C40F', 'sma_200': '#9B59B6'}
        for ma, color in ma_colors.items():
            if ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ma], mode='lines', name=ma.upper(),
                    line=dict(color=color, width=1.5)
                ), row=1, col=1)

        # Add SMC/Wyckoff overlays from the data
        self.add_smart_overlays(fig, df, row=1)

        # Volume
        vol_colors = ['#27AE60' if c >= o else '#E74C3C' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df.index, y=df['volume'], name='Volume', marker_color=vol_colors, opacity=0.5
        ), row=2, col=1)

        # RSI
        if 'rsi_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi_14'], mode='lines', name='RSI', line=dict(color='#F39C12')
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="white", opacity=0.3, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="white", opacity=0.3, row=3, col=1)

        # MACD
        if 'MACD_12_26_9' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_12_26_9'], mode='lines', name='MACD', line=dict(color='#8E44AD')
            ), row=3, col=1)
        
        fig.update_layout(
            template='plotly_dark', height=800, showlegend=True,
            xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, zeroline=False)

        st.plotly_chart(fig, use_container_width=True)

    def add_smart_overlays(self, fig, df, row):
        """Add overlays for SMC and Wyckoff events found in the data."""
        # Fair Value Gaps
        if 'bullish_fvg' in df.columns and df['bullish_fvg'].sum() > 0:
            bull_fvgs = df[df['bullish_fvg']]
            fig.add_trace(go.Scatter(
                x=bull_fvgs.index, y=bull_fvgs['low'] * 0.998, mode='markers', name='Bullish FVG',
                marker=dict(symbol='triangle-up', color='#00D2FF', size=10, line=dict(width=1, color='white'))
            ), row=row, col=1)

        if 'bearish_fvg' in df.columns and df['bearish_fvg'].sum() > 0:
            bear_fvgs = df[df['bearish_fvg']]
            fig.add_trace(go.Scatter(
                x=bear_fvgs.index, y=bear_fvgs['high'] * 1.002, mode='markers', name='Bearish FVG',
                marker=dict(symbol='triangle-down', color='#FF416C', size=10, line=dict(width=1, color='white'))
            ), row=row, col=1)

        # Structure Breaks
        if 'structure_break' in df.columns and df['structure_break'].sum() > 0:
            breaks = df[df['structure_break']]
            fig.add_trace(go.Scatter(
                x=breaks.index, y=breaks['high'] * 1.005, mode='markers', name='Structure Break',
                marker=dict(symbol='x', color='#F1C40F', size=8)
            ), row=row, col=1)

def main():
    dashboard = EnrichedZanflowDashboard()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()