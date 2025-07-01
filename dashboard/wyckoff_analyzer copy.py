import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ────────────────────────────────────────────────────────────────────────────────
# Secrets‑based paths
# ────────────────────────────────────────────────────────────────────────────────
DATA_DIR = st.secrets.get("data_directory", "./data")
RAW_DATA_DIR = st.secrets.get("raw_data_directory", DATA_DIR)
BAR_DATA_DIR  = st.secrets.get("bar_data_directory", RAW_DATA_DIR)

# ────────────────────────────────────────────────────────────────────────────────
# Extend PYTHONPATH to include core engines
# ────────────────────────────────────────────────────────────────────────────────
import sys
CORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "zanalytics", "core"))
if CORE_PATH not in sys.path:
    sys.path.append(CORE_PATH)

# ────────────────────────────────────────────────────────────────────────────────
# External Engines (graceful fall‑backs)
# ────────────────────────────────────────────────────────────────────────────────
try:
    from micro_wyckoff_phase_engine import MicroWyckoffPhaseEngine
    from wyckoff_phase_engine import WyckoffPhaseEngine
    from wyckoff_phase_tracker import WyckoffPhaseTracker
    from fibonacci_filter import FibonacciFilter
    from vwap_engine import VWAPEngine
    from advanced_stoploss_lots_engine import AdvancedSLLotsEngine
    from vsa_signals_mentfx import VSASignals
    from poi_manager_smc import POIManager
    from poi_hit_watcher_smc import POIHitWatcher
    from smc_enrichment_engine import SMCEnrichmentEngine
except Exception:
    class _Stub:
        """Fallback stub to satisfy missing dependency calls"""
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return None
        def __getattr__(self, _):
            return lambda *a, **kw: None
        def __getattr__(self, _):
            return lambda *a, **kw: None
    MicroWyckoffPhaseEngine = WyckoffPhaseEngine = WyckoffPhaseTracker = _Stub  # type: ignore
    FibonacciFilter = VWAPEngine = AdvancedSLLotsEngine = _Stub                # type: ignore
    VSASignals = POIManager = POIHitWatcher = SMCEnrichmentEngine = _Stub      # type: ignore

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit App Config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Advanced Wyckoff Dashboard ✨")

st.title("🏛️ Advanced Wyckoff Intelligence Dashboard — v2.1")
st.caption("Institutional‑Grade Trade Research powered by Multi‑Engine Confluence")

# ────────────────────────────────────────────────────────────────────────────────
# Data Loader helpers
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(path: str):
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df.set_index("timestamp", inplace=True)
    return df

# Collect available files from configured directories
csv_files = []
for folder in {DATA_DIR, RAW_DATA_DIR, BAR_DATA_DIR}:
    if os.path.isdir(folder):
        csv_files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")])

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗂️ Data Source")

    if csv_files:
        file_choice = st.selectbox("Select CSV File", options=csv_files, format_func=lambda p: os.path.basename(p))
    else:
        file_choice = None
        st.warning("No CSV files found in configured data directories.")

    st.markdown("---")
    st.header("⚙️ Modules")
    enable_vwap   = st.checkbox("VWAP Engine", value=True)
    enable_fib    = st.checkbox("Fibonacci Filter", value=True)
    enable_vsa    = st.checkbox("VSA Signals", value=True)
    enable_poi    = st.checkbox("POI Manager", value=True)
    enable_stop   = st.checkbox("Advanced SL/Lot Engine", value=True)

# ────────────────────────────────────────────────────────────────────────────────
# Main Logic
# ────────────────────────────────────────────────────────────────────────────────
if file_choice is None:
    st.info("Please select a CSV file from the sidebar.")
    st.stop()

df_raw = load_csv(file_choice)

# Normalize column names to lower‑case for consistency
df_raw.columns = [c.lower() for c in df_raw.columns]

# Primary Wyckoff Engines
wyckoff_engine = WyckoffPhaseEngine()
phase_tracker  = WyckoffPhaseTracker()
micro_engine   = MicroWyckoffPhaseEngine()

# Pre‑process & annotate
phase_info     = wyckoff_engine.detect_phases(df_raw)
phase_events   = wyckoff_engine.detect_events(df_raw)
micro_events   = micro_engine.detect_micro_events(df_raw)

# Optional Overlays
fib_filter = FibonacciFilter(df_raw)       if enable_fib  else None
vwap_eng    = VWAPEngine(df_raw)           if enable_vwap else None
vsa_eng     = VSASignals(df_raw)           if enable_vsa  else None
poi_mgr     = POIManager(df_raw)           if enable_poi  else None
sl_lot_eng  = AdvancedSLLotsEngine(df_raw) if enable_stop else None
smc_enricher= SMCEnrichmentEngine(df_raw)

smc_meta = smc_enricher.enrich()

# ────────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────────
tab_price, tab_liq, tab_signals, tab_risk = st.tabs([
    "📈 Price / Wyckoff", "💧 Liquidity", "🎯 Signals", "🛡️ Risk & VWAP/Fib"])

# Price Tab
with tab_price:
    st.subheader("Wyckoff Phase & Event Map")

    # --- Column mapping fallback -------------------------------------------
    def map_col(names):
        for n in names:
            if n in df_raw.columns:
                return n
        return None

    open_col  = map_col(["open", "o", "op", "price_open"])
    high_col  = map_col(["high", "h", "price_high"])
    low_col   = map_col(["low", "l", "price_low"])
    close_col = map_col(["close", "c", "price_close", "last"])
    vol_col   = map_col(["volume", "vol", "v"])

    if None in {open_col, high_col, low_col, close_col}:
        # Attempt to synthesise OHLC from a single price column (mid/bid/ask/last)
        mid_col = map_col(["price_mid", "last", "price", "bid", "ask"])
        if mid_col is None:
            st.error("CSV lacks standard OHLC or a recognisable price column (mid/last/bid/ask). Please verify your data.")
            st.stop()
        # Fabricate synthetic bars
        df_raw["open_syn"]  = df_raw[mid_col].shift().fillna(df_raw[mid_col])
        df_raw["close_syn"] = df_raw[mid_col]
        df_raw["high_syn"]  = pd.concat([df_raw["open_syn"], df_raw["close_syn"]], axis=1).max(axis=1)
        df_raw["low_syn"]   = pd.concat([df_raw["open_syn"], df_raw["close_syn"]], axis=1).min(axis=1)
        open_col, high_col, low_col, close_col = "open_syn", "high_syn", "low_syn", "close_syn"
        st.warning("OHLC columns missing – synthetic candlesticks generated from '{mid}' column.".format(mid=mid_col))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(
        go.Candlestick(
            x=df_raw.index,
            open=df_raw[open_col],
            high=df_raw[high_col],
            low=df_raw[low_col],
            close=df_raw[close_col],
            name="Price"
        ),
        row=1,
        col=1
    )
    segments = phase_tracker.segment_phases(df_raw, phase_info)
    if segments:
        for seg in segments:
            fig.add_vrect(
                x0=seg['start'], x1=seg['end'],
                fillcolor=seg['color'], opacity=0.07,
                layer="below", line_width=0,
                annotation_text=seg['label'], annotation_position="top left"
            )
    for ev in (*phase_events, *micro_events):
        fig.add_annotation(x=ev['timestamp'], y=ev['price'], text=ev['type'], showarrow=True, arrowhead=2, bgcolor="yellow" if 'micro' in ev['type'].lower() else "lightblue")
    # Volume plot (fallback to zeros if volume column missing)
    vol_series = df_raw[vol_col] if vol_col else pd.Series(np.zeros(len(df_raw)), index=df_raw.index)
    fig.add_trace(go.Bar(x=df_raw.index, y=vol_series, name="Volume", marker_color='rgba(100,150,255,0.4)'), row=2, col=1)
    fig.update_layout(height=700, showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# Liquidity Tab
with tab_liq:
    st.subheader("Liquidity & POI Map")
    if poi_mgr:
        st.dataframe(poi_mgr.extract_pois().head())
    else:
        st.info("POI Manager disabled.")

# Signals Tab
with tab_signals:
    st.subheader("Predictive Signals & Confluence")
    signals = micro_engine.generate_signals(df_raw, phase_events, micro_events, smc_meta)
    st.json(signals)
    if vsa_eng and (vsa := vsa_eng.detect_signals()):
        st.markdown("### VSA Signals")
        st.dataframe(vsa.head())

# Risk / VWAP / Fib Tab
with tab_risk:
    st.subheader("VWAP, Fibonacci, and Dynamic Risk")
    col_vwap, col_fib = st.columns(2)
    with col_vwap:
        if vwap_eng:
            vwap = vwap_eng.compute()
            st.line_chart(vwap, use_container_width=True)
        else:
            st.info("VWAP Engine disabled.")
    with col_fib:
        if fib_filter:
            fib_levels = fib_filter.levels()
            fib_fig = go.Figure()
            for lvl in fib_levels:
                fib_fig.add_hline(y=lvl['price'], line_dash="dash", annotation_text=f"{lvl['level']}%")
            st.plotly_chart(fib_fig, use_container_width=True)
        else:
            st.info("Fibonacci Filter disabled.")
    st.markdown("---")
    if sl_lot_eng:
        st.subheader("Advanced Stop‑Loss & Position Sizing")
        risk_table = sl_lot_eng.calculate(df_raw)
        st.dataframe(risk_table.head())
    else:
        st.info("SL/Lot Engine disabled.")
