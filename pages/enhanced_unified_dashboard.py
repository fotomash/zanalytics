
import sys
from pathlib import Path

# Ensure project root and ./core are on sys.path for dynamic module loading
project_root = Path(__file__).parent.resolve()
core_dir = project_root / "core"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

"""
ZANFLOW Enhanced Unified Microstructure Dashboard
Integrating ALL discovered strategies and modules
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import asyncio
from datetime import datetime, timedelta
import importlib
import traceback

# --- Custom Styling: Dark theme, gold highlights, background, compact font, background image ---
st.markdown(
    """
    <style>
    body, .stApp {
        background: linear-gradient(rgba(24,24,27,0.94), rgba(24,24,27,0.97)), url("theme/image_af247b.jpg") center center/cover no-repeat fixed !important;
        color: #e5e5e5 !important;
        font-size: 13px !important;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1n76uvr, .block-container {
        background: rgba(24,24,27,0.98) !important;
        color: #e5e5e5 !important;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1n76uvr {
        font-size: 13px !important;
    }
    .st-emotion-cache-6qob1r, .st-emotion-cache-1n76uvr, .sidebar-content, .css-1d391kg, .stSidebar {
        background: #18181b !important;
        color: #e5e5e5 !important;
        font-size: 12px !important;
        padding: 0.3rem 0.5rem !important;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1n76uvr, .stSidebar, .css-1d391kg {
        background: #18181b !important;
    }
    .st-emotion-cache-1v0mbdj h1, .st-emotion-cache-1v0mbdj h2, .st-emotion-cache-1v0mbdj h3, .st-emotion-cache-1v0mbdj h4,
    .st-emotion-cache-1n76uvr h1, .st-emotion-cache-1n76uvr h2, .st-emotion-cache-1n76uvr h3, .st-emotion-cache-1n76uvr h4 {
        color: #FFD700 !important;
        font-size: 1.1rem !important;
    }
    .st-emotion-cache-1v0mbdj .stButton>button, .st-emotion-cache-1n76uvr .stButton>button,
    .st-emotion-cache-1v0mbdj .stDownloadButton>button, .st-emotion-cache-1n76uvr .stDownloadButton>button {
        background-color: #FFD700 !important;
        color: #18181b !important;
        border-radius: 4px;
        font-size: 13px !important;
    }
    /* Gold highlights for metrics */
    .stMetric label, .st-emotion-cache-1v0mbdj .stMetric label, .st-emotion-cache-1n76uvr .stMetric label {
        color: #FFD700 !important;
    }
    /* Compact selectboxes */
    .stSelectbox, .st-emotion-cache-1n76uvr .stSelectbox {
        font-size: 13px !important;
        min-height: 28px !important;
    }
    /* Info box for summary stats */
    .zanflow-infobox {
        background: #23232a;
        border-left: 4px solid #FFD700;
        color: #FFD700;
        padding: 0.5rem 0.9rem !important;
        margin-top: 1.2rem;
        border-radius: 4px;
        font-size: 12px !important;
        display: inline-block;
    }
    /* Sidebar compactness */
    .st-emotion-cache-6qob1r, .sidebar-content, .stSidebar, .css-1d391kg {
        font-size: 12px !important;
        padding: 0.3rem 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Enhanced Module Registry
MODULE_REGISTRY = {
    "Core Analysis": {
        "ncOS_ultimate_microstructure_analyzer": {
            "path": "ncOS_ultimate_microstructure_analyzer",
            "function": "run_comprehensive_analysis",
            "description": "Ultimate microstructure analysis with spoofing detection"
        },
        "zanflow_microstructure_analyzer": {
            "path": "zanflow_microstructure_analyzer",
            "function": "analyze_microstructure",
            "description": "Core microstructure patterns and Wyckoff analysis"
        }
    },
    "SMC Advanced": {
        "liquidity_sweep_detector": {
            "path": "core.liquidity_sweep_detector",
            "function": "detect_liquidity_sweeps",
            "description": "Detects liquidity sweeps and stop hunts"
        },
        "poi_manager_smc": {
            "path": "core.poi_manager_smc",
            "function": "find_and_validate_smc_pois",
            "description": "Manages POIs (OB, FVG, BB, MB)"
        },
        "entry_executor_smc": {
            "path": "core.entry_executor_smc",
            "function": "execute_smc_entry",
            "description": "Automated SMC entry execution"
        },
        "confirmation_engine_smc": {
            "path": "core.confirmation_engine_smc",
            "function": "confirm_smc_entry",
            "description": "Validates SMC setups"
        },
        "wick_liquidity_monitor": {
            "path": "core.wick_liquidity_monitor",
            "function": "monitor_wick_liquidity",
            "description": "Monitors wick-based liquidity patterns"
        }
    },
    "Inducement & Sweeps": {
        "liquidity_engine_smc": {
            "path": "core.liquidity_engine_smc",
            "function": "detect_inducement_from_structure",
            "description": "ZSI Agent: Inducement-Sweep-POI Framework"
        },
        "advanced_smc_orchestrator": {
            "path": "core.advanced_smc_orchestrator",
            "function": "orchestrate_smc_analysis",
            "description": "Orchestrates all SMC components"
        }
    },
    "Wyckoff Analysis": {
        "wyckoff_phase_engine": {
            "path": "core.wyckoff_phase_engine",
            "function": "detect_wyckoff_phase",
            "description": "Wyckoff phase detection"
        },
        "micro_wyckoff_phase_engine": {
            "path": "core.micro_wyckoff_phase_engine",
            "function": "detect_micro_wyckoff",
            "description": "Micro-timeframe Wyckoff analysis"
        }
    },
    "Advanced Strategies": {
        "mentfx_ici_engine": {
            "path": "core.mentfx_ici_engine",
            "function": "tag_mentfx_ici",
            "description": "MENTFX Impulse-Correction-Impulse"
        },
        "vsa_signals_mentfx": {
            "path": "core.vsa_signals_mentfx",
            "function": "detect_vsa_signals",
            "description": "Volume Spread Analysis signals"
        },
        "divergence_engine": {
            "path": "core.divergence_engine",
            "function": "detect_divergences",
            "description": "Multi-indicator divergence detection"
        },
        "fibonacci_filter": {
            "path": "core.fibonacci_filter",
            "function": "apply_fibonacci_filter",
            "description": "Advanced Fibonacci analysis"
        }
    },
    "Risk Management": {
        "risk_model": {
            "path": "core.risk_model",
            "function": "calculate_risk_metrics",
            "description": "Comprehensive risk analysis"
        },
        "advanced_stoploss_lots_engine": {
            "path": "core.advanced_stoploss_lots_engine",
            "function": "calculate_advanced_stops",
            "description": "Dynamic stop-loss calculation"
        }
    },
    "Market Intelligence": {
        "intermarket_sentiment": {
            "path": "core.intermarket_sentiment",
            "function": "analyze_intermarket",
            "description": "Cross-market correlation analysis"
        },
        "macro_sentiment_enricher": {
            "path": "core.macro_sentiment_enricher",
            "function": "enrich_macro_sentiment",
            "description": "Macro sentiment analysis"
        }
    }
}

class EnhancedLLMConnector:
    """Enhanced LLM Connector with multi-strategy integration"""

    def __init__(self):
        self.analysis_cache = {}

    async def analyze_pattern(self, pattern_data, strategy_results):
        """Comprehensive pattern analysis with all strategies"""
        prompt = f"""
        Analyze this comprehensive market situation:

        Pattern Data: {json.dumps(pattern_data, indent=2)}

        Strategy Results:
        - SMC Analysis: {strategy_results.get('smc', 'N/A')}
        - Wyckoff Phase: {strategy_results.get('wyckoff', 'N/A')}
        - Inducement/Sweep: {strategy_results.get('inducement', 'N/A')}
        - VSA Signals: {strategy_results.get('vsa', 'N/A')}
        - Risk Metrics: {strategy_results.get('risk', 'N/A')}

        Provide:
        1. Market Context (institutional perspective)
        2. Entry Strategy (specific levels and conditions)
        3. Risk Management (stops and targets)
        4. Confluence Score (0-100)
        5. Trade Recommendation
        """

        # Simulate LLM response (replace with actual API call)
        return {
            "analysis": "Multi-strategy confluence detected",
            "entry_strategy": "Wait for sweep of liquidity at X level",
            "risk_management": "Stop below structure, target at liquidity void",
            "confluence_score": 85,
            "recommendation": "HIGH PROBABILITY SETUP"
        }

    def generate_alert(self, analysis_results):
        """Generate actionable alerts from analysis"""
        alert = f"""
        🚨 ZANFLOW ALERT 🚨

        Setup: {analysis_results.get('setup_type', 'Unknown')}
        Confluence: {analysis_results.get('confluence_score', 0)}%

        Entry: {analysis_results.get('entry_level', 'TBD')}
        Stop: {analysis_results.get('stop_level', 'TBD')}
        Target: {analysis_results.get('target_level', 'TBD')}

        Bias: {analysis_results.get('bias', 'Neutral')}
        """
        return alert

# --- Progress-bar enabled UnifiedAnalysisEngine ---
class UnifiedAnalysisEngine:
    """Orchestrates all analysis modules, with live progress bar."""

    def __init__(self):
        self.loaded_modules = {}
        self.llm_connector = EnhancedLLMConnector()

    def load_module(self, module_path, function_name):
        """Dynamically load analysis modules"""
        try:
            if module_path not in self.loaded_modules:
                module = importlib.import_module(module_path)
                self.loaded_modules[module_path] = module

            return getattr(self.loaded_modules[module_path], function_name, None)
        except Exception as e:
            st.error(f"Failed to load {module_path}: {str(e)}")
            return None

    async def run_comprehensive_analysis(self, df, selected_strategies):
        """Run all selected analysis strategies, yielding progress."""
        results = {}
        # Count total strategies (not categories)
        total_strategies = sum(len(strategies) for strategies in selected_strategies.values())
        current = 0
        progress_bar = st.progress(0)

        for category, strategies in selected_strategies.items():
            for strategy_name, strategy_info in strategies.items():
                try:
                    func = self.load_module(
                        strategy_info['path'],
                        strategy_info['function']
                    )
                    if func:
                        st.info(f"Running {strategy_name}...")
                        # Optionally, await if coroutine
                        result = func(df)
                        results[strategy_name] = result
                except Exception as e:
                    st.warning(f"{strategy_name} error: {str(e)}")
                    results[strategy_name] = {"error": str(e)}
                current += 1
                progress_bar.progress(min(current / total_strategies, 1.0))
                await asyncio.sleep(0.05)  # allow UI to update

        # LLM Analysis
        if results:
            llm_analysis = await self.llm_connector.analyze_pattern(
                {"timestamp": datetime.now().isoformat()},
                results
            )
            results['llm_analysis'] = llm_analysis
        progress_bar.progress(1.0)
        return results

# Main Dashboard
def main():
    # Sidebar (more compact)
    with st.sidebar:
        st.markdown(
            '<div style="font-size:14px; color:#FFD700; margin-bottom:0.7rem;"><b>⚙️ Configuration</b></div>',
            unsafe_allow_html=True
        )
        # Module Selection (with expanders)
        st.markdown('<div style="font-size:13px; color:#FFD700;"><b>📊 Select Analysis Modules</b></div>', unsafe_allow_html=True)
        selected_modules = {}
        for category, modules in MODULE_REGISTRY.items():
            with st.expander(f"{category}", expanded=False):
                selected_modules[category] = {}
                for module_name, module_info in modules.items():
                    if st.checkbox(
                        module_name.replace('_', ' ').title(),
                        value=True,
                        help=module_info['description'],
                        key=f"{category}_{module_name}"
                    ):
                        selected_modules[category][module_name] = module_info
        st.markdown('<div style="font-size:13px; color:#FFD700; margin-top:0.7rem;"><b>🎯 Analysis Settings</b></div>', unsafe_allow_html=True)
        lookback_period = st.slider("Lookback Period", 100, 5000, 1000)
        st.markdown('<div style="font-size:13px; color:#FFD700; margin-top:0.7rem;"><b>🤖 LLM Configuration</b></div>', unsafe_allow_html=True)
        use_llm = st.checkbox("Enable LLM Analysis", value=True)
        llm_model = st.selectbox(
            "LLM Model",
            ["Claude Opus", "GPT-4", "Custom Model"]
        )

    # Main Content
    col1, col2 = st.columns([3, 1])

    with col1:
        import glob
        from pathlib import Path
        import re
        df = None
        data_root = Path(st.secrets.get("PARQUET_DATA_DIR", "."))
        # Scan for all symbols (subfolders) and timeframes (by file pattern)
        pattern_list = ["*.csv", "*.parquet"]
        available_files = []
        for pattern in pattern_list:
            available_files.extend(data_root.rglob(pattern))
        # Build symbol -> timeframe -> file mapping
        symbol_tf_map = {}
        tf_order = ["M1", "M5", "M15", "M30", "H1", "H4", "D", "W", "MN"]
        tf_regexes = [
            (r"1m", "M1"), (r"5m", "M5"), (r"15m", "M15"), (r"30m", "M30"),
            (r"60m|1h", "H1"), (r"4h", "H4"), (r"1d|daily", "D"),
            (r"1w|weekly", "W"), (r"1mn|monthly|mn", "MN")
        ]
        for file_path in available_files:
            rel = file_path.relative_to(data_root)
            parts = rel.parts
            if len(parts) < 2:
                continue
            symbol = parts[0]
            tf_found = None
            # Try to match timeframe from filename
            for rgx, tf_lbl in tf_regexes:
                if re.search(rgx, str(file_path).lower()):
                    tf_found = tf_lbl
                    break
            if not tf_found:
                # fallback: try to extract e.g. 1m, 5m, etc.
                m = re.search(r'(\d+)[mhdw]', str(file_path).lower())
                if m:
                    tf_map = {
                        "1": "M1", "5": "M5", "15": "M15", "30": "M30", "60": "H1", "240": "H4"
                    }
                    tf_found = tf_map.get(m.group(1))
            if tf_found:
                symbol_tf_map.setdefault(symbol, {}).setdefault(tf_found, []).append(file_path)
        if not symbol_tf_map:
            st.warning(f"No CSV or Parquet files found under {data_root}.")
            df = None
            selected_symbol = None
            selected_tf = None
            selected_file = None
        else:
            symbols = sorted(symbol_tf_map.keys())
            # Horizontal selectboxes for symbol, then timeframe
            c1, c2 = st.columns([1,1])
            with c1:
                selected_symbol = st.selectbox(
                    "Symbol",
                    symbols,
                    key="symbol_selectbox"
                )
            tfs_for_symbol = [tf for tf in tf_order if tf in symbol_tf_map[selected_symbol]]
            with c2:
                selected_tf = st.selectbox(
                    "Timeframe",
                    tfs_for_symbol,
                    key="tf_selectbox"
                )
            # Find the file for this symbol/timeframe (prefer parquet over csv)
            files = symbol_tf_map[selected_symbol][selected_tf]
            preferred_file = None
            for ext in [".parquet", ".csv"]:
                for f in files:
                    if f.suffix.lower() == ext:
                        preferred_file = f
                        break
                if preferred_file:
                    break
            selected_file = preferred_file
            if not selected_file:
                st.warning(f"No data file found for {selected_symbol} / {selected_tf}.")
                df = None
            else:
                try:
                    if selected_file.suffix.lower() == ".csv":
                        df = pd.read_csv(selected_file)
                    else:
                        df = pd.read_parquet(selected_file)
                    st.success(
                        f"<span style='color:#FFD700;'>Loaded <b>{len(df):,}</b> rows for <b>{selected_symbol} {selected_tf}</b></span>",
                        icon="✅"
                    )
                except Exception as e:
                    st.warning(f"Could not load file: {selected_file}\n{e}")
                    df = None
        # Data Preview in expander
        if df is not None:
            with st.expander("📈 Data Overview", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            # Run Analysis
            if st.button("🚀 Run Comprehensive Analysis", type="primary"):
                engine = UnifiedAnalysisEngine()
                results = asyncio.run(
                    engine.run_comprehensive_analysis(df, selected_modules)
                )
                st.markdown("<div style='color:#FFD700; font-size:15px; margin-top:0.5rem;'><b>📊 Analysis Results</b></div>", unsafe_allow_html=True)
                tabs = st.tabs([
                    "Summary", "SMC Analysis", "Wyckoff",
                    "Risk Analysis", "LLM Insights", "Raw Data"
                ])
                with tabs[0]:  # Summary
                    if 'llm_analysis' in results:
                        llm = results['llm_analysis']
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Confluence Score", f"{llm.get('confluence_score', 0)}%")
                        with c2:
                            st.metric("Setup Quality", llm.get('recommendation', 'N/A'))
                        with c3:
                            st.metric("Risk:Reward", "1:3")
                        with c4:
                            st.metric(
                                "Strategies Aligned",
                                f"{len([r for r in results.values() if not isinstance(r, dict) or 'error' not in r])}"
                            )
                        alert = engine.llm_connector.generate_alert(llm)
                        st.info(alert)
                with tabs[1]:  # SMC Analysis
                    smc_results = {k: v for k, v in results.items() if 'smc' in k.lower()}
                    for name, result in smc_results.items():
                        st.markdown(f"**{name}**")
                        st.json(result)
                with tabs[2]:  # Wyckoff
                    wyckoff_results = {k: v for k, v in results.items() if 'wyckoff' in k.lower()}
                    for name, result in wyckoff_results.items():
                        st.markdown(f"**{name}**")
                        st.json(result)
                with tabs[3]:  # Risk Analysis
                    risk_results = {k: v for k, v in results.items() if 'risk' in k.lower()}
                    for name, result in risk_results.items():
                        st.markdown(f"**{name}**")
                        st.json(result)
                with tabs[4]:  # LLM Insights
                    if 'llm_analysis' in results:
                        st.markdown("### 🤖 AI-Powered Market Insights")
                        llm = results['llm_analysis']
                        st.markdown(f"**Analysis:** {llm.get('analysis')}")
                        st.markdown(f"**Entry Strategy:** {llm.get('entry_strategy')}")
                        st.markdown(f"**Risk Management:** {llm.get('risk_management')}")
                with tabs[5]:  # Raw Data
                    st.json(results)
        # --- Info box with summary stats at the bottom ---
        if df is not None:
            # Total ticks, time range, strategy count
            try:
                ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
                t0 = df[ts_col].iloc[0]
                t1 = df[ts_col].iloc[-1]
                # Try to parse as datetime if possible
                def try_parse(x):
                    try:
                        return pd.to_datetime(x)
                    except Exception:
                        return x
                t0p, t1p = try_parse(t0), try_parse(t1)
                trange = f"{t0p} to {t1p}"
            except Exception:
                trange = "-"
            strat_count = sum(len(v) for v in selected_modules.values())
            info_html = f"""
                <div class="zanflow-infobox">
                    <b>Total Ticks:</b> {len(df):,} &nbsp;|&nbsp;
                    <b>Time Range:</b> {trange} &nbsp;|&nbsp;
                    <b>Strategies:</b> {strat_count}
                </div>
            """
            st.markdown(info_html, unsafe_allow_html=True)

    # Remove col2 sections (Quick Stats, Strategy Mix, Module Status)
    # Instead, keep col2 empty or with a subtle logo if desired.

if __name__ == "__main__":
    main()
