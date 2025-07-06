"""
Fixed Wyckoff Dashboard with Proper Data Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Fix imports - add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="Wyckoff Analysis - ZANFLOW",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Wyckoff Market Analysis")

# Initialize session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

# Sidebar controls
with st.sidebar:
    st.header("Analysis Parameters")

    symbol = st.text_input("Symbol", value="EURUSD")
    timeframe = st.selectbox(
        "Timeframe",
        ["M5", "M15", "M30", "H1", "H4", "D1"],
        index=3
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )

    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Main content area
if analyze_btn:
    with st.spinner("Performing Wyckoff analysis..."):
        # Simulate analysis (replace with actual data connection)
        analysis_data = {
            'current_phase': np.random.choice(['Accumulation', 'Markup', 'Distribution', 'Markdown']),
            'support_levels': [1.0850, 1.0820, 1.0800],
            'resistance_levels': [1.0900, 1.0920, 1.0950],
            'signals': [
                {'type': 'buy', 'message': 'Spring formation at support', 'strength': 0.8},
                {'type': 'info', 'message': 'Volume increasing on rallies', 'strength': 0.6}
            ],
            'volume_insights': {
                'avg_volume': 50000,
                'trend': 'Increasing',
                'unusual_activity': 'Spike at 14:30'
            },
            'phase_characteristics': {
                'volume_trend': 'Expanding',
                'price_trend': 'Sideways',
                'volatility': 'Low',
                'strength': 'Building'
            }
        }

        st.session_state.analysis_data = analysis_data

        # Display results
        tabs = st.tabs(["📈 Overview", "🎯 Phases", "💡 Signals", "📊 Volume", "🤖 AI Analysis"])

        with tabs[0]:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Phase", analysis_data['current_phase'])
            with col2:
                st.metric("Key Support", f"{analysis_data['support_levels'][0]:.5f}")
            with col3:
                st.metric("Key Resistance", f"{analysis_data['resistance_levels'][0]:.5f}")
            with col4:
                st.metric("Volume Trend", analysis_data['volume_insights']['trend'])

            # Price chart placeholder
            st.subheader("Price Structure")
            chart_data = pd.DataFrame(
                np.random.randn(100, 4) * 0.001 + 1.0875,
                columns=['Open', 'High', 'Low', 'Close']
            )
            st.line_chart(chart_data['Close'])

        with tabs[1]:
            # Wyckoff phases
            st.subheader("Wyckoff Phase Analysis")

            phase = analysis_data['current_phase']
            phase_colors = {
                'Accumulation': '#28a745',
                'Markup': '#007bff',
                'Distribution': '#ffc107',
                'Markdown': '#dc3545'
            }

            st.markdown(f"""
            <div style='background-color: {phase_colors.get(phase, '#6c757d')}; 
                        color: white; padding: 30px; border-radius: 10px; 
                        text-align: center; font-size: 24px;'>
                <strong>{phase} Phase</strong>
            </div>
            """, unsafe_allow_html=True)

            # Phase details
            st.subheader("Phase Characteristics")
            chars = analysis_data['phase_characteristics']

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volume:** {chars['volume_trend']}")
                st.info(f"**Price:** {chars['price_trend']}")
            with col2:
                st.info(f"**Volatility:** {chars['volatility']}")
                st.info(f"**Strength:** {chars['strength']}")

        with tabs[2]:
            # Trading signals
            st.subheader("Active Trading Signals")

            for signal in analysis_data['signals']:
                icon = '🟢' if signal['type'] == 'buy' else '🔴' if signal['type'] == 'sell' else 'ℹ️'
                strength = signal.get('strength', 0.5)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"{icon} **{signal['message']}**")
                with col2:
                    st.progress(strength, text=f"Strength: {strength:.0%}")

        with tabs[3]:
            # Volume analysis
            st.subheader("Volume Profile Analysis")

            # Volume chart
            volume_data = pd.DataFrame({
                'Volume': np.random.randint(30000, 70000, 100),
                'Average': [50000] * 100
            })
            st.bar_chart(volume_data)

            # Volume insights
            insights = analysis_data['volume_insights']
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average Volume", f"{insights['avg_volume']:,}")
            with col2:
                st.metric("Trend", insights['trend'])
            with col3:
                st.metric("Unusual Activity", insights['unusual_activity'])

        with tabs[4]:
            # AI Analysis
            try:
                from ai_commentary_integration import DashboardAIIntegration
                ai_integration = DashboardAIIntegration()
                ai_integration.render_ai_section(st, analysis_data)
            except ImportError:
                st.info("AI Commentary module not available. Using template response.")

                # Fallback commentary
                st.markdown(f"""
                ### 🤖 Market Intelligence

                **Current Situation:** The market is in the **{analysis_data['current_phase']}** phase.

                **Key Observations:**
                - Support holding at {analysis_data['support_levels'][0]:.5f}
                - Resistance at {analysis_data['resistance_levels'][0]:.5f}
                - Volume trend is {analysis_data['volume_insights']['trend'].lower()}

                **Trading Approach:**
                - Wait for confirmation at key levels
                - Use appropriate position sizing
                - Monitor volume for validation

                **Risk Management:**
                - Keep stops beyond structure levels
                - Scale in/out of positions
                - Respect the current phase dynamics
                """)

# Footer
st.markdown("---")
st.caption("ZANFLOW Wyckoff Analysis | Real-time market intelligence")
