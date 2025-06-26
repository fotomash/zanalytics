
#!/usr/bin/env python3
"""
Economic Data Integration for ZANFLOW v12 Ultimate Trading Dashboard
Integrates macro sentiment analysis with intermarket data
"""

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple, Any

class EconomicDataManager:
    """Manages economic data fetching and analysis"""

    def __init__(self, finnhub_api_key: str, twelve_data_api_key: str):
        self.finnhub_key = finnhub_api_key
        self.twelve_data_key = twelve_data_api_key
        self.base_url_finnhub = "https://finnhub.io/api/v1"
        self.base_url_twelve = "https://api.twelvedata.com"
        self.cache = {}
        self.last_update = None

    def fetch_macro_snapshot(self) -> Dict[str, Any]:
        """Fetch comprehensive macro sentiment snapshot"""
        try:
            snapshot = {
                'timestamp': datetime.now(),
                'vix': self.get_vix_data(),
                'us10y': self.get_us10y_data(),
                'dxy_proxy': self.get_dxy_proxy_data(),
                'forex_major_pairs': self.get_major_forex_pairs(),
                'macro_score': 0  # Will be calculated
            }

            # Calculate macro sentiment score
            snapshot['macro_score'] = self.calculate_macro_sentiment_score(snapshot)

            self.cache['macro_snapshot'] = snapshot
            self.last_update = datetime.now()

            return snapshot

        except Exception as e:
            st.error(f"Error fetching macro data: {e}")
            return self.get_cached_or_default()

    def get_vix_data(self) -> Dict[str, float]:
        """Get VIX volatility index data"""
        try:
            url = f"{self.base_url_finnhub}/quote?symbol=^VIX&token={self.finnhub_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            if 'c' in data:
                return {
                    'current': data['c'],
                    'high': data['h'],
                    'low': data['l'],
                    'open': data['o'],
                    'prev_close': data['pc'],
                    'change': data['c'] - data['pc'],
                    'change_pct': ((data['c'] - data['pc']) / data['pc']) * 100 if data['pc'] != 0 else 0
                }
            else:
                return self.get_fallback_vix()

        except Exception as e:
            st.warning(f"VIX fetch error: {e}")
            return self.get_fallback_vix()

    def get_us10y_data(self) -> Dict[str, float]:
        """Get US 10-Year Treasury yield data"""
        try:
            # Try Finnhub economic endpoint first
            url = f"{self.base_url_finnhub}/economic?symbol=US10Y&token={self.finnhub_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            if isinstance(data, dict) and 'value' in data:
                return {
                    'current': data['value'],
                    'symbol': data.get('symbol', 'US10Y'),
                    'unit': data.get('unit', '%'),
                    'datetime': data.get('datetime', datetime.now().strftime('%Y-%m-%d'))
                }
            else:
                # Fallback to Twelve Data
                return self.get_us10y_twelve_data()

        except Exception as e:
            st.warning(f"US10Y fetch error: {e}")
            return {'current': 4.5, 'symbol': 'US10Y', 'unit': '%', 'datetime': datetime.now().strftime('%Y-%m-%d')}

    def get_us10y_twelve_data(self) -> Dict[str, float]:
        """Fallback US10Y from Twelve Data"""
        try:
            url = f"{self.base_url_twelve}/quote?symbol=US10Y&apikey={self.twelve_data_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            if 'close' in data:
                return {
                    'current': float(data['close']),
                    'symbol': 'US10Y',
                    'unit': '%',
                    'datetime': datetime.now().strftime('%Y-%m-%d')
                }
            else:
                return {'current': 4.5, 'symbol': 'US10Y', 'unit': '%', 'datetime': datetime.now().strftime('%Y-%m-%d')}

        except Exception:
            return {'current': 4.5, 'symbol': 'US10Y', 'unit': '%', 'datetime': datetime.now().strftime('%Y-%m-%d')}

    def get_dxy_proxy_data(self) -> Dict[str, Any]:
        """Get DXY proxy data via major USD pairs"""
        try:
            url = f"{self.base_url_finnhub}/forex/rates?base=USD&token={self.finnhub_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            if 'quote' in data:
                rates = data['quote']

                # Calculate DXY proxy strength
                dxy_components = {
                    'EUR': 1/rates.get('EUR', 1.1) if rates.get('EUR') else 0.9,  # USD/EUR strength
                    'JPY': rates.get('JPY', 150) / 100,  # USD/JPY normalized
                    'GBP': 1/rates.get('GBP', 0.8) if rates.get('GBP') else 1.25,  # USD/GBP strength
                    'CHF': 1/rates.get('CHF', 0.9) if rates.get('CHF') else 1.1,  # USD/CHF strength
                    'CAD': 1/rates.get('CAD', 1.35) if rates.get('CAD') else 0.74,  # USD/CAD strength
                }

                # Simple DXY proxy calculation
                dxy_proxy = np.mean(list(dxy_components.values())) * 100

                return {
                    'dxy_proxy': dxy_proxy,
                    'components': dxy_components,
                    'raw_rates': rates,
                    'usd_strength': 'Strong' if dxy_proxy > 100 else 'Weak'
                }
            else:
                return self.get_fallback_dxy()

        except Exception as e:
            st.warning(f"DXY proxy fetch error: {e}")
            return self.get_fallback_dxy()

    def get_major_forex_pairs(self) -> Dict[str, Dict[str, float]]:
        """Get major forex pairs for correlation analysis"""
        major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD']
        forex_data = {}

        for pair in major_pairs:
            try:
                # Convert to Finnhub format
                symbol = pair.replace('/', '')
                url = f"{self.base_url_finnhub}/quote?symbol=OANDA:{symbol}&token={self.finnhub_key}"
                response = requests.get(url, timeout=5)
                data = response.json()

                if 'c' in data:
                    forex_data[pair] = {
                        'current': data['c'],
                        'change': data['c'] - data['pc'],
                        'change_pct': ((data['c'] - data['pc']) / data['pc']) * 100 if data['pc'] != 0 else 0,
                        'high': data['h'],
                        'low': data['l']
                    }

            except Exception:
                # Fallback values
                forex_data[pair] = {
                    'current': 1.1 if 'EUR' in pair else 1.0,
                    'change': 0,
                    'change_pct': 0,
                    'high': 1.1,
                    'low': 1.0
                }

        return forex_data

    def calculate_macro_sentiment_score(self, snapshot: Dict[str, Any]) -> float:
        """Calculate comprehensive macro sentiment score (-5 to +5)"""
        score = 0

        # VIX component (lower VIX = more bullish for risk assets)
        vix_current = snapshot['vix'].get('current', 20)
        if vix_current < 15:
            score += 2  # Very low fear
        elif vix_current < 20:
            score += 1  # Low fear
        elif vix_current > 30:
            score -= 2  # High fear
        elif vix_current > 25:
            score -= 1  # Elevated fear

        # US10Y component (moderate yields = goldilocks)
        us10y_current = snapshot['us10y'].get('current', 4.5)
        if 3.5 <= us10y_current <= 4.5:
            score += 1  # Goldilocks zone
        elif us10y_current > 5.0:
            score -= 1  # Too high, growth concerns
        elif us10y_current < 3.0:
            score -= 0.5  # Too low, recession fears

        # DXY component (moderate USD strength is balanced)
        dxy_proxy = snapshot['dxy_proxy'].get('dxy_proxy', 100)
        if 95 <= dxy_proxy <= 105:
            score += 0.5  # Balanced USD
        elif dxy_proxy > 110:
            score -= 1  # Too strong USD, emerging market stress
        elif dxy_proxy < 90:
            score -= 0.5  # Weak USD, potential instability

        return max(-5, min(5, score))  # Clamp between -5 and +5

    def get_fallback_vix(self) -> Dict[str, float]:
        """Fallback VIX data"""
        return {
            'current': 20.0,
            'high': 21.0,
            'low': 19.0,
            'open': 20.5,
            'prev_close': 20.2,
            'change': -0.2,
            'change_pct': -1.0
        }

    def get_fallback_dxy(self) -> Dict[str, Any]:
        """Fallback DXY data"""
        return {
            'dxy_proxy': 102.5,
            'components': {'EUR': 0.92, 'JPY': 1.57, 'GBP': 1.25, 'CHF': 1.1, 'CAD': 0.74},
            'raw_rates': {},
            'usd_strength': 'Moderate'
        }

    def get_cached_or_default(self) -> Dict[str, Any]:
        """Get cached data or return defaults"""
        if 'macro_snapshot' in self.cache:
            return self.cache['macro_snapshot']

        return {
            'timestamp': datetime.now(),
            'vix': self.get_fallback_vix(),
            'us10y': {'current': 4.5, 'symbol': 'US10Y', 'unit': '%'},
            'dxy_proxy': self.get_fallback_dxy(),
            'forex_major_pairs': {},
            'macro_score': 0
        }

# Extended Dashboard Class with Economic Integration
class UltimateZANFLOWDashboardWithEconomics(UltimateZANFLOWDashboard):
    """Extended ZANFLOW Dashboard with Economic Data Integration"""

    def __init__(self, data_directory="/Users/tom/Documents/GitHub/zanalytics/data"):
        super().__init__(data_directory)

        # Initialize economic data manager
        self.economic_manager = EconomicDataManager(
            finnhub_api_key="d07lgo1r01qrslhp3q3gd07lgo1r01qrslhp3q40",
            twelve_data_api_key="6a29ddba6b9c4a91b969704fcc1b325f"
        )

        self.macro_data = None

    def create_sidebar_controls(self):
        """Extended sidebar with economic controls"""
        super().create_sidebar_controls()

        # Add economic analysis controls
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🌍 Economic Analysis")

        st.session_state['show_macro_sentiment'] = st.sidebar.checkbox("📊 Macro Sentiment", True)
        st.session_state['show_intermarket'] = st.sidebar.checkbox("🔗 Intermarket Analysis", True)
        st.session_state['show_economic_calendar'] = st.sidebar.checkbox("📅 Economic Impact", False)

        # Macro data refresh
        if st.sidebar.button("🔄 Refresh Macro Data"):
            with st.spinner("Fetching latest macro data..."):
                self.macro_data = self.economic_manager.fetch_macro_snapshot()
                st.sidebar.success("✅ Macro data updated!")

    def display_market_overview(self):
        """Extended market overview with economic data"""
        super().display_market_overview()

        # Add economic overview
        if st.session_state.get('show_macro_sentiment', True):
            self.display_macro_sentiment_overview()

    def display_macro_sentiment_overview(self):
        """Display macro sentiment overview"""
        st.markdown("## 🌍 Macro Sentiment & Intermarket Analysis")

        # Fetch macro data if not already loaded
        if self.macro_data is None:
            with st.spinner("Loading macro sentiment data..."):
                self.macro_data = self.economic_manager.fetch_macro_snapshot()

        if self.macro_data:
            # Macro sentiment score
            macro_score = self.macro_data.get('macro_score', 0)

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                vix_data = self.macro_data.get('vix', {})
                vix_color = "🟢" if vix_data.get('current', 20) < 20 else "🔴" if vix_data.get('current', 20) > 25 else "🟡"
                st.metric(
                    "VIX Fear Index", 
                    f"{vix_color} {vix_data.get('current', 0):.1f}",
                    f"{vix_data.get('change', 0):+.2f}"
                )

            with col2:
                us10y_data = self.macro_data.get('us10y', {})
                st.metric(
                    "US 10Y Yield", 
                    f"📈 {us10y_data.get('current', 0):.2f}%"
                )

            with col3:
                dxy_data = self.macro_data.get('dxy_proxy', {})
                dxy_strength = dxy_data.get('usd_strength', 'Unknown')
                dxy_emoji = "💪" if dxy_strength == 'Strong' else "🤏"
                st.metric(
                    "USD Strength", 
                    f"{dxy_emoji} {dxy_strength}",
                    f"{dxy_data.get('dxy_proxy', 100):.1f}"
                )

            with col4:
                score_emoji = "🚀" if macro_score > 2 else "📈" if macro_score > 0 else "📉" if macro_score > -2 else "💥"
                score_color = "🟢" if macro_score > 1 else "🔴" if macro_score < -1 else "🟡"
                st.metric(
                    "Macro Score", 
                    f"{score_emoji} {score_color}",
                    f"{macro_score:+.1f}/5"
                )

            with col5:
                last_update = self.macro_data.get('timestamp', datetime.now())
                update_diff = (datetime.now() - last_update).total_seconds() / 60
                st.metric(
                    "Data Age", 
                    f"⏱️ {update_diff:.0f}min",
                    "Live" if update_diff < 5 else "Stale"
                )

            # Macro sentiment interpretation
            self.display_macro_sentiment_interpretation(macro_score)

            # Intermarket correlation
            if st.session_state.get('show_intermarket', True):
                self.create_intermarket_analysis()

    def display_macro_sentiment_interpretation(self, score: float):
        """Display macro sentiment interpretation"""
        if score > 2:
            st.success(f"""
            🚀 **Very Bullish Macro Environment** (Score: {score:+.1f}/5)
            - Low fear (VIX), balanced yields, supportive for risk assets
            - Favorable for growth currencies (AUD, NZD, CAD)
            - Consider long bias on risk-on pairs
            """)
        elif score > 0:
            st.info(f"""
            📈 **Moderately Bullish Macro** (Score: {score:+.1f}/5)
            - Generally supportive macro backdrop
            - Selective opportunities in higher-yielding currencies
            - Mild risk-on bias
            """)
        elif score > -2:
            st.warning(f"""
            📉 **Moderately Bearish Macro** (Score: {score:+.1f}/5)
            - Some macro headwinds present
            - Flight to quality possible (USD, JPY, CHF strength)
            - Defensive positioning recommended
            """)
        else:
            st.error(f"""
            💥 **Very Bearish Macro Environment** (Score: {score:+.1f}/5)
            - High fear, unstable yields, risk-off environment
            - Strong safe-haven demand (USD, JPY, CHF)
            - Avoid risk assets, consider defensive trades
            """)

    def create_intermarket_analysis(self):
        """Create intermarket correlation analysis"""
        st.markdown("### 🔗 Intermarket Correlation Analysis")

        if self.macro_data and self.pairs_data:
            # Get forex data for correlation
            forex_pairs = self.macro_data.get('forex_major_pairs', {})

            if forex_pairs:
                tab1, tab2 = st.tabs(["📊 Current Snapshot", "📈 Correlation Matrix"])

                with tab1:
                    self.display_forex_heatmap(forex_pairs)

                with tab2:
                    self.display_macro_forex_correlation()

    def display_forex_heatmap(self, forex_pairs: Dict[str, Dict[str, float]]):
        """Display forex performance heatmap"""
        try:
            # Prepare data for heatmap
            pairs_data = []
            changes_data = []

            for pair, data in forex_pairs.items():
                pairs_data.append(pair)
                changes_data.append(data.get('change_pct', 0))

            if pairs_data and changes_data:
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=[changes_data],
                    x=pairs_data,
                    y=['% Change'],
                    colorscale='RdYlGn',
                    zmid=0,
                    text=[[f"{change:+.2f}%" for change in changes_data]],
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    hoverongaps=False
                ))

                fig.update_layout(
                    title="Major Forex Pairs Performance",
                    template=st.session_state.get('chart_theme', 'plotly_dark'),
                    height=200
                )

                st.plotly_chart(fig, use_container_width=True)

                # Currency strength analysis
                self.analyze_currency_strength(forex_pairs)

        except Exception as e:
            st.error(f"Error creating forex heatmap: {e}")

    def analyze_currency_strength(self, forex_pairs: Dict[str, Dict[str, float]]):
        """Analyze individual currency strength"""
        currency_scores = {}

        # Calculate currency strength based on pair movements
        for pair, data in forex_pairs.items():
            change_pct = data.get('change_pct', 0)
            base_curr = pair.split('/')[0]
            quote_curr = pair.split('/')[1]

            # Base currency strength
            if base_curr not in currency_scores:
                currency_scores[base_curr] = []
            currency_scores[base_curr].append(change_pct)

            # Quote currency strength (inverse)
            if quote_curr not in currency_scores:
                currency_scores[quote_curr] = []
            currency_scores[quote_curr].append(-change_pct)

        # Calculate average strength
        currency_strength = {
            curr: np.mean(scores) for curr, scores in currency_scores.items()
        }

        # Display currency strength ranking
        st.markdown("#### 💪 Currency Strength Ranking")

        sorted_currencies = sorted(currency_strength.items(), key=lambda x: x[1], reverse=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🏆 Strongest Currencies**")
            for curr, strength in sorted_currencies[:3]:
                emoji = "🥇" if strength > 0.5 else "🥈" if strength > 0.2 else "🥉"
                st.markdown(f"{emoji} **{curr}**: {strength:+.2f}%")

        with col2:
            st.markdown("**⚪ Weakest Currencies**")
            for curr, strength in sorted_currencies[-3:]:
                emoji = "🔴" if strength < -0.5 else "🟡" if strength < -0.2 else "🟢"
                st.markdown(f"{emoji} **{curr}**: {strength:+.2f}%")

    def display_macro_forex_correlation(self):
        """Display correlation between macro indicators and forex"""
        try:
            # This would require historical data for proper correlation
            # For now, show theoretical correlations

            correlation_matrix = pd.DataFrame({
                'VIX': [-0.7, -0.6, 0.8, 0.5, -0.4, 0.3],
                'US10Y': [0.6, 0.4, -0.3, -0.2, 0.7, -0.5],
                'DXY': [-0.8, -0.7, 0.9, 0.6, -0.5, 0.8]
            }, index=['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD'])

            fig = px.imshow(
                correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                color_continuous_scale='RdBu',
                aspect="auto",
                title="Theoretical Macro-Forex Correlations"
            )

            fig.update_layout(
                template=st.session_state.get('chart_theme', 'plotly_dark'),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info("📊 **Correlation Guide**: Red = Negative correlation, Blue = Positive correlation")

        except Exception as e:
            st.error(f"Error creating correlation matrix: {e}")

    def display_ultimate_analysis(self):
        """Extended ultimate analysis with economic overlay"""
        super().display_ultimate_analysis()

        # Add economic context to the analysis
        if st.session_state.get('show_macro_sentiment', True):
            self.add_economic_context_to_analysis()

    def add_economic_context_to_analysis(self):
        """Add economic context to pair analysis"""
        pair = st.session_state['selected_pair']

        st.markdown("## 🌍 Economic Context for Current Analysis")

        if self.macro_data is None:
            self.macro_data = self.economic_manager.fetch_macro_snapshot()

        if self.macro_data:
            macro_score = self.macro_data.get('macro_score', 0)

            # Pair-specific economic impact
            self.analyze_pair_economic_impact(pair, macro_score)

    def analyze_pair_economic_impact(self, pair: str, macro_score: float):
        """Analyze economic impact on specific currency pair"""
        base_currency = pair.split('/')[0] if '/' in pair else pair[:3]
        quote_currency = pair.split('/')[1] if '/' in pair else pair[3:6]

        # Currency-specific economic sensitivities
        risk_currencies = ['AUD', 'NZD', 'CAD', 'NOK', 'SEK']
        safe_havens = ['USD', 'JPY', 'CHF']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### 📊 Economic Impact on {pair}")

            if base_currency in risk_currencies or quote_currency in risk_currencies:
                if macro_score > 1:
                    st.success(f"🟢 **Bullish macro environment favors {pair}** - Risk-on sentiment supports commodity/growth currencies")
                elif macro_score < -1:
                    st.error(f"🔴 **Bearish macro environment pressures {pair}** - Risk-off sentiment hurts commodity/growth currencies")
                else:
                    st.info(f"🟡 **Neutral macro impact on {pair}** - Mixed signals from economic indicators")

            elif base_currency in safe_havens or quote_currency in safe_havens:
                if macro_score < -1:
                    st.success(f"🟢 **Flight to safety benefits {pair}** - Safe haven demand increases")
                elif macro_score > 1:
                    st.warning(f"🟡 **Risk-on environment may pressure {pair}** - Reduced safe haven demand")
                else:
                    st.info(f"🟡 **Balanced macro environment for {pair}** - Safe haven flows neutral")

        with col2:
            st.markdown("#### 🎯 Trading Recommendations")

            vix_current = self.macro_data.get('vix', {}).get('current', 20)
            us10y_current = self.macro_data.get('us10y', {}).get('current', 4.5)

            recommendations = []

            if vix_current < 15:
                recommendations.append("Low VIX suggests complacency - watch for volatility spikes")
            elif vix_current > 30:
                recommendations.append("High VIX suggests fear - potential contrarian opportunities")

            if us10y_current > 5.0:
                recommendations.append("High yields may pressure growth-sensitive currencies")
            elif us10y_current < 3.0:
                recommendations.append("Low yields suggest economic concerns")

            if macro_score > 2:
                recommendations.append("Strong macro score - consider risk-on strategies")
            elif macro_score < -2:
                recommendations.append("Weak macro score - defensive positioning recommended")

            for i, rec in enumerate(recommendations[:4], 1):
                st.markdown(f"{i}. {rec}")

def main():
    """Main application entry point with economic integration"""
    dashboard = UltimateZANFLOWDashboardWithEconomics()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()
