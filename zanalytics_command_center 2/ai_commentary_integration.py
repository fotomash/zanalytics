"""
Enhanced AI Commentary Integration for ZANFLOW
Provides intelligent market analysis and trade ideas
"""

import json
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

class ZanflowAICommentary:
    """
    Integrates AI-powered commentary into your trading system
    Can work with OpenAI, local LLMs, or custom models
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.custom_gpt_endpoint = os.getenv('CUSTOM_GPT_ENDPOINT', 'http://localhost:8080/v1/chat/completions')

    def generate_wyckoff_commentary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive Wyckoff analysis commentary"""

        # Structure the analysis data
        phase = analysis_data.get('current_phase', 'Unknown')
        support_levels = analysis_data.get('support_levels', [])
        resistance_levels = analysis_data.get('resistance_levels', [])
        volume_profile = analysis_data.get('volume_insights', {})
        signals = analysis_data.get('signals', [])

        # Create structured prompt for AI
        prompt = f"""
        As a Wyckoff method expert and senior trading analyst, provide comprehensive market commentary.

        CURRENT MARKET ANALYSIS:
        - Wyckoff Phase: {phase}
        - Key Support Levels: {support_levels}
        - Key Resistance Levels: {resistance_levels}
        - Volume Profile: {json.dumps(volume_profile, indent=2)}
        - Active Signals: {json.dumps(signals, indent=2)}

        Please provide:

        1. MARKET STRUCTURE INTERPRETATION
        - What does the current Wyckoff phase tell us?
        - How strong is the current structure?
        - Key characteristics to watch

        2. CRITICAL LEVELS ANALYSIS
        - Most important support/resistance levels
        - Expected behavior at these levels
        - Volume confirmation requirements

        3. SCENARIO PLANNING
        - Bullish scenario (probability and targets)
        - Bearish scenario (probability and targets)
        - Neutral/consolidation scenario

        4. TRADE OPPORTUNITIES
        - Immediate opportunities based on current phase
        - Entry strategies for each scenario
        - Risk/reward setups

        5. RISK MANAGEMENT
        - Position sizing recommendations
        - Stop loss placement logic
        - When to stay out of the market

        6. NEXT 24-48 HOURS OUTLOOK
        - What to watch for
        - Key events or levels that could change the analysis
        - Preparation steps for traders

        Be specific, actionable, and include actual price levels where relevant.
        Format the response in clear sections with markdown.
        """

        # Generate commentary
        commentary = self._call_ai_service(prompt)

        # Generate specific trade setups
        trade_setups = self._generate_trade_setups(analysis_data)

        # Generate risk alerts
        risk_alerts = self._generate_risk_alerts(analysis_data)

        return {
            'main_commentary': commentary,
            'trade_setups': trade_setups,
            'risk_alerts': risk_alerts,
            'timestamp': datetime.now().isoformat(),
            'confidence_score': self._calculate_confidence(analysis_data)
        }

    def _generate_trade_setups(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific trade setups based on analysis"""

        phase = analysis_data.get('current_phase', 'Unknown')
        support = analysis_data.get('support_levels', [])
        resistance = analysis_data.get('resistance_levels', [])

        setups = []

        if phase == 'Accumulation' and support:
            setups.append({
                'type': 'LONG',
                'strategy': 'Accumulation Spring Trade',
                'entry': f"Buy on test of {support[0]:.5f} with volume confirmation",
                'stop_loss': f"{support[0] * 0.998:.5f} (0.2% below support)",
                'targets': [
                    f"T1: {support[0] * 1.005:.5f} (0.5% gain)",
                    f"T2: {support[0] * 1.010:.5f} (1.0% gain)",
                    f"T3: {resistance[0]:.5f} if available" if resistance else "T3: Trail stop"
                ],
                'risk_reward': '1:3',
                'confidence': 'HIGH' if len(support) > 1 else 'MEDIUM',
                'notes': 'Wait for volume spike on test of support'
            })

        elif phase == 'Distribution' and resistance:
            setups.append({
                'type': 'SHORT',
                'strategy': 'Distribution Upthrust Trade',
                'entry': f"Sell on failed test of {resistance[0]:.5f}",
                'stop_loss': f"{resistance[0] * 1.002:.5f} (0.2% above resistance)",
                'targets': [
                    f"T1: {resistance[0] * 0.995:.5f} (0.5% gain)",
                    f"T2: {resistance[0] * 0.990:.5f} (1.0% gain)",
                    f"T3: {support[0]:.5f} if available" if support else "T3: Trail stop"
                ],
                'risk_reward': '1:3',
                'confidence': 'HIGH' if len(resistance) > 1 else 'MEDIUM',
                'notes': 'Look for decreasing volume on rallies'
            })

        # Add more sophisticated setups based on signals
        for signal in analysis_data.get('signals', []):
            if signal.get('type') == 'buy' and signal.get('strength', 0) > 0.7:
                setups.append(self._create_signal_based_setup(signal, 'LONG', analysis_data))
            elif signal.get('type') == 'sell' and signal.get('strength', 0) > 0.7:
                setups.append(self._create_signal_based_setup(signal, 'SHORT', analysis_data))

        return setups

    def _generate_risk_alerts(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk alerts based on market conditions"""

        alerts = []
        phase = analysis_data.get('current_phase', 'Unknown')
        volume_trend = analysis_data.get('volume_insights', {}).get('trend', 'Neutral')

        # Phase-based alerts
        if phase == 'Distribution':
            alerts.append({
                'level': 'HIGH',
                'message': 'Distribution phase detected - Risk of markdown',
                'action': 'Consider reducing long exposure',
                'urgency': 'IMMEDIATE'
            })

        if phase == 'Markdown':
            alerts.append({
                'level': 'CRITICAL',
                'message': 'Markdown phase active - Bearish momentum strong',
                'action': 'Exit longs, consider short positions only',
                'urgency': 'IMMEDIATE'
            })

        # Volume-based alerts
        if volume_trend == 'Decreasing' and phase in ['Markup', 'Accumulation']:
            alerts.append({
                'level': 'MEDIUM',
                'message': 'Decreasing volume in bullish phase',
                'action': 'Watch for potential reversal',
                'urgency': 'MONITOR'
            })

        # Add more sophisticated risk analysis
        risk_score = self._calculate_risk_score(analysis_data)
        if risk_score > 0.7:
            alerts.append({
                'level': 'HIGH',
                'message': f'Overall risk score elevated: {risk_score:.2f}',
                'action': 'Reduce position sizes, tighten stops',
                'urgency': 'IMMEDIATE'
            })

        return alerts

    def _call_ai_service(self, prompt: str) -> str:
        """Call AI service (OpenAI, Custom GPT, or local LLM)"""

        # Try custom GPT endpoint first
        if self.custom_gpt_endpoint:
            try:
                response = requests.post(
                    self.custom_gpt_endpoint,
                    json={
                        'messages': [
                            {'role': 'system', 'content': 'You are a senior Wyckoff trading analyst.'},
                            {'role': 'user', 'content': prompt}
                        ],
                        'temperature': 0.7,
                        'max_tokens': 1000
                    },
                    headers={'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
                )

                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
            except Exception as e:
                print(f"Custom GPT error: {e}")

        # Fallback to template-based response
        return self._generate_template_response(prompt)

    def _generate_template_response(self, prompt: str) -> str:
        """Generate template-based response when AI service unavailable"""

        # Extract key information from prompt
        lines = prompt.split('\n')
        phase = 'Unknown'
        for line in lines:
            if 'Wyckoff Phase:' in line:
                phase = line.split(':')[1].strip()
                break

        template = f"""
## Market Structure Interpretation

The market is currently in the **{phase}** phase according to Wyckoff analysis.

### Key Observations:
- Volume patterns suggest {phase.lower()} activity
- Price action confirms the phase characteristics
- Smart money positioning is evident

### Critical Levels:
- Monitor support and resistance levels closely
- Volume at these levels will confirm or deny the analysis
- Break of key levels may signal phase transition

### Trading Approach:
- In {phase} phase, focus on appropriate strategies
- Maintain strict risk management
- Wait for confirmation before entering positions

### Risk Management:
- Position size: 1-2% risk per trade
- Stop placement: Beyond key structure levels
- Take profits: Scale out at predetermined targets

### Next 24-48 Hours:
- Continue monitoring volume patterns
- Watch for phase confirmation signals
- Be prepared for potential phase transitions
"""

        return template

    def _create_signal_based_setup(self, signal: Dict[str, Any], 
                                  direction: str, 
                                  analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a trade setup based on a specific signal"""

        current_price = signal.get('price', 0)

        if direction == 'LONG':
            stop_loss = current_price * 0.995  # 0.5% stop
            targets = [
                current_price * 1.005,  # 0.5% target
                current_price * 1.010,  # 1.0% target
                current_price * 1.020   # 2.0% target
            ]
        else:
            stop_loss = current_price * 1.005  # 0.5% stop
            targets = [
                current_price * 0.995,  # 0.5% target
                current_price * 0.990,  # 1.0% target
                current_price * 0.980   # 2.0% target
            ]

        return {
            'type': direction,
            'strategy': f"{signal.get('message', 'Signal-based trade')}",
            'entry': f"Enter at market ({current_price:.5f})",
            'stop_loss': f"{stop_loss:.5f}",
            'targets': [f"T{i+1}: {t:.5f}" for i, t in enumerate(targets)],
            'risk_reward': '1:4',
            'confidence': 'MEDIUM',
            'notes': signal.get('details', 'Follow signal confirmation')
        }

    def _calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the analysis"""

        confidence = 0.5  # Base confidence

        # Phase clarity
        phase = analysis_data.get('current_phase', 'Unknown')
        if phase != 'Unknown':
            confidence += 0.2

        # Support/Resistance levels
        if len(analysis_data.get('support_levels', [])) >= 2:
            confidence += 0.1
        if len(analysis_data.get('resistance_levels', [])) >= 2:
            confidence += 0.1

        # Signals alignment
        signals = analysis_data.get('signals', [])
        if signals:
            signal_types = [s.get('type') for s in signals]
            if len(set(signal_types)) == 1:  # All signals agree
                confidence += 0.1

        return min(confidence, 1.0)

    def _calculate_risk_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate risk score based on market conditions"""

        risk = 0.3  # Base risk

        phase = analysis_data.get('current_phase', 'Unknown')
        if phase == 'Distribution':
            risk += 0.3
        elif phase == 'Markdown':
            risk += 0.4

        # Volume concerns
        volume_trend = analysis_data.get('volume_insights', {}).get('trend', 'Neutral')
        if volume_trend == 'Decreasing':
            risk += 0.2

        return min(risk, 1.0)


# Integration helper for dashboards
class DashboardAIIntegration:
    """Helper class to integrate AI commentary into Streamlit dashboards"""

    def __init__(self):
        self.ai_commentary = ZanflowAICommentary()

    def render_ai_section(self, st, analysis_data: Dict[str, Any]):
        """Render AI commentary section in Streamlit"""

        st.markdown("---")
        st.header("🤖 AI Market Intelligence")

        with st.spinner("Generating AI analysis..."):
            ai_results = self.ai_commentary.generate_wyckoff_commentary(analysis_data)

        # Main commentary
        st.subheader("📊 Market Analysis")
        st.markdown(ai_results['main_commentary'])

        # Trade setups
        if ai_results['trade_setups']:
            st.subheader("💡 Trade Setups")
            for i, setup in enumerate(ai_results['trade_setups'], 1):
                with st.expander(f"Setup {i}: {setup['strategy']} ({setup['type']})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Entry:** {setup['entry']}")
                        st.markdown(f"**Stop Loss:** {setup['stop_loss']}")
                        st.markdown(f"**Risk/Reward:** {setup['risk_reward']}")

                    with col2:
                        st.markdown("**Targets:**")
                        for target in setup['targets']:
                            st.markdown(f"- {target}")

                    st.markdown(f"**Confidence:** {setup['confidence']}")
                    st.info(f"💡 {setup['notes']}")

        # Risk alerts
        if ai_results['risk_alerts']:
            st.subheader("⚠️ Risk Alerts")
            for alert in ai_results['risk_alerts']:
                if alert['level'] == 'CRITICAL':
                    st.error(f"🚨 {alert['message']} - {alert['action']}")
                elif alert['level'] == 'HIGH':
                    st.warning(f"⚠️ {alert['message']} - {alert['action']}")
                else:
                    st.info(f"ℹ️ {alert['message']} - {alert['action']}")

        # Confidence meter
        confidence = ai_results['confidence_score']
        st.subheader("📊 Analysis Confidence")
        st.progress(confidence)
        st.caption(f"Confidence Score: {confidence:.0%}")
