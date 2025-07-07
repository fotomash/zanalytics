import streamlit as st
import requests
import pandas as pd
import json

def display_enriched_metrics(symbol: str, api_url: str = "http://mm20.local:8080"):
    """Display enriched indicator metrics"""

    try:
        response = requests.get(f"{api_url}/data/{symbol}")
        if response.status_code == 200:
            data = response.json()
            indicators = data.get('indicators', {})

            # Display in organized columns
            st.markdown("### 📊 Live Indicators")

            # Trend indicators
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("**Moving Averages**")
                st.text(f"EMA 8: {indicators.get('ema_8', 'N/A')}")
                st.text(f"EMA 21: {indicators.get('ema_21', 'N/A')}")
                st.text(f"EMA 55: {indicators.get('ema_55', 'N/A')}")
                st.text(f"SMA 200: {indicators.get('sma_200', 'N/A')}")

            with col2:
                st.markdown("**Momentum**")
                rsi = indicators.get('rsi', 50)
                st.text(f"RSI: {rsi:.1f}")
                if rsi > 70:
                    st.error("⚠️ Overbought")
                elif rsi < 30:
                    st.success("⚠️ Oversold")

                st.text(f"Stoch K: {indicators.get('stoch_k', 'N/A')}")
                st.text(f"Stoch D: {indicators.get('stoch_d', 'N/A')}")
                st.text(f"CCI: {indicators.get('cci', 'N/A')}")

            with col3:
                st.markdown("**Volatility**")
                st.text(f"ATR: {indicators.get('atr', 'N/A')}")
                st.text(f"BB Upper: {indicators.get('bb_upper', 'N/A')}")
                st.text(f"BB Lower: {indicators.get('bb_lower', 'N/A')}")
                st.text(f"BB Width: {indicators.get('bb_bandwidth', 'N/A')}%")

            with col4:
                st.markdown("**Trend Strength**")
                st.text(f"ADX: {indicators.get('adx', 'N/A')}")
                st.text(f"MACD: {indicators.get('macd', 'N/A')}")
                st.text(f"Signal: {indicators.get('macd_signal', 'N/A')}")
                trend = indicators.get('trend', 'NEUTRAL')
                if 'UP' in trend:
                    st.success(f"📈 {trend}")
                elif 'DOWN' in trend:
                    st.error(f"📉 {trend}")
                else:
                    st.info(f"➡️ {trend}")

            # Market state summary
            st.markdown("---")
            momentum = indicators.get('momentum', 'NEUTRAL')
            price_bb = indicators.get('price_bb_position', 50)

            if momentum == 'OVERBOUGHT':
                st.warning("⚡ Market is OVERBOUGHT - Consider taking profits")
            elif momentum == 'OVERSOLD':
                st.info("⚡ Market is OVERSOLD - Watch for reversal")
            elif momentum == 'BULLISH':
                st.success("⚡ Bullish momentum detected")
            elif momentum == 'BEARISH':
                st.error("⚡ Bearish momentum detected")

            return True
        else:
            st.error(f"No data available for {symbol}")
            return False

    except Exception as e:
        st.error(f"Error fetching enriched data: {e}")
        return False
