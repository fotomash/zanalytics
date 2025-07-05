# macro_enrichment_engine.py
# Author: ZANZIBAR LLM Assistant (Generated for Tomasz)
# Date: 2025-04-28
# Version: 1.0.0
# Description:
#   Fetches and analyzes key macro indicators (VIX, DXY, Bonds)
#   to provide context (e.g., Risk-ON/OFF state) for asset analysis.

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import json
import traceback

# Unified data provider
try:
    from core.data.client import get_market_data
except Exception as e:
    get_market_data = None
    print(f"[ERROR][MacroEngine] Market data client init failed: {e}")

# Load Copilot config for asset mapping
COPILOT_CONFIG_PATH = Path("config/copilot_config.json")
MACRO_ASSET_MAP = {}
if COPILOT_CONFIG_PATH.is_file():
    try:
        with open(COPILOT_CONFIG_PATH, 'r') as f:
            copilot_config = json.load(f)
        MACRO_ASSET_MAP = copilot_config.get("macro_context_assets", {})
    except Exception as e:
        print(f"[ERROR][MacroEngine] Failed to load copilot config: {e}")

def detect_asset_class(asset_symbol: str) -> str:
    """ Basic detection of asset class based on symbol pattern. """
    asset_symbol = asset_symbol.upper()
    if "USD" in asset_symbol or "EUR" in asset_symbol or "GBP" in asset_symbol or "JPY" in asset_symbol or "CAD" in asset_symbol or "CHF" in asset_symbol or "AUD" in asset_symbol or "NZD" in asset_symbol:
        return "FX"
    if "BTC" in asset_symbol or "ETH" in asset_symbol:
        return "CRYPTO"
    if "XAU" in asset_symbol or "XAG" in asset_symbol or "GOLD" in asset_symbol or "SILVER" in asset_symbol or "CL=" in asset_symbol or "GC=" in asset_symbol:
        return "COMMODITIES"
    if "SPX" in asset_symbol or "NDX" in asset_symbol or "US30" in asset_symbol or "^" in asset_symbol:
         # Crude check for indices, improve if needed
         if asset_symbol not in ["^VIX", "^TNX"]: # Exclude VIX/Yields themselves
              return "INDICES"
    # Fallback or specific checks for Bonds etc.
    if asset_symbol in ["^TNX", "ZB=F", "ZN=F"]: # Add bond symbols
         return "BONDS" # Treat bonds as separate or part of default?
    return "DEFAULT" # Default for stocks, etc.

def detect_risk_state(macro_data: Dict[str, pd.DataFrame]) -> str:
    """ Simple Risk-ON / Risk-OFF detection based on VIX and Bond Yields (TNX). """
    risk_state = "Neutral"
    try:
        vix_df = macro_data.get("^VIX")
        tnx_df = macro_data.get("^TNX") # Assuming ^TNX is fetched

        if vix_df is not None and not vix_df.empty and tnx_df is not None and not tnx_df.empty:
            vix_level = vix_df['Close'].iloc[-1]
            # Check yield trend (simple diff)
            tnx_trend = tnx_df['Close'].iloc[-1] - tnx_df['Close'].iloc[-5] if len(tnx_df) >= 5 else 0

            print(f"[DEBUG][MacroEngine] Risk State Check: VIX={vix_level:.2f}, TNX Trend={tnx_trend:.3f}")

            if vix_level > 25 and tnx_trend < -0.05: # High VIX, falling yields
                risk_state = "Risk OFF"
            elif vix_level < 15 and tnx_trend >= 0: # Low VIX, stable/rising yields
                risk_state = "Risk ON"
            elif vix_level < 18 and tnx_trend > 0.03: # Lowish VIX, rising yields strongly
                 risk_state = "Risk ON"
            elif vix_level > 22: # Elevated VIX tends towards Risk OFF
                 risk_state = "Risk OFF leaning"
            # Add more nuanced rules here

    except Exception as e:
        print(f"[ERROR][MacroEngine] Failed to detect risk state: {e}")
        traceback.print_exc()

    print(f"[INFO][MacroEngine] Determined Risk State: {risk_state}")
    return risk_state

def fetch_macro_context(asset_symbol: str) -> Dict:
    """
    Fetches relevant macro data for a given asset and determines risk state.

    Args:
        asset_symbol (str): The primary asset being analyzed (e.g., "OANDA:EUR_USD").

    Returns:
        Dict: Containing macro data (optional) and risk state.
              {'risk_state': 'Risk ON'/'Risk OFF'/'Neutral', 'macro_data': {'VIX': vix_val, ...}}
    """
    print(f"[INFO][MacroEngine] Fetching macro context for asset: {asset_symbol}")
    context = {"risk_state": "Neutral", "macro_data": {}}
    if not get_market_data:
        print("[ERROR][MacroEngine] Market data client unavailable. Cannot fetch macro context.")
        return context

    asset_class = detect_asset_class(asset_symbol)
    macro_symbols_to_fetch = MACRO_ASSET_MAP.get(asset_class, MACRO_ASSET_MAP.get("DEFAULT", []))

    print(f"[DEBUG][MacroEngine] Asset class: {asset_class}, Fetching symbols: {macro_symbols_to_fetch}")

    macro_raw_data = {}
    # Fetch data for each macro symbol (e.g., last day's worth of H1 or D1)
    # DataManager handles resampling internally
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=2) # Fetch last 2 days for context

    for symbol in macro_symbols_to_fetch:
        print(f"[DEBUG][MacroEngine] Fetching macro symbol: {symbol}")
        try:
            if get_market_data:
                df_context = get_market_data(symbol, 'd1', bars=2)
                if df_context.empty:
                    df_context = get_market_data(symbol, 'h4', bars=12)
            else:
                df_context = pd.DataFrame()
            if not df_context.empty:
                macro_raw_data[symbol] = df_context
                print(f"[DEBUG][MacroEngine] Successfully fetched context data for {symbol}")
            else:
                print(f"[WARN][MacroEngine] No suitable context data found for {symbol}")
        except Exception as e:
            print(f"[ERROR][MacroEngine] Exception fetching {symbol}: {e}")
            traceback.print_exc()

    # Analyze fetched macro data
    context["risk_state"] = detect_risk_state(macro_raw_data)

    # Store latest values for quick reference (optional)
    for symbol, df in macro_raw_data.items():
        if not df.empty:
            context["macro_data"][symbol] = df['Close'].iloc[-1] # Store last closing price

    print(f"[INFO][MacroEngine] Macro context fetched: {context}")
    return context

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Macro Enrichment Engine ---")
    test_asset = "OANDA:XAU_USD"
    # Ensure FINNHUB_API_KEY is set as an environment variable for this test
    if 'FINNHUB_API_KEY' not in os.environ:
         print("WARNING: FINNHUB_API_KEY environment variable not set. Using default/dummy key.")

    if get_market_data:
        macro_info = fetch_macro_context(test_asset)
        print("\n--- Macro Context Result ---")
        print(json.dumps(macro_info, indent=2, default=str))
    else:
        print("\nCannot run test: Market data client is unavailable.")

    print("\n--- Test Complete ---")
