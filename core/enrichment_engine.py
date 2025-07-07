# enrichment_engine.py

import pandas as pd

from poi_manager_smc import detect_poi
from phase_detector_wyckoff_v1 import detect_phases
from confirmation_engine_smc import detect_confirmations
from liquidity_engine_smc import detect_liquidity
from volatility_engine import detect_volatility
from impulse_correction_detector import detect_impulse_corrections
from fibonacci_filter import detect_fibonacci

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Wyckoff phases
    try:
        df['wyckoff_phase'] = detect_phases(df)
    except Exception as e:
        print(f"Wyckoff phase detection error: {e}")

    # SMC Points of Interest
    try:
        for k, v in detect_poi(df).items():
            df[k] = v
    except Exception as e:
        print(f"SMC POI detection error: {e}")

    # Confirmations
    try:
        for k, v in detect_confirmations(df).items():
            df[k] = v
    except Exception as e:
        print(f"Confirmation detection error: {e}")

    # Liquidity detection
    try:
        for k, v in detect_liquidity(df).items():
            df[k] = v
    except Exception as e:
        print(f"Liquidity detection error: {e}")

    # Volatility detection
    try:
        for k, v in detect_volatility(df).items():
            df[k] = v
    except Exception as e:
        print(f"Volatility detection error: {e}")

    # Impulse corrections
    try:
        for k, v in detect_impulse_corrections(df).items():
            df[k] = v
    except Exception as e:
        print(f"Impulse corrections detection error: {e}")

    # Fibonacci filter
    try:
        for k, v in detect_fibonacci(df).items():
            df[k] = v
    except Exception as e:
        print(f"Fibonacci filter detection error: {e}")

    return df