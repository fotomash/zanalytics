# volatility_engine.py
# Author: ZANZIBAR LLM Assistant (Generated for Tomasz)
# Date: 2025-04-28
# Version: 1.0.0
# Description:
#   Analyzes price volatility to determine the current market regime
#   (Quiet, Normal, Explosive) using ATR and Bollinger Band Width.
#   Provides input for adaptive risk sizing and potentially entry filtering.

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import traceback

# --- TA-Lib Check ---
try:
    # Attempt to import the TA-Lib library
    import talib
    TALIB_AVAILABLE = True
    print("[INFO][VolatilityEngine] TA-Lib library found.")
except ImportError:
    # If TA-Lib is not installed, set flag to False and print warning
    TALIB_AVAILABLE = False
    print("[WARN][VolatilityEngine] TA-Lib library not found. Using pandas/numpy fallbacks.")

# --- Indicator Calculation Helpers (with Fallbacks) ---
# These functions calculate specific indicators, using TA-Lib if available
# for performance, otherwise implementing the logic using pandas/numpy.

def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Calculates Average True Range (ATR).

    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
        period (int): The time period for the ATR calculation.

    Returns:
        pd.Series: Series containing the calculated ATR values, or NaNs if calculation fails.
    """
    # Check for sufficient data length
    min_len = period + 1
    if high.isnull().all() or low.isnull().all() or close.isnull().all() or len(high) < min_len:
        return pd.Series(np.nan, index=high.index) # Return NaNs if data is insufficient

    if TALIB_AVAILABLE:
        try:
            # Use TA-Lib's optimized ATR function
            return talib.ATR(high, low, close, timeperiod=period)
        except Exception as e:
            # Log warning and fallback to pandas if TA-Lib fails
            print(f"[WARN][VolatilityEngine] TA-Lib ATR({period}) failed: {e}. Falling back to pandas.")
            pass # Continue to pandas implementation

    # Pandas fallback implementation:
    # Calculate True Range components
    high_low = high - low                      # Current High - Current Low
    high_close_prev = abs(high - close.shift(1)) # Current High - Previous Close
    low_close_prev = abs(low - close.shift(1))  # Current Low - Previous Close
    # True Range is the maximum of the three components
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev}).max(axis=1)
    # Calculate ATR using Exponential Moving Average (common method)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr

def _calculate_bbands(series: pd.Series, period: int, nbdev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates Bollinger Bands (Upper, Middle, Lower).

    Args:
        series (pd.Series): Input data series (typically Close prices).
        period (int): The time period for the middle band (SMA).
        nbdev (float): The number of standard deviations for the upper/lower bands.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: upper_band, middle_band (SMA), lower_band. Returns NaNs if calculation fails.
    """
    nan_series = pd.Series(np.nan, index=series.index) # Pre-create NaN series
    # Check for sufficient data length
    if series.isnull().all() or len(series) < period:
        return nan_series, nan_series, nan_series # Return NaNs if data is insufficient

    if TALIB_AVAILABLE:
        try:
            # Use TA-Lib's optimized BBANDS function (matype=0 specifies SMA for middle band)
            return talib.BBANDS(series, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev, matype=0)
        except Exception as e:
             # Log warning and fallback to pandas if TA-Lib fails
            print(f"[WARN][VolatilityEngine] TA-Lib BBANDS({period},{nbdev}) failed: {e}. Falling back to pandas.")
            pass # Continue to pandas implementation

    # Pandas fallback implementation:
    # Calculate Simple Moving Average (Middle Band)
    middle = series.rolling(window=period, min_periods=period).mean()
    # Calculate Standard Deviation
    std_dev = series.rolling(window=period, min_periods=period).std()
    # Calculate Upper and Lower Bands
    upper = middle + nbdev * std_dev
    lower = middle - nbdev * std_dev
    return upper, middle, lower

# --- Volatility Regime Detection Logic ---

def detect_volatility_regime(
    df: pd.DataFrame,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_stddev: float = 2.0,
    atr_ma_period: int = 10, # Period for smoothing ATR trend
    bbw_ma_period: int = 10, # Period for smoothing BBW trend
    quiet_atr_threshold_pct: float = 0.7, # ATR below X% of its recent average = Quiet
    explosive_atr_threshold_pct: float = 1.5, # ATR above Y% of its recent average = Explosive
    quiet_bbw_threshold_pct: float = 0.6, # BBW below X% of its recent average = Quiet/Squeeze
    explosive_bbw_threshold_pct: float = 1.4 # BBW above Y% of its recent average = Explosive
    ) -> pd.DataFrame:
    """
    Analyzes ATR and Bollinger Band Width to tag volatility regimes.

    Args:
        df (pd.DataFrame): Input OHLCV DataFrame. Requires 'High', 'Low', 'Close'.
        atr_period (int): Period for ATR calculation.
        bb_period (int): Period for Bollinger Bands calculation.
        bb_stddev (float): Standard deviation for Bollinger Bands.
        atr_ma_period (int): Smoothing period for ATR trend analysis.
        bbw_ma_period (int): Smoothing period for BBW trend analysis.
        quiet_atr_threshold_pct (float): ATR threshold for 'Quiet' regime (as % of ATR MA).
        explosive_atr_threshold_pct (float): ATR threshold for 'Explosive' regime (as % of ATR MA).
        quiet_bbw_threshold_pct (float): BBW threshold for 'Quiet' regime (as % of BBW MA).
        explosive_bbw_threshold_pct (float): BBW threshold for 'Explosive' regime (as % of BBW MA).


    Returns:
        pd.DataFrame: DataFrame with added columns:
            - 'atr_14': Calculated ATR (or based on atr_period).
            - 'bbw': Bollinger Band Width (Normalized).
            - 'volatility_regime': 'Quiet', 'Normal', 'Explosive'.
            - 'bb_squeeze': Boolean indicating potential BB Squeeze.
    """
    df_out = df.copy() # Work on a copy to avoid modifying original DataFrame
    print("[INFO][VolatilityEngine] Detecting volatility regimes...")

    # --- Input Validation ---
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df_out.columns for col in required_cols):
        print("[WARN][VolatilityEngine] Missing required columns (HLC). Skipping regime detection.")
        df_out['volatility_regime'] = 'N/A (Missing Data)'
        df_out['bb_squeeze'] = False
        return df_out
    # Check for sufficient data length for all calculations
    min_data_needed = max(atr_period + atr_ma_period, bb_period + bbw_ma_period) + 1
    if df_out.empty or len(df_out) < min_data_needed:
        print(f"[WARN][VolatilityEngine] Insufficient data ({len(df_out)} rows, need ~{min_data_needed}). Skipping volatility calculations.")
        df_out['volatility_regime'] = 'N/A (Insufficient Data)'
        df_out['bb_squeeze'] = False
        return df_out

    try:
        # --- Calculations ---
        # 1. Calculate ATR and its Moving Average
        atr_col = f'ATR_{atr_period}'
        df_out[atr_col] = _calculate_atr(df_out['High'], df_out['Low'], df_out['Close'], atr_period)
        atr_ma_col = f'{atr_col}_MA_{atr_ma_period}'
        df_out[atr_ma_col] = df_out[atr_col].rolling(window=atr_ma_period, min_periods=atr_ma_period).mean()

        # 2. Calculate Bollinger Bands and Band Width (BBW) and its Moving Average
        bb_upper, bb_middle, bb_lower = _calculate_bbands(df_out['Close'], bb_period, bb_stddev)
        # Normalize BBW by dividing by the middle band (SMA) to make it comparable across price levels
        # Replace potential zero middle band values with NaN before division
        bb_middle_safe = bb_middle.replace(0, np.nan)
        df_out['bbw'] = (bb_upper - bb_lower) / bb_middle_safe
        # Handle potential NaNs resulting from division or initial calculations
        df_out['bbw'] = df_out['bbw'].fillna(method='bfill').fillna(0) # Backfill then fill remaining NaNs with 0
        bbw_ma_col = f'bbw_MA_{bbw_ma_period}'
        df_out[bbw_ma_col] = df_out['bbw'].rolling(window=bbw_ma_period, min_periods=bbw_ma_period).mean()

        # --- Regime Classification ---
        # 3. Determine Regime based on thresholds relative to moving averages
        df_out['volatility_regime'] = 'Normal' # Default to 'Normal'

        # --- Quiet Regime Conditions ---
        # Condition: Low ATR relative to its average AND Low BBW relative to its average
        # Check if MA columns exist and are not all NaN before comparing
        if atr_ma_col in df_out and bbw_ma_col in df_out and \
           df_out[atr_ma_col].notna().any() and df_out[bbw_ma_col].notna().any():
            quiet_cond = (
                (df_out[atr_col] < df_out[atr_ma_col] * quiet_atr_threshold_pct) &
                (df_out['bbw'] < df_out[bbw_ma_col] * quiet_bbw_threshold_pct)
            )
            df_out.loc[quiet_cond, 'volatility_regime'] = 'Quiet'
        else:
            quiet_cond = pd.Series(False, index=df_out.index) # Default to False if MAs are missing
            print("[WARN][VolatilityEngine] Cannot determine Quiet regime: ATR MA or BBW MA missing/NaN.")


        # --- Explosive Regime Conditions ---
        # Condition: High ATR relative to its average OR High BBW relative to its average
        # Check if MA columns exist and are not all NaN before comparing
        if atr_ma_col in df_out and bbw_ma_col in df_out and \
           df_out[atr_ma_col].notna().any() and df_out[bbw_ma_col].notna().any():
            explosive_cond = (
                (df_out[atr_col] > df_out[atr_ma_col] * explosive_atr_threshold_pct) |
                (df_out['bbw'] > df_out[bbw_ma_col] * explosive_bbw_threshold_pct)
            )
            # Ensure Quiet takes precedence if conditions overlap slightly
            # Only classify as Explosive if not already classified as Quiet
            df_out.loc[explosive_cond & (~quiet_cond), 'volatility_regime'] = 'Explosive'
        else:
             print("[WARN][VolatilityEngine] Cannot determine Explosive regime: ATR MA or BBW MA missing/NaN.")

        # --- BB Squeeze Tagging ---
        # 4. Tag BB Squeeze (Simplified: BBW is low relative to its recent history)
        if bbw_ma_col in df_out and df_out[bbw_ma_col].notna().any():
             df_out['bb_squeeze'] = df_out['bbw'] < df_out[bbw_ma_col] * quiet_bbw_threshold_pct
        else:
             df_out['bb_squeeze'] = False # Default to False if BBW MA is missing
             print("[WARN][VolatilityEngine] Cannot determine BB Squeeze: BBW MA missing/NaN.")


        print("[INFO][VolatilityEngine] Volatility regime detection complete.")

    except Exception as e:
        print(f"[ERROR][VolatilityEngine] Failed during volatility regime detection: {e}")
        traceback.print_exc()
        # Add empty/error columns if calculation failed partway through
        if 'volatility_regime' not in df_out: df_out['volatility_regime'] = 'N/A (Error)'
        if 'bb_squeeze' not in df_out: df_out['bb_squeeze'] = False
        # Ensure other potentially calculated columns exist even on error
        if f'ATR_{atr_period}' not in df_out: df_out[f'ATR_{atr_period}'] = np.nan
        if 'bbw' not in df_out: df_out['bbw'] = np.nan

    # Optionally drop intermediate MA columns if not needed downstream
    # df_out = df_out.drop(columns=[atr_ma_col, bbw_ma_col], errors='ignore')

    return df_out


# --- Function to get the latest profile ---
def get_volatility_profile(df: pd.DataFrame, config: Optional[Dict] = None) -> Dict:
    """
    Calculates volatility metrics on the input DataFrame and returns the profile
    for the *latest* timestamp.

    Args:
        df (pd.DataFrame): Input OHLCV DataFrame.
        config (Dict, optional): Configuration for detect_volatility_regime.

    Returns:
        Dict: Profile containing latest ATR, BBW, Regime, Squeeze status.
              Returns {'error': ...} if calculation fails or no data.
    """
    profile = {
        "timestamp": None,
        "atr_value": None,
        "bbw_value": None,
        "volatility_regime": None,
        "bb_squeeze": None,
        "error": None
    }
    if df is None or df.empty:
        profile["error"] = "Input DataFrame is empty."
        return profile

    if config is None: config = {} # Use empty dict if no config provided

    try:
        # Calculate regimes on the input dataframe
        # Pass the config dictionary using **kwargs
        vol_df = detect_volatility_regime(df.copy(), **config) # Use copy

        if vol_df.empty:
            profile["error"] = "Volatility regime calculation returned empty DataFrame."
            return profile

        # Get the last row which contains the latest profile
        latest_profile_series = vol_df.iloc[-1]

        # Extract relevant info safely using .get()
        atr_period = config.get('atr_period', 14) # Get period from config or default
        atr_col = f"ATR_{atr_period}"

        # Update profile dictionary with values from the latest row
        profile.update({
            "timestamp": latest_profile_series.name.isoformat() if isinstance(latest_profile_series.name, pd.Timestamp) else str(latest_profile_series.name),
            "atr_value": latest_profile_series.get(atr_col),
            "bbw_value": latest_profile_series.get('bbw'),
            "volatility_regime": latest_profile_series.get('volatility_regime'),
            "bb_squeeze": bool(latest_profile_series.get('bb_squeeze', False)), # Ensure boolean type
        })

        # Round float values for cleaner output, handle potential NaNs
        if profile["atr_value"] is not None and pd.notna(profile["atr_value"]):
             profile["atr_value"] = round(profile["atr_value"], 5)
        else: profile["atr_value"] = None # Ensure None if NaN

        if profile["bbw_value"] is not None and pd.notna(profile["bbw_value"]):
             profile["bbw_value"] = round(profile["bbw_value"], 5)
        else: profile["bbw_value"] = None # Ensure None if NaN

    except Exception as e:
        print(f"[ERROR][VolatilityEngine] Failed getting volatility profile: {e}")
        traceback.print_exc()
        profile["error"] = str(e) # Store error message

    return profile


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Volatility Engine ---")
    # Create dummy data
    periods = 200
    data = {
        'Open': np.random.rand(periods) * 10 + 100,
        'High': np.random.rand(periods) * 1 + 100.5,
        'Low': np.random.rand(periods) * 1 + 99.5,
        'Close': np.random.rand(periods) * 10 + 100,
        'Volume': np.random.randint(100, 10000, size=periods).astype(float)
    }
    index = pd.date_range(start='2024-01-01', periods=periods, freq='H')
    dummy_df = pd.DataFrame(data, index=index)
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    dummy_df['High'] = dummy_df[['Open', 'Close']].max(axis=1) + np.random.rand(periods)*0.5
    dummy_df['Low'] = dummy_df[['Open', 'Close']].min(axis=1) - np.random.rand(periods)*0.5
    # Ensure OHLC types are float
    for col in ['Open', 'High', 'Low', 'Close']:
        dummy_df[col] = dummy_df[col].astype(float)

    # Simulate a squeeze period then expansion
    squeeze_start, squeeze_end = 50, 100
    expansion_start = 150
    # Reduce range during squeeze
    close_mean_squeeze = dummy_df['Close'][squeeze_start:squeeze_end].mean()
    dummy_df.loc[index[squeeze_start:squeeze_end], 'High'] = close_mean_squeeze + 0.1
    dummy_df.loc[index[squeeze_start:squeeze_end], 'Low'] = close_mean_squeeze - 0.1
    dummy_df.loc[index[squeeze_start:squeeze_end], 'Close'] = np.random.uniform(low=close_mean_squeeze-0.05, high=close_mean_squeeze+0.05, size=squeeze_end-squeeze_start)
    # Increase range during expansion
    dummy_df.loc[index[expansion_start:], 'High'] = dummy_df['High'][expansion_start:] + np.random.rand(periods-expansion_start)*1.5
    dummy_df.loc[index[expansion_start:], 'Low'] = dummy_df['Low'][expansion_start:] - np.random.rand(periods-expansion_start)*1.5


    print("Calculating Volatility Regimes...")
    # Define custom thresholds for testing different regimes
    test_config = {
        'atr_period': 14, 'bb_period': 20, 'bb_stddev': 2.0,
        'atr_ma_period': 20, 'bbw_ma_period': 20, # Longer MAs for stability
        'quiet_atr_threshold_pct': 0.6, 'explosive_atr_threshold_pct': 1.8,
        'quiet_bbw_threshold_pct': 0.5, 'explosive_bbw_threshold_pct': 1.6
    }
    vol_df = detect_volatility_regime(dummy_df.copy(), **test_config)

    print("\nSample Output DataFrame (Tail):")
    display_cols = ['Close', 'ATR_14', 'bbw', 'volatility_regime', 'bb_squeeze']
    # Ensure calculated columns exist before trying to display them
    display_cols = [col for col in display_cols if col in vol_df.columns]
    # Use pandas display options for better formatting
    with pd.option_context('display.max_rows', 15, 'display.max_columns', None, 'display.width', 120):
        print(vol_df[display_cols].tail(15).round(4))


    print("\nGetting Latest Volatility Profile:")
    profile = get_volatility_profile(dummy_df, config=test_config)
    print(json.dumps(profile, indent=2))

    print("\n--- Test Complete ---")
