# Zanzibar v5.1 Core Module
# Version: 5.1.0
# Module: liquidity_engine_smc.py
# Description: Detects inducement sweeps of ERL points based on HTF structure bias using intermediate timeframe OHLCV data.
# llm_trader_v1_4_2/liquidity_engine_smc.py
# Module responsible for detecting inducement based on HTF structure analysis.
# Takes output from market_structure_analyzer_smc and intermediate timeframe data.
# Includes detailed logging for clarity.

import pandas as pd
import numpy as np  # Needed for example usage block
import json        # Needed for example usage block
import logging

log = logging.getLogger(__name__)

# --- Optional Import for Session Tagging ---
# Ensure session_flow.py exists and contains a tag_session function
# If not available, comment out the import and the 'session' assignment below.
try:
    from core.session_flow import tag_session
    SESSION_TAGGING_ENABLED = True
    # print("INFO (liquidity_engine_smc): session_flow.tag_session imported successfully.") # Less verbose startup
except ImportError:
    SESSION_TAGGING_ENABLED = False
    logging.getLogger(__name__).warning(
        "Could not import tag_session from core.session_flow. Session tagging disabled."
    )
    def tag_session(timestamp):
        """Dummy function if session_flow is not available."""
        return None

# --- Core Inducement Detection Function ---

def detect_inducement_from_structure(intermediate_df, df_timeframe_str, structure_data, sweep_window=30):
    """
    Scans for post-structure inducement sweeps of weak points (ERL)
    in the direction of HTF bias, with detailed logging.

    Args:
        intermediate_df (pd.DataFrame): H1/M15 OHLCV DataFrame with DatetimeIndex.
                                        Needs 'Low', 'High', 'Close', 'Open' columns.
        df_timeframe_str (str): String representation of the intermediate_df timeframe (e.g., 'H1', '15min').
        structure_data (dict): Output dictionary from analyze_market_structure().
        sweep_window (int): Max candles after the ERL point's timestamp to search
                            for a sweep candle.

    Returns:
        dict: inducement_map = {
            'status': True/False,      # True if inducement detected
            'swept_point': dict|None,  # The ERL point that was swept {'timestamp', 'price', 'type'}
            'sweep_candle': dict|None, # Details of the candle that performed the sweep {'timestamp', 'high', 'low', 'close', 'open'}
            'session': str|None        # Session tag (e.g., 'London') if enabled/available
        }
    """
    log.info("--- Entering Inducement Detection (Timeframe: %s) ---", df_timeframe_str)
    # Initialize the result dictionary
    inducement = {
        'status': False,
        'swept_point': None,
        'sweep_candle': None,
        'session': None
    }

    # --- Input Validation ---
    # (Keep validation checks as before, potentially add logging on failure)
    if not isinstance(structure_data, dict):
        log.error("structure_data is not a dictionary. Aborting.")
        return inducement
    if not isinstance(intermediate_df, pd.DataFrame) or intermediate_df.empty:
        log.error("intermediate_df is empty or not a DataFrame. Aborting.")
        return inducement
    required_cols = ['Low', 'High', 'Close', 'Open']
    if not all(col in intermediate_df.columns for col in required_cols):
         log.error("intermediate_df missing required columns: %s. Aborting.", required_cols)
         return inducement
    if not isinstance(intermediate_df.index, pd.DatetimeIndex):
         log.error("intermediate_df index must be a DatetimeIndex. Aborting.")
         return inducement
    if not intermediate_df.index.is_monotonic_increasing:
        log.warning("intermediate_df index is not monotonic increasing. Sorting...")
        intermediate_df = intermediate_df.sort_index()


    # --- Check for Valid Bias ---
    bias = structure_data.get('htf_bias')
    log.info("Task: Checking HTF bias provided in structure_data.")
    log.info("Result: HTF Bias = %s", bias)
    if bias not in ['Bullish', 'Bearish']:
        log.info("Outcome: Bias is not directional ('Bullish' or 'Bearish'). No inducement check needed.")
        return inducement
    log.info("Outcome: Proceeding with %s inducement check.", bias)

    # --- Get Required Structure Data ---
    erl_points = structure_data.get('erl_targets', [])
    structure_points_list = structure_data.get('structure_points', [])
    if not isinstance(erl_points, list): erl_points = []
    if not isinstance(structure_points_list, list): structure_points_list = []

    strong_points = [p for p in structure_points_list if isinstance(p, dict) and 'Strong' in p.get('type', '')]

    if not strong_points:
        log.warning("No 'Strong' structure points found in input. Cannot determine context.")
        return inducement
    if not erl_points:
         log.warning("No ERL targets ('Weak' points) found in input. Cannot find points to sweep.")
         return inducement

    # Get latest strong point
    try:
        strong_points.sort(key=lambda x: x.get('timestamp'))
        latest_strong = strong_points[-1]
        latest_strong_ts = latest_strong.get('timestamp')
        latest_strong_type = latest_strong.get('type')
        if not isinstance(latest_strong_ts, pd.Timestamp):
             log.error("Latest strong point has invalid timestamp. Aborting.")
             return inducement
        log.info("Context: Latest strong structure point = %s at %s", latest_strong_type, latest_strong_ts)
    except (TypeError, KeyError, IndexError) as e:
         log.error("Could not sort or access latest strong point - %s. Aborting.", e)
         return inducement


    # --- Scan ERL Points Formed After the Latest Strong Point ---
    candidate_erl_points = []
    for erl in erl_points:
         if isinstance(erl, dict) and \
            isinstance(erl.get('timestamp'), pd.Timestamp) and \
            isinstance(erl.get('price'), (int, float)) and \
            isinstance(erl.get('type'), str) and \
            ('Weak' in erl['type']) and \
            erl['timestamp'] > latest_strong_ts:
             candidate_erl_points.append(erl)

    candidate_erl_points.sort(key=lambda x: x['timestamp'])
    log.info("Context: Found %s candidate ERL points after latest strong point.", len(candidate_erl_points))

    if not candidate_erl_points:
        log.info("Outcome: No suitable ERL points found after the last strong point.")
        return inducement

    for erl in candidate_erl_points:
        erl_price = erl['price']
        erl_timestamp = erl['timestamp']
        erl_type = erl['type'] # e.g., 'Weak Low' or 'Weak High'

        log.info("Task: Scanning for sweep of ERL point: %s at %.5f (%s)", erl_type, erl_price, erl_timestamp)

        # Ensure the ERL point's timestamp exists in the intermediate DataFrame index
        if erl_timestamp not in intermediate_df.index:
            log.warning("Skipped: ERL timestamp not found in %s data index.", df_timeframe_str)
            continue

        try:
            erl_idx_loc = intermediate_df.index.get_loc(erl_timestamp)
        except KeyError:
            log.warning("Skipped: Could not get index location for ERL timestamp %s.", erl_timestamp)
            continue

        # Define the search window *after* the ERL candle for the sweep candle
        start_scan_idx = erl_idx_loc + 1
        end_scan_idx = min(start_scan_idx + sweep_window, len(intermediate_df))

        log.debug(
            "Task: Analyzing %s candles from index %s to %s (Sweep Window: %s candles)",
            df_timeframe_str,
            start_scan_idx,
            end_scan_idx - 1,
            sweep_window,
        )

        # --- Scan Candles After ERL Point for the Sweep ---
        sweep_found_for_this_erl = False
        for i in range(start_scan_idx, end_scan_idx):
            candle = intermediate_df.iloc[i]
            candle_timestamp = intermediate_df.index[i]

            # Check for wick-based sweep according to bias and ERL type
            sweep_detected = False
            if bias == 'Bullish' and erl_type == 'Weak Low':
                if candle['Low'] < erl_price and candle['Close'] > erl_price:
                    sweep_detected = True
                    log.info(
                        "Result: Bullish Inducement Sweep DETECTED at %s (Swept %s @ %.5f)",
                        candle_timestamp,
                        erl_type,
                        erl_price,
                    )

            elif bias == 'Bearish' and erl_type == 'Weak High':
                if candle['High'] > erl_price and candle['Close'] < erl_price:
                    sweep_detected = True
                    log.info(
                        "Result: Bearish Inducement Sweep DETECTED at %s (Swept %s @ %.5f)",
                        candle_timestamp,
                        erl_type,
                        erl_price,
                    )

            # If sweep detected, populate result and return immediately (first inducement found)
            if sweep_detected:
                session_tag = tag_session(candle_timestamp) if SESSION_TAGGING_ENABLED else None
                log.info(
                    "Sweep Candle Details: T=%s, H=%.5f, L=%.5f, C=%.5f, O=%.5f",
                    candle_timestamp,
                    candle['High'],
                    candle['Low'],
                    candle['Close'],
                    candle['Open'],
                )
                log.info("Session Tag: %s", session_tag if SESSION_TAGGING_ENABLED else "Disabled")

                inducement.update({
                    'status': True,
                    'swept_point': {'timestamp': erl_timestamp, 'price': erl_price, 'type': erl_type},
                    'sweep_candle': {'timestamp': candle_timestamp, 'high': candle['High'], 'low': candle['Low'], 'close': candle['Close'], 'open': candle['Open']},
                    'session': session_tag
                })
                log.info("Outcome: Inducement Confirmed. Returning result.")
                log.info("--- Exiting Inducement Detection ---")
                return inducement # Found the first inducement

        # If inner loop finishes for this ERL without finding a sweep
        if not sweep_found_for_this_erl:
             log.info("Outcome: No sweep found within window for ERL at %s", erl_timestamp)


    # If no inducement found after checking all candidate ERLs
    log.info("Outcome: No inducement found matching criteria for any candidate ERL.")
    log.info("--- Exiting Inducement Detection ---")
    return inducement

# --- Example Usage Block ---

if __name__ == '__main__':
    log.info("\n--- Testing Liquidity Engine SMC with Detailed Logging ---")

    # Example for loading data from either CSV or Parquet file
    # (Uncomment and set the file_path to test with your own data)
    #
    # file_path = "your_data_file.csv"  # or "your_data_file.parquet"
    # # Support both CSV and Parquet formats for input data
    # if file_path.endswith(".csv"):
    #     df = pd.read_csv(file_path)
    # elif file_path.endswith(".parquet"):
    #     df = pd.read_parquet(file_path)
    # else:
    #     raise ValueError("Unsupported file type. Please provide a CSV or Parquet file.")
    #
    # print(df.head())

    # Create Sample DataFrames and Structure Data for testing
    # (Using the same dummy data generation as before)

    # --- Bullish Scenario Setup ---
    dates_h1_bullish = pd.date_range(start='2024-05-14 00:00:00', periods=50, freq='H', tz='UTC')
    dummy_h1_data_bullish = pd.DataFrame({
        'Open': np.linspace(100, 105, 50),
        'High': np.linspace(100, 105, 50) + np.random.rand(50) * 2 + 0.5,
        'Low': np.linspace(100, 105, 50) - np.random.rand(50) * 2 - 0.5,
        'Close': np.linspace(100, 105, 50) + np.random.randn(50) * 0.5,
        'Volume': np.random.randint(100, 1000, 50) # Added Volume
    }, index=dates_h1_bullish)
    dummy_h1_data_bullish['High'] = dummy_h1_data_bullish[['Open', 'High', 'Close']].max(axis=1)
    dummy_h1_data_bullish['Low'] = dummy_h1_data_bullish[['Open', 'Low', 'Close']].min(axis=1)

    erl_time_bullish = pd.Timestamp('2024-05-14 10:00:00', tz='UTC')
    erl_price_bullish = 101.0
    sweep_time_bullish = pd.Timestamp('2024-05-14 13:00:00', tz='UTC') # 3 hours after ERL

    if erl_time_bullish not in dummy_h1_data_bullish.index:
         dummy_h1_data_bullish.loc[erl_time_bullish] = [101.1, 101.3, erl_price_bullish, 101.2, 100]
         dummy_h1_data_bullish = dummy_h1_data_bullish.sort_index()
    else:
         dummy_h1_data_bullish.loc[erl_time_bullish, 'Low'] = min(dummy_h1_data_bullish.loc[erl_time_bullish, 'Low'], erl_price_bullish)

    if sweep_time_bullish in dummy_h1_data_bullish.index:
        dummy_h1_data_bullish.loc[sweep_time_bullish, 'Low'] = erl_price_bullish - 0.1 # Wick below ERL
        dummy_h1_data_bullish.loc[sweep_time_bullish, 'Close'] = erl_price_bullish + 0.2 # Close above ERL
        dummy_h1_data_bullish.loc[sweep_time_bullish, 'Open'] = erl_price_bullish + 0.1
        dummy_h1_data_bullish.loc[sweep_time_bullish, 'High'] = max(dummy_h1_data_bullish.loc[sweep_time_bullish, 'Open'], dummy_h1_data_bullish.loc[sweep_time_bullish, 'Close']) + 0.1
    else:
         log.warning("Sweep time %s not in dummy index, cannot simulate sweep.", sweep_time_bullish)


    sample_structure_data_bullish = {
        'htf_bias': 'Bullish',
        'structure_points': [
            {'timestamp': pd.Timestamp('2024-05-14 05:00:00', tz='UTC'), 'price': 99.0, 'type': 'Strong Low', 'index': 5},
            {'timestamp': erl_time_bullish, 'price': erl_price_bullish, 'type': 'Weak Low', 'index': 10},
        ],
        'erl_targets': [
             {'timestamp': erl_time_bullish, 'price': erl_price_bullish, 'type': 'Weak Low'}
        ],
        'valid_trading_range': {'type': 'Uptrend', 'start':{'timestamp': pd.Timestamp('2024-05-14 05:00:00', tz='UTC'), 'price': 99.0}, 'end':{'timestamp': pd.Timestamp('2024-05-14 15:00:00', tz='UTC'), 'price': 105.0}}
    }

    log.info("\n--- Testing Bullish Scenario ---")
    # Pass the timeframe string explicitly
    inducement_result_bullish = detect_inducement_from_structure(
        dummy_h1_data_bullish,
        'H1', # Explicitly pass the timeframe string
        sample_structure_data_bullish,
        sweep_window=10
    )
    log.info("--- Final Bullish Inducement Result ---")
    log.info(json.dumps(inducement_result_bullish, indent=4, default=str))


    # --- Bearish Scenario Setup ---
    # (Setup code remains the same as previous version)
    dates_h1_bearish = pd.date_range(start='2024-05-15 00:00:00', periods=50, freq='H', tz='UTC')
    dummy_h1_data_bearish = pd.DataFrame({
        'Open': np.linspace(110, 105, 50),
        'High': np.linspace(110, 105, 50) + np.random.rand(50) * 2 + 0.5,
        'Low': np.linspace(110, 105, 50) - np.random.rand(50) * 2 - 0.5,
        'Close': np.linspace(110, 105, 50) + np.random.randn(50) * 0.5,
        'Volume': np.random.randint(100, 1000, 50) # Added Volume
    }, index=dates_h1_bearish)
    dummy_h1_data_bearish['High'] = dummy_h1_data_bearish[['Open', 'High', 'Close']].max(axis=1)
    dummy_h1_data_bearish['Low'] = dummy_h1_data_bearish[['Open', 'Low', 'Close']].min(axis=1)

    erl_time_bearish = pd.Timestamp('2024-05-15 10:00:00', tz='UTC')
    erl_price_bearish = 108.0
    sweep_time_bearish = pd.Timestamp('2024-05-15 14:00:00', tz='UTC')

    if erl_time_bearish not in dummy_h1_data_bearish.index:
         dummy_h1_data_bearish.loc[erl_time_bearish] = [107.9, erl_price_bearish, 107.8, 107.8, 100]
         dummy_h1_data_bearish = dummy_h1_data_bearish.sort_index()
    else:
         dummy_h1_data_bearish.loc[erl_time_bearish, 'High'] = max(dummy_h1_data_bearish.loc[erl_time_bearish, 'High'], erl_price_bearish)

    if sweep_time_bearish in dummy_h1_data_bearish.index:
        dummy_h1_data_bearish.loc[sweep_time_bearish, 'High'] = erl_price_bearish + 0.1
        dummy_h1_data_bearish.loc[sweep_time_bearish, 'Close'] = erl_price_bearish - 0.2
        dummy_h1_data_bearish.loc[sweep_time_bearish, 'Open'] = erl_price_bearish - 0.1
        dummy_h1_data_bearish.loc[sweep_time_bearish, 'Low'] = min(dummy_h1_data_bearish.loc[sweep_time_bearish, 'Open'], dummy_h1_data_bearish.loc[sweep_time_bearish, 'Close']) - 0.1
    else:
         log.warning("Sweep time %s not in dummy index, cannot simulate sweep.", sweep_time_bearish)

    sample_structure_data_bearish = {
        'htf_bias': 'Bearish',
        'structure_points': [
            {'timestamp': pd.Timestamp('2024-05-15 04:00:00', tz='UTC'), 'price': 111.0, 'type': 'Strong High', 'index': 4},
            {'timestamp': erl_time_bearish, 'price': erl_price_bearish, 'type': 'Weak High', 'index': 10},
        ],
         'erl_targets': [
             {'timestamp': erl_time_bearish, 'price': erl_price_bearish, 'type': 'Weak High'}
        ],
        'valid_trading_range': {'type': 'Downtrend', 'start':{...}, 'end':{...}} # Ellipsis for brevity
    }

    log.info("\n--- Testing Bearish Scenario ---")
    # Pass the timeframe string explicitly
    inducement_result_bearish = detect_inducement_from_structure(
        dummy_h1_data_bearish,
        'H1', # Explicitly pass the timeframe string
        sample_structure_data_bearish,
        sweep_window=10
    )
    log.info("--- Final Bearish Inducement Result ---")
    log.info(json.dumps(inducement_result_bearish, indent=4, default=str))

    log.info("\n--- Liquidity Engine SMC Test Complete ---")

