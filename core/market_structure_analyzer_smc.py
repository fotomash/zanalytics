# llm_trader_v1_4_2/market_structure_analyzer_smc.py
# Generated code based on logic defined in MarketStructureAnalyzer.json
# NOTE: This code provides a functional starting point based on the JSON logic.
#       Review, testing, and refinement (especially for swing detection parameters,
#       BoS confirmation rules, and sweep definitions) are recommended.

import pandas as pd
import numpy as np
from datetime import datetime, timezone # For timestamp formatting in logs

# --- Logging Helper ---
# Assuming a similar log_info function exists or is imported
# For standalone testing, define a basic one:
def log_info(message, level="INFO"):
    """Prepends module name to log messages and adds London time."""
    module_name = "MSA_SMC" # Module name for Market Structure Analyzer
    try:
        # Format message if it's a timestamp
        log_message = message
        if isinstance(message, pd.Timestamp):
             if message.tzinfo is None: message = message.tz_localize('UTC')
             london_time = message.tz_convert('Europe/London')
             log_message = london_time.strftime('%Y-%m-%d %H:%M:%S %Z')

        # Get current time in London for prefix
        now_london = pd.Timestamp.now(tz='Europe/London').strftime('%H:%M:%S %Z')
        print(f"[{level}][{module_name}][{now_london}] {log_message}")
    except Exception as log_err:
        # Fallback if timezone handling fails
        print(f"[{level}][{module_name}] {message} (Log Timezone Error: {log_err})")


# --- Helper Functions ---

def find_swing_highs_lows(df, n=5):
    """
    Identifies swing highs and lows using a simple rolling window method.
    Args:
        df (pd.DataFrame): Input OHLCV data with DatetimeIndex.
        n (int): Number of candles on each side to check for HH/LL.
    Returns:
        pd.DataFrame: DataFrame with added 'swing_high' and 'swing_low' columns.
                     Value is the price of the swing high/low at that index, else NaN.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        # Return an empty DataFrame with expected columns if input is invalid
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'swing_high', 'swing_low'])

    df_out = df.copy() # Work on a copy
    df_out['swing_high'] = np.nan
    df_out['swing_low'] = np.nan

    # Ensure columns exist before trying to access
    if not all(col in df_out.columns for col in ['High', 'Low']):
        log_info("Missing 'High' or 'Low' column in DataFrame.", "ERROR")
        return df_out # Return df with NaN swing columns

    # Iterate through the DataFrame indices safely
    for i in range(n, len(df_out) - n):
        try:
            current_index = df_out.index[i]
            # Define window indices relative to current index i
            start_idx = max(0, i - n)
            end_idx = min(len(df_out), i + n + 1)
            window_indices = df_out.index[start_idx:end_idx]


            # Use .loc with index labels for window slicing and access
            window = df_out.loc[window_indices]
            current_high = df_out.loc[current_index, 'High']
            current_low = df_out.loc[current_index, 'Low']

            # Skip if current values are NaN
            if pd.isna(current_high) or pd.isna(current_low):
                continue

            # Check for Swing High (Strict: higher than n bars left and right)
            # Ensure window max/min calculation ignores NaNs
            if current_high == window['High'].max(skipna=True):
                 is_high = True
                 # Check left (up to n candles before index i)
                 if not window['High'].iloc[:n].dropna().empty and (window['High'].iloc[:n].dropna() >= current_high).any():
                     is_high = False
                 # Check right (up to n candles after index i)
                 if is_high and not window['High'].iloc[n+1:].dropna().empty and (window['High'].iloc[n+1:].dropna() >= current_high).any():
                     is_high = False
                 # Assign if it's a strict high relative to neighbors
                 if is_high:
                      df_out.loc[current_index, 'swing_high'] = current_high

            # Check for Swing Low (Strict: lower than n bars left and right)
            if current_low == window['Low'].min(skipna=True):
                is_low = True
                 # Check left (up to n candles before index i)
                if not window['Low'].iloc[:n].dropna().empty and (window['Low'].iloc[:n].dropna() <= current_low).any():
                    is_low = False
                 # Check right (up to n candles after index i)
                if is_low and not window['Low'].iloc[n+1:].dropna().empty and (window['Low'].iloc[n+1:].dropna() <= current_low).any():
                    is_low = False
                # Assign if it's a strict low relative to neighbors
                if is_low:
                     df_out.loc[current_index, 'swing_low'] = current_low
        except Exception as e:
            # Log error but continue processing other points
            log_info(f"Error finding swing point at index {i}: {e}", "ERROR")
            continue

    return df_out

def detect_bos(df, swing_index, look_forward=20, is_check_high=True):
    """
    Detects Break of Structure (BoS) after a given swing point.
    Args:
        df (pd.DataFrame): OHLCV data with swing points potentially identified.
        swing_index (int): Index location of the swing point (e.g., a confirmed low or high).
        look_forward (int): How many candles to look forward for a break.
        is_check_high (bool): True to check for break *above* a swing high, False for break *below* swing low.
    Returns:
        tuple: (index_of_break, price_of_break) or (None, None) if no break.
        NOTE: Uses Candle Body Close for BoS confirmation - this rule might need refinement.
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame) or not all(c in df.columns for c in ['High', 'Low', 'Close']):
        # log_info("detect_bos: Input DataFrame missing required columns.", "WARN") # Less verbose
        return None, None
    if swing_index < 0 or swing_index >= len(df) - 1:
         # print(f"[WARN][MSA_SMC] detect_bos: Invalid swing_index {swing_index} for df length {len(df)}.")
         return None, None # Cannot look forward from the last candle or invalid index

    try:
        if is_check_high:
            swing_price = df['High'].iloc[swing_index]
            # Check if swing_price is valid
            if pd.isna(swing_price): return None, None
            for i in range(swing_index + 1, min(swing_index + 1 + look_forward, len(df))):
                close_price = df['Close'].iloc[i]
                if pd.notna(close_price) and close_price > swing_price:
                    return i, close_price # Return index and closing price of breaking candle
        else: # Check for break below swing low
            swing_price = df['Low'].iloc[swing_index]
             # Check if swing_price is valid
            if pd.isna(swing_price): return None, None
            for i in range(swing_index + 1, min(swing_index + 1 + look_forward, len(df))):
                close_price = df['Close'].iloc[i]
                if pd.notna(close_price) and close_price < swing_price:
                    return i, close_price # Return index and closing price of breaking candle
    except Exception as e:
        log_info(f"Error during BoS detection near index {swing_index}: {e}", "ERROR")

    return None, None # No break found

def detect_liquidity_sweep(df, target_swing_index, sweep_candle_index, is_check_high=True):
    """
    Checks if a specific candle swept liquidity beyond a specific swing point.
    Args:
        df (pd.DataFrame): OHLCV data.
        target_swing_index (int): Index location of the swing high/low being potentially swept.
        sweep_candle_index (int): Index location of the candle performing the potential sweep.
        is_check_high (bool): True if checking for a sweep above a high, False for below a low.
    Returns:
        bool: True if the candle swept liquidity beyond the target swing, False otherwise.
        NOTE: This checks wick vs close. Real sweeps might involve more complex logic.
    """
     # Validate inputs
    if not isinstance(df, pd.DataFrame) or not all(c in df.columns for c in ['High', 'Low', 'Close']):
        # log_info("detect_liquidity_sweep: Input DataFrame missing required columns.", "WARN") # Less verbose
        return False
    if target_swing_index < 0 or target_swing_index >= len(df) or \
       sweep_candle_index < 0 or sweep_candle_index >= len(df):
        # print(f"[WARN][MSA_SMC] detect_liquidity_sweep: Invalid index. Target: {target_swing_index}, Sweep: {sweep_candle_index}, Len: {len(df)}")
        return False

    try:
        if is_check_high:
            target_swing_price = df['High'].iloc[target_swing_index]
            sweep_candle_high = df['High'].iloc[sweep_candle_index]
            sweep_candle_close = df['Close'].iloc[sweep_candle_index]
            # Check for NaN values before comparison
            if pd.isna(target_swing_price) or pd.isna(sweep_candle_high) or pd.isna(sweep_candle_close):
                return False
            # Swept if wick went above target high, but candle closed below it
            return sweep_candle_high > target_swing_price and sweep_candle_close < target_swing_price
        else: # Check for sweep below low
            target_swing_price = df['Low'].iloc[target_swing_index]
            sweep_candle_low = df['Low'].iloc[sweep_candle_index]
            sweep_candle_close = df['Close'].iloc[sweep_candle_index]
            # Check for NaN values before comparison
            if pd.isna(target_swing_price) or pd.isna(sweep_candle_low) or pd.isna(sweep_candle_close):
                return False
            # Swept if wick went below target low, but candle closed above it
            return sweep_candle_low < target_swing_price and sweep_candle_close > target_swing_price
    except Exception as e:
        log_info(f"Error during liquidity sweep detection: {e}", "ERROR")
        return False


# --- Main Analysis Function ---

def analyze_market_structure(df, swing_n=5, bos_look_forward=20):
    """
    Analyzes market structure based on SMC principles (Strong/Weak Highs/Lows, Ranges).
    Args:
        df (pd.DataFrame): Input OHLCV data (e.g., D1, H4). Assumes DatetimeIndex.
        swing_n (int): Lookback/forward period for swing point identification.
        bos_look_forward (int): How far ahead to look for a BoS after a swing.
    Returns:
        dict: A dictionary containing the analysis results:
              'structure_points': List of classified swing points (timestamp, price, type).
              'valid_trading_range': Dictionary with 'start', 'end', 'type', 'source_tf', 'is_valid' or None.
              'htf_bias': 'Bullish', 'Bearish', or 'Uncertain'.
              'discount_premium': Dictionary with 'midpoint', 'discount_max', 'premium_min' or None.
              'erl_targets': List of weak highs/lows marked as targets.
    """
    # --- Input Validation & Initial Logging ---
    log_info("--- Running Market Structure Analysis (SMC) ---")
    if not isinstance(df, pd.DataFrame) or df.empty or not all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
        log_info("Invalid input DataFrame. Must have OHLC columns.", "ERROR")
        return { 'structure_points': [], 'valid_trading_range': None, 'htf_bias': 'Uncertain', 'discount_premium': None, 'erl_targets': [] }
    if not isinstance(df.index, pd.DatetimeIndex):
         log_info("DataFrame index must be a DatetimeIndex.", "ERROR")
         return { 'structure_points': [], 'valid_trading_range': None, 'htf_bias': 'Uncertain', 'discount_premium': None, 'erl_targets': [] }

    source_tf_str = 'Unknown'; df_tz = df.index.tz # Get timezone early
    if hasattr(df.index, 'freqstr') and df.index.freqstr:
        source_tf_str = df.index.freqstr.replace('1H', 'H1').replace('4H', 'H4').replace('1D', 'D1') # Normalize common frequencies

    # Log input parameters
    log_info(f"Input Config: Timeframe={source_tf_str}, Swing N={swing_n}, BoS Look Forward={bos_look_forward}")

    # --- Swing Point Identification ---
    log_info(f"Task: Identifying swing points (n={swing_n})...")
    df_swings = find_swing_highs_lows(df, n=swing_n)
    swing_highs = df_swings[df_swings['swing_high'].notna()]
    swing_lows = df_swings[df_swings['swing_low'].notna()]
    log_info(f"Result: Identified {len(swing_highs)} swing highs, {len(swing_lows)} swing lows.")

    all_swings = []
    # Combine and sort all swings with original index location
    for idx, row in swing_highs.iterrows():
        if idx in df_swings.index:
            try: loc = df_swings.index.get_loc(idx); all_swings.append({'timestamp': idx, 'price': row['swing_high'], 'type': 'High', 'index': loc})
            except KeyError: log_info(f"WARN: Could not get index location for swing high at {idx}", "WARN")
    for idx, row in swing_lows.iterrows():
         if idx in df_swings.index:
            try: loc = df_swings.index.get_loc(idx); all_swings.append({'timestamp': idx, 'price': row['swing_low'], 'type': 'Low', 'index': loc})
            except KeyError: log_info(f"WARN: Could not get index location for swing low at {idx}", "WARN")

    all_swings.sort(key=lambda x: x['index']) # Sort by index location

    if not all_swings:
        log_info("No swing points identified after processing.", "WARN")
        return { 'structure_points': [], 'valid_trading_range': None, 'htf_bias': 'Uncertain', 'discount_premium': None, 'erl_targets': [] }

    # --- Classify Swings (Strong/Weak) ---
    log_info("Task: Classifying swing points as Strong/Weak...")
    structure_points = [] # Store classified points
    # NOTE: Classification logic remains complex and may need strategy-specific refinement
    for i in range(len(all_swings)):
        current_swing = all_swings[i]
        point_type = "Undetermined" # Default

        # --- Check Strong High ---
        if current_swing['type'] == 'High':
            low_before = None; low_before_before = None; swept_prior_low = False; bos_after_high = False
            for j in range(i - 1, -1, -1): # Find low_before
                if all_swings[j]['type'] == 'Low': low_before = all_swings[j]; break
            if low_before: # Find low_before_before
                for j in range(low_before['index'] - 1, -1, -1):
                    if j < 0: break
                    potential_low = next((s for s in all_swings if s['index'] == j and s['type'] == 'Low'), None)
                    if potential_low: low_before_before = potential_low; break
            if low_before and low_before_before: # Check sweep
                 if detect_liquidity_sweep(df_swings, low_before_before['index'], low_before['index'], is_check_high=False): swept_prior_low = True
            if low_before: # Check BoS
                 bos_idx, _ = detect_bos(df_swings, low_before['index'], bos_look_forward, is_check_high=False)
                 if bos_idx is not None and bos_idx > current_swing['index']: bos_after_high = True
            if swept_prior_low and bos_after_high: point_type = "Strong High"
            # --- Check Weak High ---
            elif point_type == "Undetermined":
                 subsequent_low = None
                 for j in range(i + 1, len(all_swings)):
                      if all_swings[j]['type'] == 'Low': subsequent_low = all_swings[j]; break
                 if subsequent_low:
                      break_high_idx, _ = detect_bos(df_swings, current_swing['index'], bos_look_forward, is_check_high=True)
                      if break_high_idx is not None and break_high_idx > subsequent_low['index']: point_type = "Weak High"

        # --- Check Strong Low ---
        elif current_swing['type'] == 'Low':
             high_before = None; high_before_before = None; swept_prior_high = False; bos_after_low = False
             for j in range(i - 1, -1, -1): # Find high_before
                 if all_swings[j]['type'] == 'High': high_before = all_swings[j]; break
             if high_before: # Find high_before_before
                 for j in range(high_before['index'] - 1, -1, -1):
                     if j < 0: break
                     potential_high = next((s for s in all_swings if s['index'] == j and s['type'] == 'High'), None)
                     if potential_high: high_before_before = potential_high; break
             if high_before and high_before_before: # Check sweep
                  if detect_liquidity_sweep(df_swings, high_before_before['index'], high_before['index'], is_check_high=True): swept_prior_high = True
             if high_before: # Check BoS
                  bos_idx, _ = detect_bos(df_swings, high_before['index'], bos_look_forward, is_check_high=True)
                  if bos_idx is not None and bos_idx > current_swing['index']: bos_after_low = True
             if swept_prior_high and bos_after_low: point_type = "Strong Low"
             # --- Check Weak Low ---
             elif point_type == "Undetermined":
                  subsequent_high = None
                  for j in range(i + 1, len(all_swings)):
                       if all_swings[j]['type'] == 'High': subsequent_high = all_swings[j]; break
                  if subsequent_high:
                       break_low_idx, _ = detect_bos(df_swings, current_swing['index'], bos_look_forward, is_check_high=False)
                       if break_low_idx is not None and break_low_idx > subsequent_high['index']: point_type = "Weak Low"

        if point_type != "Undetermined":
            structure_points.append({
                'timestamp': current_swing['timestamp'], 'price': current_swing['price'],
                'type': point_type, 'index': current_swing['index']
            })

    structure_points.sort(key=lambda x: x['timestamp'])
    log_info(f"Result: Classified {len(structure_points)} Strong/Weak points.")

    # --- Determine Trading Range and Bias ---
    log_info("Task: Determining Valid Trading Range and HTF Bias...")
    valid_trading_range = None
    htf_bias = "Uncertain"
    discount_premium = None
    erl_targets = []
    range_confirmed = False # Flag for logging

    strong_points = [p for p in structure_points if 'Strong' in p['type']]
    if len(strong_points) >= 2:
        last_strong_point = strong_points[-1]
        prev_strong_point = strong_points[-2]

        # Check Uptrend Range (Prev=SL, Last=SH)
        if "Strong Low" in prev_strong_point['type'] and "Strong High" in last_strong_point['type']:
            preceding_swing_high = None
            for k_idx in range(last_strong_point['index'] - 1, prev_strong_point['index'], -1):
                 potential_swing = next((s for s in all_swings if s['index'] == k_idx and s['type'] == 'High'), None)
                 if potential_swing: preceding_swing_high = potential_swing; break
            if preceding_swing_high:
                bos_idx, bos_price = detect_bos(df_swings, preceding_swing_high['index'], bos_look_forward, is_check_high=True)
                if bos_idx is not None and bos_idx > last_strong_point['index']:
                    range_confirmed = True; htf_bias = "Bullish"
                    # --- Added Log ---
                    log_info(f"[Range] Confirmed {htf_bias} Range: {prev_strong_point['type']} at {prev_strong_point['price']:.5f} ({prev_strong_point['timestamp'].date()}) -> {last_strong_point['type']} at {last_strong_point['price']:.5f} ({last_strong_point['timestamp'].date()})")
                    log_info(f"[Range] Preceding swing high ({preceding_swing_high['timestamp'].date()} at {preceding_swing_high['price']:.5f}) broken post-range at index {bos_idx} ({df_swings.index[bos_idx].date()}) confirms BOS")
                    valid_trading_range = {
                        'start': {'timestamp': prev_strong_point['timestamp'], 'price': prev_strong_point['price'], 'type': prev_strong_point['type'], 'source_tf': source_tf_str},
                        'end': {'timestamp': last_strong_point['timestamp'], 'price': last_strong_point['price'], 'type': last_strong_point['type'], 'source_tf': source_tf_str},
                        'type': 'Uptrend', 'is_valid': True
                    }

        # Check Downtrend Range (Prev=SH, Last=SL)
        elif "Strong High" in prev_strong_point['type'] and "Strong Low" in last_strong_point['type']:
            preceding_swing_low = None
            for k_idx in range(last_strong_point['index'] - 1, prev_strong_point['index'], -1):
                 potential_swing = next((s for s in all_swings if s['index'] == k_idx and s['type'] == 'Low'), None)
                 if potential_swing: preceding_swing_low = potential_swing; break
            if preceding_swing_low:
                bos_idx, bos_price = detect_bos(df_swings, preceding_swing_low['index'], bos_look_forward, is_check_high=False)
                if bos_idx is not None and bos_idx > last_strong_point['index']:
                    range_confirmed = True; htf_bias = "Bearish"
                     # --- Added Log ---
                    log_info(f"[Range] Confirmed {htf_bias} Range: {prev_strong_point['type']} at {prev_strong_point['price']:.5f} ({prev_strong_point['timestamp'].date()}) -> {last_strong_point['type']} at {last_strong_point['price']:.5f} ({last_strong_point['timestamp'].date()})")
                    log_info(f"[Range] Preceding swing low ({preceding_swing_low['timestamp'].date()} at {preceding_swing_low['price']:.5f}) broken post-range at index {bos_idx} ({df_swings.index[bos_idx].date()}) confirms BOS")
                    valid_trading_range = {
                        'start': {'timestamp': prev_strong_point['timestamp'], 'price': prev_strong_point['price'], 'type': prev_strong_point['type'], 'source_tf': source_tf_str},
                        'end': {'timestamp': last_strong_point['timestamp'], 'price': last_strong_point['price'], 'type': last_strong_point['type'], 'source_tf': source_tf_str},
                        'type': 'Downtrend', 'is_valid': True
                    }

    if not range_confirmed:
        # --- Added Log ---
        log_info("[Range] No valid HTF range confirmed - BOS condition unmet or insufficient strong points.")

    # --- Calculate Discount/Premium ---
    if valid_trading_range:
        start_price = valid_trading_range.get('start', {}).get('price')
        end_price = valid_trading_range.get('end', {}).get('price')
        if isinstance(start_price, (int, float)) and isinstance(end_price, (int, float)) and pd.notna(start_price) and pd.notna(end_price):
            range_high = max(start_price, end_price); range_low = min(start_price, end_price)
            if abs(range_high - range_low) > 1e-9:
                 midpoint = range_low + (range_high - range_low) * 0.5
                 discount_premium = {'midpoint': midpoint, 'discount_max': midpoint, 'premium_min': midpoint }

    # --- Identify ERL Targets ---
    weak_points = [p for p in structure_points if 'Weak' in p['type']]
    if valid_trading_range:
        range_start_ts = valid_trading_range['start']['timestamp']; range_end_ts = valid_trading_range['end']['timestamp']
        if htf_bias == "Bullish":
             erl_targets = [{'timestamp': p['timestamp'], 'price': p['price'], 'type': p['type']} for p in weak_points if 'High' in p['type'] and isinstance(p.get('timestamp'), pd.Timestamp) and p['timestamp'] > range_end_ts]
             erl_targets.append({'timestamp': valid_trading_range['end']['timestamp'], 'price': valid_trading_range['end']['price'], 'type': 'Range High (Strong)'})
        elif htf_bias == "Bearish":
             erl_targets = [{'timestamp': p['timestamp'], 'price': p['price'], 'type': p['type']} for p in weak_points if 'Low' in p['type'] and isinstance(p.get('timestamp'), pd.Timestamp) and p['timestamp'] > range_end_ts]
             erl_targets.append({'timestamp': valid_trading_range['end']['timestamp'], 'price': valid_trading_range['end']['price'], 'type': 'Range Low (Strong)'})
    else: erl_targets = [{'timestamp': p['timestamp'], 'price': p['price'], 'type': p['type']} for p in weak_points]


    # --- Compile Final Output ---
    final_structure_points = []
    for p in structure_points:
        point_copy = p.copy(); point_copy.pop('index', None)
        final_structure_points.append(point_copy)

    # --- Added Final Summary Log ---
    log_info(f"--- Analysis Complete ---")
    log_info(f"Determined HTF Bias: {htf_bias}")
    if valid_trading_range:
        start_p = valid_trading_range.get('start',{}); end_p = valid_trading_range.get('end',{})
        log_info(f"Confirmed Trading Range: {valid_trading_range.get('type')} from {start_p.get('type')} ({start_p.get('timestamp')}) to {end_p.get('type')} ({end_p.get('timestamp')})")
    else:
        log_info("Confirmed Trading Range: None")

    results = {
        'structure_points': final_structure_points,
        'valid_trading_range': valid_trading_range,
        'htf_bias': htf_bias,
        'discount_premium': discount_premium,
        'erl_targets': erl_targets
    }
    return results


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Running Market Structure Analyzer Example ---")
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    price = 100; prices = []
    for i in range(200): price += np.random.normal(0, 1.5) + (0.05 if i % 40 < 20 else -0.05); prices.append(price)
    dummy_df = pd.DataFrame(index=dates)
    dummy_df['Open'] = prices
    dummy_df['High'] = dummy_df['Open'] + np.abs(np.random.normal(0, 3, 200)) + 1
    dummy_df['Low'] = dummy_df['Open'] - np.abs(np.random.normal(0, 3, 200)) - 1
    dummy_df['Close'] = dummy_df['Open'] + np.random.normal(0, 2, 200)
    dummy_df['High'] = dummy_df[['Open','High','Low','Close']].max(axis=1)
    dummy_df['Low'] = dummy_df[['Open','High','Low','Close']].min(axis=1)
    dummy_df['Volume'] = np.random.randint(100, 1000, size=200)

    print("Analyzing Dummy D1 data...")
    try:
        swing_param = 5; bos_param = 15
        structure_results = analyze_market_structure(dummy_df, swing_n=swing_param, bos_look_forward=bos_param)

        print("\n--- Analysis Results ---")
        print(f"HTF Bias: {structure_results.get('htf_bias')}")
        tr = structure_results.get('valid_trading_range')
        if tr and isinstance(tr.get('start'), dict) and isinstance(tr.get('end'), dict):
            print(f"Valid Trading Range ({tr.get('type')}):")
            start_ts = tr['start'].get('timestamp', pd.NaT); start_price = tr['start'].get('price', np.nan); start_type = tr['start'].get('type', 'N/A')
            end_ts = tr['end'].get('timestamp', pd.NaT); end_price = tr['end'].get('price', np.nan); end_type = tr['end'].get('type', 'N/A')
            start_date_str = start_ts.date() if pd.notna(start_ts) else 'N/A'; end_date_str = end_ts.date() if pd.notna(end_ts) else 'N/A'
            print(f"  Start: {start_date_str} at {start_price:.2f} ({start_type})")
            print(f"  End:   {end_date_str} at {end_price:.2f} ({end_type})")
        else: print("Valid Trading Range: None or invalid structure")
        dp = structure_results.get('discount_premium'); print(f"Discount/Premium: {'Midpoint=' + str(round(dp['midpoint'], 2)) if dp else 'None'}")
        print("\nStructure Points:"); [print(f"  - {(p.get('timestamp', pd.NaT).date() if pd.notna(p.get('timestamp')) else 'N/A')}: {p.get('type', 'N/A')} at {p.get('price', np.nan):.2f}") for p in structure_results.get('structure_points', [])]
        print("\nERL Targets:"); [print(f"  - {(t.get('timestamp', pd.NaT).date() if pd.notna(t.get('timestamp')) else 'N/A')}: {t.get('type', 'N/A')} at {t.get('price', np.nan):.2f}") for t in structure_results.get('erl_targets', [])]
    except Exception as e: print(f"\nError during analysis: {e}"); import traceback; traceback.print_exc()
    print("\n--- Example Complete ---")

