# mentfx_stoploss_model_v2_adaptive.py
# Zanzibar-compatible SL placement engine for Mentfx-style entries
# V2: Incorporates adaptive buffers based on volatility_engine

import pandas as pd
import numpy as np
from datetime import datetime

# Assume volatility_engine.py is in the same directory or accessible via PYTHONPATH
# If not, adjust import path accordingly
try:
    # Import specific functions we might use
    from volatility_engine import compute_tick_volatility, compute_spread
    VOLATILITY_ENGINE_AVAILABLE = True
    print("Info: volatility_engine.py found and imported.")
except ImportError:
    print("Warning: volatility_engine.py not found. Adaptive buffers disabled. Using fixed buffers only.")
    VOLATILITY_ENGINE_AVAILABLE = False
    # Define dummy functions if import fails, to prevent runtime errors later
    # These will just return NaN, ensuring adaptive logic is skipped gracefully
    def compute_tick_volatility(df, mid_col='mid', window=20): return pd.Series(np.nan, index=df.index)
    def compute_spread(df, bid_col='bid', ask_col='ask'): return pd.Series(np.nan, index=df.index)


def compute_mentfx_stop_loss_adaptive(
    tick_df: pd.DataFrame,
    entry_time: datetime,
    entry_price: float,
    trade_type: str,
    min_buffer_base: float = 0.3, # Base minimum structural buffer
    max_buffer: float = 2.5, # Maximum allowed SL distance from entry
    spread_buffer_base: float = 0.1, # Base spread buffer component
    volatility_window: int = 20, # Lookback window for tick volatility calculation near entry
    volatility_factor: float = 0.5, # How much tick volatility scales the min_buffer (0 = no scaling)
    use_adaptive_buffer: bool = True # Master flag to enable/disable adaptive logic
    ):
    """
    Computes Mentfx-style Stop Loss from recent tick structure, optionally using
    adaptive buffers based on near-entry tick volatility.

    Assumes the entry signal has already been confirmed.

    Parameters:
        tick_df (pd.DataFrame): DataFrame containing tick data.
                                Must include 'datetime' (or be DatetimeIndex), 'bid', 'ask'.
        entry_time (datetime): Timestamp of the intended trade entry.
        entry_price (float): Actual or intended entry price (use Ask for buys, Bid for sells).
        trade_type (str): 'buy' or 'sell'.
        min_buffer_base (float): Base structural buffer added below low (for buy) or above high (for sell).
        max_buffer (float): Maximum allowed stop distance in price units from the entry_price.
        spread_buffer_base (float): Base buffer component added to account for spread.
        volatility_window (int): Number of ticks to look back for calculating volatility near entry.
        volatility_factor (float): Multiplier applied to calculated tick volatility to adjust the min_buffer.
                                   Set to 0 to disable volatility scaling even if use_adaptive_buffer is True.
        use_adaptive_buffer (bool): If True and volatility_engine is available, attempts to use
                                    tick volatility to adjust the min_buffer.

    Returns:
        dict: A dictionary containing detailed stop loss information:
              - entry_time: Actual timestamp used for calculation (nearest to input).
              - entry_price: Input entry price.
              - trade_type: Input trade type.
              - structure_extreme_price: The bid low (for buy) or ask high (for sell) found in the structure window.
              - initial_sl_candidate: SL calculated before applying max_buffer constraint.
              - computed_stop_loss: Final SL price after applying max_buffer.
              - stop_distance: Distance in price units from entry_price to computed_stop_loss.
              - max_buffer_applied: Boolean indicating if the max_buffer constraint was hit.
              - survived_tick_path: Boolean indicating if the computed_stop_loss was breached by subsequent ticks.
              - stop_hit_time: Timestamp when the stop was first hit (if survived_tick_path is False).
              - params: Dictionary of parameters used, including calculated volatility and final buffers.
    """

    # --- Input Validation & Preparation ---
    if tick_df.empty:
        raise ValueError("Input tick_df is empty.")
    if not all(col in tick_df.columns for col in ['bid', 'ask']):
         raise ValueError("tick_df must contain 'bid' and 'ask' columns.")

    df = tick_df.copy()
    # Ensure datetime is the index or a column that can be compared
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime' in df.columns:
            # Convert if needed, assuming UTC if no timezone info
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                 df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df = df.set_index('datetime')
        else:
            raise ValueError("tick_df must have a 'datetime' column or DatetimeIndex")
    # Ensure timezone consistency (convert entry_time if needed)
    if entry_time.tzinfo is None and df.index.tz is not None:
        entry_time = entry_time.replace(tzinfo=df.index.tz) # Assume same timezone
    elif entry_time.tzinfo is not None and df.index.tz is not None and entry_time.tzinfo != df.index.tz:
        entry_time = entry_time.tz_convert(df.index.tz) # Convert to DataFrame timezone

    df['mid'] = (df['bid'] + df['ask']) / 2
    df.sort_index(inplace=True) # Ensure data is time-ordered

    # Find nearest index location in the DataFrame to the requested entry_time
    try:
        nearest_idx_loc = df.index.get_indexer([entry_time], method='nearest')[0]
        actual_entry_time = df.index[nearest_idx_loc]
    except IndexError:
        raise ValueError(f"Could not find data near entry_time: {entry_time}")


    # Define structure window (e.g., 10 ticks *before* the identified entry tick)
    structure_window_ticks = 10
    structure_window_end_loc = nearest_idx_loc # End of window is the entry tick itself
    structure_window_start_loc = max(0, structure_window_end_loc - structure_window_ticks)
    # Slice using iloc for location-based indexing
    structure_df = df.iloc[structure_window_start_loc:structure_window_end_loc]

    # Define post-entry window for survival check (from entry tick onwards)
    post_entry_df = df.iloc[nearest_idx_loc:]

    # --- Calculate Volatility around Entry ---
    adaptive_min_buffer = min_buffer_base
    current_tick_volatility = np.nan
    current_spread = np.nan # Placeholder for potential future use

    # Check master flag and if volatility engine was successfully imported
    attempt_adaptive = use_adaptive_buffer and VOLATILITY_ENGINE_AVAILABLE and volatility_factor != 0

    if attempt_adaptive and not structure_df.empty:
        try:
            # Calculate tick volatility in the structure window just before entry
            # Use .copy() to avoid potential SettingWithCopyWarning if compute_tick_volatility modifies df
            volatility_series = compute_tick_volatility(structure_df.copy(), mid_col='mid', window=volatility_window)

            if not volatility_series.empty and pd.notna(volatility_series.iloc[-1]):
                 current_tick_volatility = volatility_series.iloc[-1]
                 # Scale the min_buffer based on volatility (simple linear scaling example)
                 # Ensure buffer doesn't go negative if base is small and factor is large negative (unlikely)
                 adaptive_min_buffer = max(0.01, min_buffer_base + (current_tick_volatility * volatility_factor))
                 print(f"Info: Adaptive buffer applied. Vol={current_tick_volatility:.4f}, Factor={volatility_factor}, Base={min_buffer_base:.3f}, Adapted={adaptive_min_buffer:.3f}")

            # Example: Calculate spread near entry (could be used to adapt spread_buffer)
            # spread_series = compute_spread(structure_df, bid_col='bid', ask_col='ask')
            # if not spread_series.empty and pd.notna(spread_series.iloc[-1]):
            #      current_spread = spread_series.iloc[-1]

        except Exception as e:
            print(f"Warning: Error calculating adaptive metrics: {e}. Using base buffers for this calculation.")
            current_tick_volatility = np.nan # Ensure reset on error
            attempt_adaptive = False # Disable adaptive for safety if calculation fails

    # Use adaptive buffers if calculated successfully, otherwise use base
    final_min_buffer = adaptive_min_buffer if attempt_adaptive and pd.notna(current_tick_volatility) else min_buffer_base
    final_spread_buffer = spread_buffer_base # Using base spread buffer for now

    # --- Calculate Stop Loss Candidate based on Structure Window ---
    sl_candidate = np.nan
    structure_extreme = np.nan

    if structure_df.empty:
        print("Warning: Structure window is empty. Falling back to entry_price based SL.")
        # Fallback: Place SL based on entry price and buffers if no preceding structure data
        sl_candidate = entry_price - final_min_buffer - final_spread_buffer if trade_type == "buy" else entry_price + final_min_buffer + final_spread_buffer
    else:
        if trade_type == "buy":
            structure_extreme = structure_df["bid"].min() # Lowest low (bid) in window
            if pd.notna(structure_extreme):
                 sl_candidate = structure_extreme - final_min_buffer - final_spread_buffer
            else: # Handle case where min() returns NaN (e.g., all NaNs in column)
                 print("Warning: Could not determine structure low (bid). Falling back.")
                 sl_candidate = entry_price - final_min_buffer - final_spread_buffer
        else: # sell
            structure_extreme = structure_df["ask"].max() # Highest high (ask) in window
            if pd.notna(structure_extreme):
                 sl_candidate = structure_extreme + final_min_buffer + final_spread_buffer
            else: # Handle case where max() returns NaN
                 print("Warning: Could not determine structure high (ask). Falling back.")
                 sl_candidate = entry_price + final_min_buffer + final_spread_buffer

    # --- Clamp Stop Loss to Max Buffer distance from Entry Price ---
    initial_sl_candidate = sl_candidate # Store pre-clamp value
    max_buffer_applied = False
    stop_distance = np.inf # Default to infinity if SL is NaN

    if pd.notna(sl_candidate):
        stop_distance = abs(entry_price - sl_candidate)
        if stop_distance > max_buffer:
            print(f"Info: Initial SL ({initial_sl_candidate:.3f}) distance {stop_distance:.3f} exceeds max_buffer ({max_buffer}). Clamping SL.")
            sl_candidate = entry_price - max_buffer if trade_type == "buy" else entry_price + max_buffer
            stop_distance = max_buffer # Update clamped distance
            max_buffer_applied = True
    else:
        print("Warning: SL candidate is NaN before clamping.")


    # --- Check Survivability using Post-Entry Ticks ---
    survived = True # Assume survived unless proven otherwise
    stop_hit_time = pd.NaT

    if post_entry_df.empty:
        print("Warning: No post-entry data available for survival check.")
        survived = None # Indicate unknown survival status
    elif pd.isna(sl_candidate):
        print("Warning: Cannot check survival, computed_stop_loss is NaN.")
        survived = None
    else:
        # Check if any tick after entry breached the calculated SL
        if trade_type == "buy":
            # For buys, check if the BID price dropped below the SL
            hit_mask = post_entry_df["bid"] < sl_candidate
            if hit_mask.any():
                survived = False
                stop_hit_time = post_entry_df[hit_mask].index.min() # Find the first time it was hit
        else: # sell
            # For sells, check if the ASK price rose above the SL
            hit_mask = post_entry_df["ask"] > sl_candidate
            if hit_mask.any():
                survived = False
                stop_hit_time = post_entry_df[hit_mask].index.min() # Find the first time it was hit

    # --- Format and Return Output Dictionary ---
    output = {
        "entry_time": str(actual_entry_time),
        "entry_price": round(entry_price, 3), # Using 3 decimal places for precision
        "trade_type": trade_type,
        "structure_extreme_price": round(structure_extreme, 3) if pd.notna(structure_extreme) else None,
        "initial_sl_candidate": round(initial_sl_candidate, 3) if pd.notna(initial_sl_candidate) else None,
        "computed_stop_loss": round(sl_candidate, 3) if pd.notna(sl_candidate) else None,
        "stop_distance": round(stop_distance, 3) if pd.notna(stop_distance) and stop_distance != np.inf else None,
        "max_buffer_applied": max_buffer_applied,
        "survived_tick_path": survived,
        "stop_hit_time": str(stop_hit_time) if pd.notna(stop_hit_time) else None,
        "params": {
             "use_adaptive_buffer": use_adaptive_buffer,
             "min_buffer_base": min_buffer_base,
             "spread_buffer_base": spread_buffer_base,
             "max_buffer": max_buffer,
             "volatility_window": volatility_window,
             "volatility_factor": volatility_factor,
             "current_tick_volatility": round(current_tick_volatility, 5) if pd.notna(current_tick_volatility) else None,
             "final_min_buffer_used": round(final_min_buffer, 5) if pd.notna(final_min_buffer) else None,
             "final_spread_buffer_used": round(final_spread_buffer, 5) if pd.notna(final_spread_buffer) else None,
             "structure_window_ticks": structure_window_ticks
        }
    }
    return output

# --- Example Usage Placeholder ---
# This part would be in your main script (e.g., orchestrator or backtester)

# Example:
# try:
#     # Load tick data (ensure it has 'datetime', 'bid', 'ask')
#     # loaded_tick_data = load_my_tick_data(...)
#     # Define entry parameters
#     # trade_entry_timestamp = pd.Timestamp('2025-04-17 14:30:05', tz='UTC')
#     # trade_entry_px = 2350.50 # Example entry price (Ask for buy)
#
#     sl_info = compute_mentfx_stop_loss_adaptive(
#         tick_df=loaded_tick_data,
#         entry_time=trade_entry_timestamp,
#         entry_price=trade_entry_px,
#         trade_type='buy',
#         use_adaptive_buffer=True, # Enable adaptive logic
#         min_buffer_base=0.3,      # Base buffer in price units
#         max_buffer=2.0,           # Max SL distance allowed
#         spread_buffer_base=0.1,   # Base spread component
#         volatility_window=15,     # Lookback for tick volatility
#         volatility_factor=0.6     # Sensitivity to volatility
#     )
#     print("\n--- Adaptive SL Calculation Result ---")
#     import json
#     print(json.dumps(sl_info, indent=2))
#     print("------------------------------------")
#
# except ValueError as ve:
#      print(f"Error during SL calculation: {ve}")
# except Exception as e:
#      print(f"An unexpected error occurred: {e}")

