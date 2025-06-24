import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timezone, timedelta # Added timedelta
from pathlib import Path
import json
import sys
import os

import argparse

parser = argparse.ArgumentParser(description='Run full stack analysis on M1 CSV')
parser.add_argument('--file', type=str, required=True, help='Path to M1 CSV file')
args = parser.parse_args()

csv_file_path = args.file

# --- Add core directory to path if needed (adjust relative path if necessary) ---
# This allows importing modules from the 'core' subdirectory
script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
core_dir = script_dir / "core"
indicators_dir = core_dir / "indicators"
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))
if str(indicators_dir) not in sys.path:
     sys.path.insert(0, str(indicators_dir))
# --- End Path Addition ---


# --- Configuration ---
# Use the uploaded file path
output_dir = Path('analysis_output') # Where to save aggregated data and charts
output_dir.mkdir(parents=True, exist_ok=True) # Create output directory

target_timeframes = {
    "m5": "5min",
    "m15": "15min",
    "m30": "30min",
    "h1": "1H",
    "h4": "4H",
    # "h12": "12H", # Pandas doesn't have a direct 12H rule, needs custom handling if required
    "d1": "D",
    "w1": "W"
}
aggregation_rules = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}

# --- Import Enrichment Engines & Charting ---
# Assuming these are in the 'core' or 'core/indicators' directories
try: from indicator_enrichment_engine import calculate_standard_indicators
except ImportError: calculate_standard_indicators = None; print("WARN: Indicator Engine not found.")
try: from smc_enrichment_engine import tag_smc_zones
except ImportError: tag_smc_zones = None; print("WARN: SMC Enrichment Engine not found.")
try: from phase_detector_wyckoff_v1 import detect_wyckoff_phases_and_events # Corrected import path
except ImportError: detect_wyckoff_phases_and_events = None; print("WARN: Wyckoff Detector not found.")
try: from liquidity_sweep_detector import tag_liquidity_sweeps
except ImportError: tag_liquidity_sweeps = None; print("WARN: Liquidity Sweep Detector not found.")
# Import charting function from orchestrator
try:
    from copilot_orchestrator import generate_analysis_chart_json, load_strategy_profile
    print("Charting import successful")
except ImportError as import_err:
    generate_analysis_chart_json = None
    load_strategy_profile = None
    import traceback
    print("WARN: Charting/Profile functions from orchestrator not found.")
    print("DETAILS:", import_err)
    traceback.print_exc()

# --- Logging ---
def log_step(message):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] {message}")

# --- Main Simulation ---
if __name__ == "__main__":
    log_step(f"--- Starting Full Stack Simulation for {csv_file_path} ---")
    all_tf_data = {}
    enriched_tf_data = {}
    wyckoff_result = None

    # --- 1. Load and Preprocess MT5 Export ---
    log_step("Step 1: Loading and Preprocessing M1 Data...")
    try:
        # Load using tab separator
        df_m1_raw = pd.read_csv(csv_file_path, sep='\t', header=0)
        log_step(f" -> Loaded {len(df_m1_raw)} rows.")
        log_step(f" -> Initial Columns: {df_m1_raw.columns.tolist()}")

        # Rename columns based on common MT5 headers
        column_map = {
            '<DATE>': 'Date', '<TIME>': 'Time',
            '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
            '<TICKVOL>': 'TickVolume', '<VOL>': 'Volume', '<SPREAD>': 'Spread'
        }
        # Only rename columns that exist in the DataFrame
        cols_to_rename = {k: v for k, v in column_map.items() if k in df_m1_raw.columns}
        df_m1_raw.rename(columns=cols_to_rename, inplace=True)
        log_step(f" -> Columns after renaming: {df_m1_raw.columns.tolist()}")


        # Check for required Date and Time columns
        if 'Date' not in df_m1_raw.columns or 'Time' not in df_m1_raw.columns:
             raise ValueError("Missing required <DATE> or <TIME> columns in the CSV.")

        # Combine Date and Time into a single Datetime column
        log_step(" -> Combining Date and Time columns...")
        # Attempt common formats robustly
        combined_datetime = df_m1_raw['Date'] + ' ' + df_m1_raw['Time']
        df_m1_raw['Timestamp'] = pd.to_datetime(combined_datetime, errors='coerce', infer_datetime_format=True)

        # Check for parsing errors
        if df_m1_raw['Timestamp'].isnull().any():
            failed_count = df_m1_raw['Timestamp'].isnull().sum()
            log_step(f"WARN: Failed to parse datetime for {failed_count} rows. Attempting alternative format YYYY-MM-DD...")
            try: # Try YYYY-MM-DD format
                 df_m1_raw['Timestamp'] = pd.to_datetime(df_m1_raw['Date'] + ' ' + df_m1_raw['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                 if df_m1_raw['Timestamp'].isnull().any():
                      raise ValueError("Alternative datetime parsing also failed.")
            except Exception as e:
                 log_step(f"ERROR: Datetime parsing failed: {e}")
                 raise ValueError("Could not parse Date and Time columns.")

        # Drop rows where timestamp parsing failed
        original_len = len(df_m1_raw)
        df_m1_raw.dropna(subset=['Timestamp'], inplace=True)
        if len(df_m1_raw) < original_len:
             log_step(f" -> Dropped {original_len - len(df_m1_raw)} rows due to datetime parsing errors.")

        # Set Timestamp as index and localize to UTC
        df_m1 = df_m1_raw.set_index('Timestamp')
        try:
            df_m1 = df_m1.tz_localize('UTC')
            log_step(" -> Localized Timestamp index to UTC.")
        except TypeError: # Already localized
            df_m1 = df_m1.tz_convert('UTC') # Convert if different timezone
            log_step(" -> Index already localized, converted to UTC.")
        except Exception as tz_err:
             log_step(f"ERROR: Failed to localize/convert index timezone: {tz_err}")
             raise tz_err

        # Select and rename final OHLCV columns
        if 'Volume' not in df_m1.columns and 'TickVolume' in df_m1.columns:
            log_step(" -> Using TickVolume as Volume.")
            df_m1.rename(columns={'TickVolume': 'Volume'}, inplace=True)
        elif 'Volume' not in df_m1.columns:
             log_step("WARN: No Volume or TickVolume column found. Creating dummy Volume column with value 1.")
             df_m1['Volume'] = 1.0 # Add dummy volume if none exists

        # Ensure OHLCV columns are numeric, coercing errors
        final_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        log_step(f" -> Ensuring numeric types for {final_cols}...")
        for col in final_cols:
            if col in df_m1.columns:
                df_m1[col] = pd.to_numeric(df_m1[col], errors='coerce')
            else:
                 raise ValueError(f"Required column '{col}' not found after processing.")

        # Drop rows with NaN in OHLC columns
        initial_rows = len(df_m1)
        df_m1.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        if len(df_m1) < initial_rows:
            log_step(f" -> Dropped {initial_rows - len(df_m1)} rows with NaN OHLC values.")

        # Fill NaN volumes with 0
        nan_vol_count = df_m1['Volume'].isnull().sum()
        if nan_vol_count > 0:
            log_step(f" -> Filling {nan_vol_count} NaN Volume values with 0.")
            df_m1['Volume'].fillna(0, inplace=True)

        # Keep only standard columns and sort
        df_m1 = df_m1[final_cols]
        df_m1 = df_m1.sort_index() # Ensure chronological order

        log_step(f" -> Preprocessing complete. Final M1 DataFrame shape: {df_m1.shape}")
        all_tf_data['m1'] = df_m1

    except FileNotFoundError:
        log_step(f"ERROR: CSV file not found at {csv_file_path}")
        exit()
    except ValueError as ve:
        log_step(f"ERROR: Data processing error - {ve}")
        exit()
    except Exception as e:
        log_step(f"ERROR: An unexpected error occurred during loading/preprocessing: {e}")
        traceback.print_exc()
        exit()

    # --- 2. Aggregate to Higher Timeframes ---
    log_step("\n--- Step 2: Aggregating M1 data to Higher Timeframes ---")
    if not df_m1.empty:
        for tf_key, freq in target_timeframes.items():
            try:
                log_step(f" -> Aggregating to {tf_key.upper()} ({freq})...")
                df_agg = df_m1.resample(freq).agg(aggregation_rules)
                df_agg.dropna(how='all', subset=['Open', 'High', 'Low', 'Close'], inplace=True)
                df_agg['Volume'].fillna(0, inplace=True) # Fill NaN volumes that might appear from resampling gaps

                if not df_agg.empty:
                    all_tf_data[tf_key] = df_agg
                    log_step(f"   -> {tf_key.upper()} shape: {df_agg.shape}")
                else:
                    log_step(f"   -> {tf_key.upper()} resulted in empty DataFrame after aggregation/dropna.")
            except Exception as agg_err:
                log_step(f"ERROR: Failed to aggregate {tf_key.upper()}: {agg_err}")
                traceback.print_exc()
    else:
        log_step("Skipping aggregation as M1 data is empty.")

    # --- 3. Apply Enrichment ---
    log_step("\n--- Step 3: Applying Data Enrichment ---")
    # Define dummy configs for now, load from files in real implementation
    indicator_config = {}
    smc_config = {'swing_n': 2} # Example config for SMC engine
    liquidity_config = {'fractal_n': 2} # Example config for Liquidity engine
    wyckoff_config = {'pivot_lookback': 20, 'min_volume_surge_multiplier': 1.8} # Example config for Wyckoff

    for tf_key, df_tf in all_tf_data.items():
        if df_tf is None or df_tf.empty:
            enriched_tf_data[tf_key] = df_tf
            continue
        log_step(f" -> Enriching {tf_key.upper()} data ({len(df_tf)} rows)...")
        df_enriched = df_tf.copy()
        try:
            # Apply enrichments sequentially
            if calculate_standard_indicators:
                df_enriched = calculate_standard_indicators(df_enriched, tf=tf_key.upper(), config=indicator_config)
            if tag_smc_zones:
                df_enriched = tag_smc_zones(df_enriched, tf=tf_key.upper(), config=smc_config)
            if tag_liquidity_sweeps:
                 df_enriched = tag_liquidity_sweeps(df_enriched, tf=tf_key.upper(), config=liquidity_config)
            # Add other enrichment calls here

            enriched_tf_data[tf_key] = df_enriched
        except Exception as enrich_err:
            log_step(f"ERROR during enrichment for TF {tf_key}: {enrich_err}")
            traceback.print_exc()
            enriched_tf_data[tf_key] = df_tf # Store original if enrichment fails
    log_step(" -> Enrichment phase complete.")

    # --- 4. Run Wyckoff Analysis (Example on H1) ---
    log_step("\n--- Step 4: Running Wyckoff Analysis (Example on H1) ---")
    wyckoff_tf = 'h1' # Choose TF for Wyckoff
    if detect_wyckoff_phases_and_events and wyckoff_tf in enriched_tf_data and not enriched_tf_data[wyckoff_tf].empty:
        try:
            wyckoff_result = detect_wyckoff_phases_and_events(
                df=enriched_tf_data[wyckoff_tf],
                timeframe=wyckoff_tf.upper(),
                config=wyckoff_config
            )
            log_step(f" -> Wyckoff Analysis Result (H1): Current Phase = {wyckoff_result.get('current_phase')}")
            # Display detected events for verification
            if wyckoff_result.get('detected_events'):
                log_step(" -> Detected Wyckoff Events (H1):")
                sorted_events = sorted(wyckoff_result['detected_events'].items(), key=lambda item: item[1]['time'])
                for key, val in sorted_events:
                    try:
                        price_str = f"{float(val.get('price')):.2f}" if isinstance(val.get('price'), (int, float)) else str(val.get('price'))
                        volume_val = val.get('volume', 'N/A')
                        volume_str = f"{int(volume_val):<8}" if isinstance(volume_val, (int, float)) else str(volume_val)
                        print(f"     - {key:<10}: Price={price_str:<8} Time={val.get('time')} Vol={volume_str}")
                    except Exception as event_fmt_err:
                        print(f"[WARN] Could not format Wyckoff event {key}: {event_fmt_err}")
        except Exception as wy_err:
            log_step(f"ERROR during Wyckoff analysis: {wy_err}")
            traceback.print_exc()
            wyckoff_result = {"error": f"Wyckoff analysis failed: {wy_err}"}
    else:
        log_step(f"Skipping Wyckoff analysis: Module or {wyckoff_tf.upper()} data not available/empty.")

    # --- 5. Simulate Analysis Routing & Charting (Using M15) ---
    log_step("\n--- Step 5: Generating M15 Chart ---")
    chart_tf = 'm15'
    if chart_tf in enriched_tf_data and not enriched_tf_data[chart_tf].empty:
        try:
            if not generate_analysis_chart_json:
                 log_step("WARN: Charting function not available. Skipping chart generation.")
            else:
                log_step(f" -> Generating chart for {chart_tf.upper()}...")
                # Create dummy structure/phase results for charting demonstration
                dummy_structure = {"htf_bias": "Unknown (Simulated)"}
                dummy_phase = {"phase": "Unknown (Simulated)"}

                chart_json = generate_analysis_chart_json(
                     price_df=enriched_tf_data[chart_tf].tail(300), # Plot last 300 bars
                     chart_tf=chart_tf.upper(),
                     pair="XAUUSD",
                     target_time=datetime.now(timezone.utc).isoformat(),
                     structure_data=dummy_structure,
                     phase_result=dummy_phase,
                     wyckoff_result=wyckoff_result, # Pass the actual Wyckoff result
                     variant_name="Full_Stack_Test",
                     # Add other results (confirmation, entry) as None for now
                     confirmation_result=None,
                     entry_result=None
                )

                if chart_json:
                    log_step(" -> Chart JSON generated successfully.")
                    # Save chart JSON to a file for inspection
                    chart_file = output_dir / f"XAUUSD_{chart_tf.upper()}_analysis_chart.json"
                    with open(chart_file, 'w', encoding='utf-8') as f:
                        f.write(chart_json)
                    log_step(f" -> Chart JSON saved to {chart_file}")
                else:
                    log_step(" -> Chart generation failed.")

        except Exception as chart_err:
            log_step(f"ERROR during chart generation: {chart_err}")
            traceback.print_exc()
    else:
        log_step(f"Cannot generate chart: Enriched data for {chart_tf.upper()} is missing or empty.")

    # --- Save enriched data (optional) ---
    # for tf_key, df_enriched in enriched_tf_data.items():
    #     if df_enriched is not None and not df_enriched.empty:
    #          save_path = output_dir / f"XAUUSD_{tf_key.upper()}_enriched.csv"
    #          df_enriched.to_csv(save_path)
    #          log_step(f"Saved enriched {tf_key.upper()} data to {save_path}")

    log_step("\n--- Full Stack Analysis Simulation Complete ---")
