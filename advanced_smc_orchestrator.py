import pandas as pd
from typing import Dict, Optional, List
import json
from pathlib import Path
import traceback # For detailed error logging

# Version: 5.2.0 (Fully Prop Desk Compliant)

# --- Import necessary analysis modules ---
# Core analysis modules
try: from market_structure_analyzer_smc import analyze_market_structure
except ImportError: analyze_market_structure = None; print("[WARN][AdvSMCOrch] market_structure_analyzer_smc not found.")
try: from liquidity_engine_smc import detect_inducement_from_structure # Or other relevant liquidity functions
except ImportError: detect_inducement_from_structure = None; print("[WARN][AdvSMCOrch] liquidity_engine_smc not found.")
try: from poi_manager_smc import find_and_validate_smc_pois # Or function to get relevant POIs
except ImportError: find_and_validate_smc_pois = None; print("[WARN][AdvSMCOrch] poi_manager_smc not found.")
try: from confluence_engine import ConfluenceEngine # If needed for specific variants
except ImportError: ConfluenceEngine = None; print("[WARN][AdvSMCOrch] confluence_engine not found.")

# --- Import the newly drafted modules ---
try: from impulse_correction_detector import detect_impulse_correction_phase
except ImportError: detect_impulse_correction_phase = None; print("[ERROR][AdvSMCOrch] impulse_correction_detector.py not found!")
try: from confirmation_engine_smc import confirm_smc_entry
except ImportError: confirm_smc_entry = None; print("[ERROR][AdvSMCOrch] confirmation_engine_smc.py not found!")
try: from entry_executor_smc import execute_smc_entry
except ImportError: execute_smc_entry = None; print("[ERROR][AdvSMCOrch] entry_executor_smc.py not found!")

# --- Import the POI Tap Checker --- ### MODIFICATION START ###
try: from poi_hit_watcher_smc import check_poi_tap_smc
except ImportError: check_poi_tap_smc = None; print("[ERROR][AdvSMCOrch] poi_hit_watcher_smc.py not found!")
# --- MODIFICATION END --- ###

# --- Signal dispatching ---
def dispatch_pine_payload(entry_result: Dict):
    """ Sends the trade signal (actual implementation expected in prod). """
    if entry_result and entry_result.get("pine_payload"):
        payload = entry_result["pine_payload"]
        print(f"[INFO][AdvSMCOrch] === DISPATCHING TRADE SIGNAL ===")
        print(json.dumps(payload, indent=2))
        # TODO: Implement actual dispatch logic
        print(f"[INFO][AdvSMCOrch] === SIGNAL DISPATCH COMPLETE ===")
    else:
        print("[WARN][AdvSMCOrch] No valid Pine payload found in entry result to dispatch.")

# --- Function to load strategy profiles ---
def load_strategy_profile(variant_name: str) -> Dict:
    """ Loads specific configuration for a strategy variant from strategy_profiles.json. Fallbacks to default if not found. """
    config_path = Path("strategy_profiles.json")
    if not config_path.exists():
        print(f"[ERROR][AdvSMCOrch] strategy_profiles.json not found at {config_path.resolve()}.")
        return {}
    try:
        with open(config_path, "r") as f:
            all_profiles = json.load(f)
        profile = None
        if variant_name in all_profiles:
            profile = all_profiles[variant_name]
            print(f"[INFO][AdvSMCOrch] Loaded strategy profile for variant: {variant_name}")
        elif "default" in all_profiles:
            print(f"[WARN][AdvSMCOrch] Strategy variant '{variant_name}' not found in profiles. Falling back to 'default'.")
            profile = all_profiles["default"]
        else:
            print(f"[WARN][AdvSMCOrch] Strategy variant '{variant_name}' not found and no 'default' profile present. Using empty config.")
            profile = {}
        return profile
    except Exception as e:
        print(f"[ERROR][AdvSMCOrch] Failed to load or parse strategy_profiles.json: {e}")
        return {}

# --- Main Orchestration Function ---
def run_advanced_smc_strategy(
    all_tf_data: Dict[str, pd.DataFrame],
    strategy_variant: str,
    target_timestamp: pd.Timestamp, # Note: This timestamp might be less relevant now, focus is on latest data + POI taps
    symbol: str = "XAUUSD"
) -> Dict:
    """
    Orchestrates the Advanced SMC strategy flow using specialized modules,
    incorporating POI tap checks and data slicing.
    """
    print(f"\n--- Running Advanced SMC Orchestrator for Variant: {strategy_variant} ---")
    orchestration_result = {
        "variant": strategy_variant,
        "symbol": symbol,
        "target_timestamp": target_timestamp.strftime('%Y-%m-%d %H:%M:%S'), # Keep for reference
        "steps": [],
        "final_entry_result": None,
        "error": None
    }

    # --- 1. Load Strategy Variant Configuration ---
    variant_config = load_strategy_profile(strategy_variant)
    phase_config = variant_config.get("phase_detection_config", {})
    confirmation_config = variant_config.get("confirmation_config", {})
    execution_config = variant_config.get("execution_config", {})
    risk_config = variant_config.get("risk_model_config", {})
    poi_tap_config = variant_config.get("poi_tap_config", {}) # Config for POI tap checker
    htf_structure_tf = variant_config.get("structure_timeframe", "h4")
    poi_tf = variant_config.get("poi_timeframe", "h1") # TF where POIs are identified
    poi_tap_check_tf = variant_config.get("poi_tap_check_timeframe", "m15") # TF to check for POI tap
    confirmation_tf = variant_config.get("confirmation_timeframe", "m5")
    execution_tf = variant_config.get("execution_timeframe", "m1")

    required_tfs = {htf_structure_tf, poi_tf, poi_tap_check_tf, confirmation_tf, execution_tf}
    for tf in required_tfs:
        if tf not in all_tf_data or all_tf_data[tf].empty:
            error_msg = f"Missing or empty data for required timeframe: {tf}"
            print(f"[ERROR][AdvSMCOrch] {error_msg}")
            orchestration_result["error"] = error_msg
            return orchestration_result

    # --- 2. Perform Initial HTF Analysis (Structure, POI) ---
    print(f"[INFO][AdvSMCOrch] Performing HTF Analysis (Structure TF: {htf_structure_tf}, POI TF: {poi_tf})...")
    structure_data = None
    if analyze_market_structure:
        try:
            structure_data = analyze_market_structure(all_tf_data[htf_structure_tf])
            orchestration_result["steps"].append({"step": "HTF Structure Analysis", "status": "Success", "bias": structure_data.get('htf_bias')})
        except Exception as e:
             print(f"[ERROR][AdvSMCOrch] Structure Analysis failed: {e}")
             orchestration_result["steps"].append({"step": "HTF Structure Analysis", "status": "Failed", "error": str(e)})
    else:
         orchestration_result["steps"].append({"step": "HTF Structure Analysis", "status": "Skipped", "reason": "Module not available"})

    # --- Get Relevant HTF POIs ---
    relevant_htf_pois = []
    if find_and_validate_smc_pois and structure_data:
         try:
             # *** Stub POI for testing flow ***
             htf_poi_type = structure_data.get('htf_bias', 'Unknown') if structure_data else 'Unknown'
             # Make stub POI relative to latest price for better testing chance
             latest_price = all_tf_data[poi_tap_check_tf]['Close'].iloc[-1]
             if htf_poi_type == 'Bullish':
                  poi_low = latest_price * 0.999
                  poi_high = latest_price * 0.9995
                  relevant_htf_pois.append({'range': [poi_low, poi_high], 'type': 'Bullish', 'timestamp': pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=4), 'source_tf': poi_tf})
             elif htf_poi_type == 'Bearish':
                  poi_low = latest_price * 1.0005
                  poi_high = latest_price * 1.001
                  relevant_htf_pois.append({'range': [poi_low, poi_high], 'type': 'Bearish', 'timestamp': pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=4), 'source_tf': poi_tf})
             print(f"[INFO][AdvSMCOrch] Found {len(relevant_htf_pois)} relevant HTF POI(s) (Stubbed near {latest_price:.5f}).")
             orchestration_result["steps"].append({"step": "HTF POI Identification", "status": "Success (Stubbed)", "count": len(relevant_htf_pois)})
         except Exception as e:
             print(f"[ERROR][AdvSMCOrch] POI Identification failed: {e}")
             orchestration_result["steps"].append({"step": "HTF POI Identification", "status": "Failed", "error": str(e)})
    else:
        orchestration_result["steps"].append({"step": "HTF POI Identification", "status": "Skipped", "reason": "Module or structure data not available"})

    # --- 3. Check Price Interaction with POI(s) ---
    entry_found = False
    if not relevant_htf_pois:
        print(f"[INFO][AdvSMCOrch] No POIs tapped or matched.")
        orchestration_result["steps"].append({"step": "POI Tap", "status": "No POIs tapped or matched"})

    for htf_poi in relevant_htf_pois:
        if entry_found: break

        print(f"\n[INFO][AdvSMCOrch] Evaluating HTF POI: Type={htf_poi['type']}, Range={htf_poi['range']}")
        orchestration_result["steps"].append({"step": "POI Evaluation Start", "poi_range": htf_poi['range'], "poi_type": htf_poi['type']})

        # --- Call POI Tap Checker ---
        poi_tap_result = None
        tap_time = None
        if check_poi_tap_smc:
            print(f"[INFO][AdvSMCOrch] Checking POI tap (Check TF: {poi_tap_check_tf})...")
            try:
                poi_tap_result = check_poi_tap_smc(
                    price_data=all_tf_data[poi_tap_check_tf], # Use specific TF for tap check
                    poi_range=htf_poi['range'],
                    config=poi_tap_config # Pass tap-specific config
                )
                orchestration_result["steps"].append({"step": "POI Tap Check", "result": poi_tap_result})
                if not poi_tap_result or not poi_tap_result.get("is_tapped"):
                    print("[INFO][AdvSMCOrch] POI was not tapped. Skipping.")
                    continue # Move to next POI if not tapped
                else:
                    tap_time = poi_tap_result.get("tap_time") # Store the tap time
                    print(f"[INFO][AdvSMCOrch] POI Tap CONFIRMED at {tap_time}.")
                    # --- Bias Injection Logging (future module connection) ---
                    bias_injection_applicable = variant_config.get("enable_bias_injection", False)
                    if bias_injection_applicable:
                        print("[INFO][AdvSMCOrch] Bias injection module is enabled and will be applied if connected.")
                    else:
                        print("[INFO][AdvSMCOrch] Bias injection not applicable for this variant or not enabled.")
            except Exception as e:
                print(f"[ERROR][AdvSMCOrch] POI Tap Check failed: {e}")
                orchestration_result["steps"].append({"step": "POI Tap Check", "status": "Failed", "error": str(e)})
                continue # Move to next POI if check fails
        else:
            print("[WARN][AdvSMCOrch] POI Tap Check skipped: Module not available.")
            orchestration_result["steps"].append({"step": "POI Tap Check", "status": "Skipped", "reason": "Module not available"})
            continue


        # --- 4. Check Impulse/Correction Phase (if required) ---
        phase_check_passed = True
        if detect_impulse_correction_phase:
            require_phase = variant_config.get("require_phase", None)
            if require_phase:
                print(f"[INFO][AdvSMCOrch] Checking market phase (Confirmation TF: {confirmation_tf}). Required: {require_phase}")
                try:
                    # Check phase on data *up to the tap time*? Or slightly after? Let's use up to tap time.
                    phase_check_data = all_tf_data[confirmation_tf][all_tf_data[confirmation_tf].index <= tap_time]
                    if phase_check_data.empty: raise ValueError("No data available up to tap time for phase check.")

                    phase_result = detect_impulse_correction_phase(
                        price_data=phase_check_data,
                        structure_data=structure_data,
                        config=phase_config
                    )
                    orchestration_result["steps"].append({"step": "Phase Detection", "result": phase_result})
                    current_phase = phase_result.get("phase", "Unknown")
                    # Check if phase matches requirement OR if valid_for_entry is true? Let's use require_phase match.
                    if current_phase == require_phase: # and phase_result.get("valid_for_entry", False):
                        phase_check_passed = True
                        print(f"[INFO][AdvSMCOrch] Phase check passed. Current phase: {current_phase}")
                    else:
                        phase_check_passed = False
                        print(f"[INFO][AdvSMCOrch] Phase check failed. Current phase: {current_phase}, Required: {require_phase}. Skipping POI.")
                except Exception as e:
                    phase_check_passed = False
                    print(f"[ERROR][AdvSMCOrch] Phase Detection failed: {e}")
                    orchestration_result["steps"].append({"step": "Phase Detection", "status": "Failed", "error": str(e)})
            else:
                 orchestration_result["steps"].append({"step": "Phase Detection", "status": "Skipped", "reason": "Not required by variant"})
        else:
             orchestration_result["steps"].append({"step": "Phase Detection", "status": "Skipped", "reason": "Module not available"})

        if not phase_check_passed:
            continue


        # --- 5. Run Confirmation Engine (using sliced data) --- ### MODIFICATION START ###
        confirmation_result = None
        ltf_poi_timestamp = None # Timestamp when the LTF POI for entry was formed
        if confirm_smc_entry and tap_time:
             print(f"[INFO][AdvSMCOrch] Running Confirmation Engine (Conf TF: {confirmation_tf}, Data After: {tap_time})...")
             try:
                 # Slice confirmation data: Only candles STRICTLY AFTER the tap time
                 confirmation_data_slice = all_tf_data[confirmation_tf][all_tf_data[confirmation_tf].index > tap_time]

                 if confirmation_data_slice.empty or len(confirmation_data_slice) < confirmation_config.get('confirmation_lookback', 30): # Check if enough data after tap
                      print("[WARN][AdvSMCOrch] Not enough data after POI tap for confirmation lookback. Skipping.")
                      orchestration_result["steps"].append({"step": "Confirmation Engine", "status": "Skipped", "reason": "Insufficient data after tap"})
                      continue
                 else:
                      print(f"[DEBUG][AdvSMCOrch] Confirmation using {len(confirmation_data_slice)} candles from {confirmation_data_slice.index.min()} onwards.")
                      confirmation_result = confirm_smc_entry(
                         htf_poi=htf_poi,
                         ltf_data=confirmation_data_slice, # Pass the SLICED data
                         strategy_variant=strategy_variant,
                         config=confirmation_config
                         # Pass confluence/liquidity data if available and needed
                      )
                      orchestration_result["steps"].append({"step": "Confirmation Engine", "result": confirmation_result})

                      if not confirmation_result or not confirmation_result.get("confirmation_status"):
                         print("[INFO][AdvSMCOrch] Confirmation failed or pattern not found.")
                         continue
                      else:
                         print("[INFO][AdvSMCOrch] Confirmation SUCCESS.")
                         # Store the timestamp of the identified LTF POI for the next slice
                         if confirmation_result.get("ltf_poi_timestamp"):
                              try:
                                   ltf_poi_timestamp = pd.Timestamp(confirmation_result["ltf_poi_timestamp"], tz='UTC')
                              except Exception as ts_parse_err:
                                   print(f"[WARN][AdvSMCOrch] Could not parse LTF POI timestamp: {ts_parse_err}")
                                   ltf_poi_timestamp = None # Ensure it's None if parsing fails


             except Exception as e:
                 print(f"[ERROR][AdvSMCOrch] Confirmation Engine failed: {e}\n{traceback.format_exc()}")
                 orchestration_result["steps"].append({"step": "Confirmation Engine", "status": "Failed", "error": str(e)})
                 continue
        else:
             orchestration_result["steps"].append({"step": "Confirmation Engine", "status": "Skipped", "reason": "Module not available or POI not tapped"})
             continue
        # --- MODIFICATION END --- ###


        # --- 6. Run Entry Executor (using sliced data) --- ### MODIFICATION START ###
        final_entry_result = None
        if execute_smc_entry and confirmation_result and confirmation_result.get("confirmation_status") and ltf_poi_timestamp:
             print(f"[INFO][AdvSMCOrch] Running Entry Executor (Exec TF: {execution_tf}, Data After: {ltf_poi_timestamp})...")
             try:
                 # Slice execution data: Only candles STRICTLY AFTER the LTF POI formation time
                 execution_data_slice = all_tf_data[execution_tf][all_tf_data[execution_tf].index > ltf_poi_timestamp]

                 if execution_data_slice.empty:
                     print("[WARN][AdvSMCOrch] No execution timeframe data available after LTF POI timestamp. Skipping entry.")
                     orchestration_result["steps"].append({"step": "Entry Execution", "status": "Skipped", "reason": "No data after LTF POI formation"})
                     continue
                 else:
                      print(f"[DEBUG][AdvSMCOrch] Execution using {len(execution_data_slice)} candles from {execution_data_slice.index.min()} onwards.")
                      final_entry_result = execute_smc_entry(
                         ltf_data=execution_data_slice, # Pass the SLICED data
                         confirmation_data=confirmation_result,
                         strategy_variant=strategy_variant,
                         risk_model_config=risk_config,
                         spread_points=variant_config.get("spread_points", 0),
                         symbol=symbol
                         # Pass confluence/execution params if available/needed
                      )
                      orchestration_result["steps"].append({"step": "Entry Execution", "result": final_entry_result})

                      if final_entry_result and final_entry_result.get("entry_confirmed"):
                         print("[INFO][AdvSMCOrch] Entry Execution SUCCESS.")
                         orchestration_result["final_entry_result"] = final_entry_result
                         entry_found = True
                         # --- 7. Dispatch Signal ---
                         dispatch_pine_payload(final_entry_result)
                      else:
                         print("[INFO][AdvSMCOrch] Entry execution failed or conditions not met.")

             except Exception as e:
                 print(f"[ERROR][AdvSMCOrch] Entry Executor failed: {e}\n{traceback.format_exc()}")
                 orchestration_result["steps"].append({"step": "Entry Execution", "status": "Failed", "error": str(e)})

        else:
             reason = "Module not available"
             if not (confirmation_result and confirmation_result.get("confirmation_status")): reason = "Confirmation failed"
             if not ltf_poi_timestamp: reason = "LTF POI timestamp missing"
             orchestration_result["steps"].append({"step": "Entry Execution", "status": "Skipped", "reason": reason})
             continue
        # --- MODIFICATION END --- ###

    # --- End of POI Loop ---
    if not entry_found:
        print(f"\n[INFO][AdvSMCOrch] No confirmed entry found for variant {strategy_variant}.")
        orchestration_result["summary"] = f"Orchestration complete: No entry found (FAILURE)."
    else:
        orchestration_result["summary"] = f"Orchestration complete: Entry successfully found and executed (SUCCESS)."

    print(f"--- Advanced SMC Orchestrator Complete for Variant: {strategy_variant} ---")
    print(f"[SUMMARY][AdvSMCOrch] {orchestration_result['summary']}")
    return orchestration_result


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("--- Testing Advanced SMC Orchestrator Stub (Slice-Aware) ---")

    # Create dummy multi-TF data
    dummy_all_tf_data = {}
    base_time = pd.Timestamp('2023-10-27 09:00:00', tz='UTC')
    for tf_code, freq, num_periods in [('m1', 'T', 300), ('m5', '5T', 100), ('m15', '15T', 50), ('h1', 'H', 24), ('h4', '4H', 10)]:
        timestamps = pd.date_range(start=base_time, periods=num_periods, freq=freq, tz='UTC')
        base_price = 1.1000
        data = {'Open': base_price + np.random.normal(0, 0.001, num_periods), 'Close': base_price + np.random.normal(0, 0.001, num_periods)}
        data['High'] = np.maximum(data['Open'], data['Close']) + np.random.uniform(0, 0.0005, num_periods)
        data['Low'] = np.minimum(data['Open'], data['Close']) - np.random.uniform(0, 0.0005, num_periods)
        df = pd.DataFrame(data, index=timestamps)
        dummy_all_tf_data[tf_code] = df

    # --- Simulate a POI tap in the M15 data ---
    poi_range_to_test = [1.0980, 1.0985] # Below typical price
    tap_time_idx = -15 # Tap 15 periods from the end
    tap_timestamp = dummy_all_tf_data['m15'].index[tap_time_idx]
    dummy_all_tf_data['m15'].loc[tap_timestamp, 'Low'] = poi_range_to_test[0] - 0.0001 # Ensure it wicks below low
    dummy_all_tf_data['m15'].loc[tap_timestamp, 'High'] = poi_range_to_test[1] # Ensure high touches POI high
    print(f"Simulated POI tap in M15 data at {tap_timestamp}")

    # --- Simulate confirmation pattern in M5 data *after* the tap time ---
    conf_start_time = tap_timestamp + pd.Timedelta(minutes=1) # Start looking after tap
    m5_after_tap = dummy_all_tf_data['m5'][dummy_all_tf_data['m5'].index > tap_timestamp]
    if len(m5_after_tap) > 10:
        swing_high_idx = m5_after_tap.index[5]
        choch_break_idx = m5_after_tap.index[8]
        ltf_poi_form_idx = m5_after_tap.index[7] # Candle before break forms POI
        swing_high_price = m5_after_tap['High'].iloc[3:7].mean() + 0.0005
        dummy_all_tf_data['m5'].loc[swing_high_idx, 'High'] = swing_high_price # Create swing high
        dummy_all_tf_data['m5'].loc[choch_break_idx, 'High'] = swing_high_price + 0.0002 # Break it (CHoCH)
        # Simulate LTF POI (e.g., OB) before break
        dummy_all_tf_data['m5'].loc[ltf_poi_form_idx, 'Open'] = dummy_all_tf_data['m5'].loc[ltf_poi_form_idx, 'Close'] + 0.0001 # Bearish candle
        print(f"Simulated CHoCH in M5 data around {choch_break_idx}")


    # --- Simulate mitigation in M1 data *after* potential LTF POI formation ---
    # Assume LTF POI formed around ltf_poi_form_idx
    exec_start_time = ltf_poi_form_idx + pd.Timedelta(minutes=1)
    m1_after_ltf_poi = dummy_all_tf_data['m1'][dummy_all_tf_data['m1'].index > ltf_poi_form_idx]
    if len(m1_after_ltf_poi) > 5:
         mitigation_idx = m1_after_ltf_poi.index[3]
         # Assume LTF POI range based on ltf_poi_form_idx candle
         ltf_poi_candle = dummy_all_tf_data['m5'].loc[ltf_poi_form_idx] # Use M5 candle that formed POI
         ltf_poi_range_sim = [ltf_poi_candle['Low'], ltf_poi_candle['High']]
         dummy_all_tf_data['m1'].loc[mitigation_idx, 'Low'] = ltf_poi_range_sim[1] - 0.00005 # Wick into assumed LTF POI
         print(f"Simulated Mitigation in M1 data around {mitigation_idx}")


    test_variant = "Inv"
    test_timestamp = dummy_all_tf_data['m1'].index[-1] # Use latest timestamp

    # Ensure strategy_profiles.json exists
    if not Path("strategy_profiles.json").exists():
         print("[WARN] Creating dummy strategy_profiles.json for testing.")
         dummy_profiles = {
             "Inv": {"structure_timeframe": "h4", "poi_timeframe": "h1", "poi_tap_check_timeframe": "m15", "confirmation_timeframe": "m5", "execution_timeframe": "m1", "risk_model_config": {"tp_rr": 3.0, "risk_percent": 1.0}, "poi_tap_config": {"lookback_candles": 20}},
             "Mentfx": {"structure_timeframe": "h4", "poi_timeframe": "h1", "poi_tap_check_timeframe": "m15", "confirmation_timeframe": "m5", "execution_timeframe": "m1", "require_phase": "Correction", "risk_model_config": {"tp_rr": 5.0, "risk_percent": 0.5}, "poi_tap_config": {"lookback_candles": 20}},
             "default": {"structure_timeframe": "h1", "poi_timeframe": "m15", "poi_tap_check_timeframe": "m15", "confirmation_timeframe": "m5", "execution_timeframe": "m1", "risk_model_config": {"tp_rr": 2.0, "risk_percent": 1.0}, "poi_tap_config": {"lookback_candles": 10}}
         }
         try:
             with open("strategy_profiles.json", "w") as f: json.dump(dummy_profiles, f, indent=2)
         except Exception as e: print(f"Failed to create dummy profiles: {e}")

    # --- Run the orchestrator ---
    orchestration_output = run_advanced_smc_strategy(
        all_tf_data=dummy_all_tf_data,
        strategy_variant=test_variant,
        target_timestamp=test_timestamp, # Less relevant now, but kept for reference
        symbol="EURUSD"
    )

    print("\n--- Orchestration Result ---")
    print(json.dumps(orchestration_output, indent=2, default=str))

    print("\n--- Orchestrator Stub Testing Complete ---")
