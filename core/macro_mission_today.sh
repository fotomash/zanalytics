#!/bin/bash
# macro_mission_today.sh
# Zanzibar v5.1  –  Pro-Desk Macro Launcher
# Author: Captain Zanzibar FTMO Desk
# Version: 5.1.0
# ----------------------------------------
# 1.  Fetch macro assets  (M5)
echo "[STEP 1] Fetching macro assets (VIX, SPX, DXY, US10Y, Gold, BTC, Oil, EURUSD)…"
python3 core/massive_macro_fetcher.py   || { echo "❌ Macro fetch failed"; exit 1; }

# 2.  Fetch intraday M1 pairs (EURUSD, GBPUSD, XAUUSD…)
echo "[STEP 2] Fetching tracked-pair M1 data…"
python3 core/m1_data_fetcher.py         || { echo "❌ M1 pair fetch failed"; exit 1; }

# 3.  Resample every M1 file to higher TFs
echo "[STEP 3] Resampling M1 ➔ M5-W1 in parallel…"
python3 core/resample_m1_to_htf_parallel.py  || { echo "❌ Resample failed"; exit 1; }

# 4.  Liquidity / wick diagnostics on macro assets
echo "[STEP 4] Scanning wicks & liquidity shifts…"
python3 core/wick_liquidity_monitor.py       || { echo "❌ Wick scan failed"; exit 1; }

# 5.  Mission complete log
STAMP=$(date '+%Y-%m-%d %H:%M')
echo "[✅  MACRO MISSION COMPLETE — $STAMP]" | tee -a logs/macro_mission_log.txt