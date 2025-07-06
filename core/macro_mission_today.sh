#!/bin/bash
# macro_mission_today.sh
# Zanzibar v5.1  –  Pro-Desk Macro Launcher
# Author: Captain Zanzibar FTMO Desk
# Version: 5.1.0
# ----------------------------------------
# 1.  Run unified data pipeline (fetch macro + M1 + resample)
echo "[STEP 1] Running data pipeline (macro + M1 fetch + resample)…"
python3 core/data_pipeline.py || { echo "❌ Data pipeline failed"; exit 1; }

# 2.  Liquidity / wick diagnostics on macro assets
echo "[STEP 2] Scanning wicks & liquidity shifts…"
python3 core/wick_liquidity_monitor.py       || { echo "❌ Wick scan failed"; exit 1; }

# 3.  Mission complete log
STAMP=$(date '+%Y-%m-%d %H:%M')
echo "[✅  MACRO MISSION COMPLETE — $STAMP]" | tee -a logs/macro_mission_log.txt