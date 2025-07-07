#!/bin/bash
# master_launch_today.sh
# Zanzibar v5.1 Shell Script
# Version: 5.1.0
# Description: Master script to fetch M1, resample HTF, prepare journal, and launch daily objectives.

# Navigate to project directory
cd /path/to/zt_v5.1/ || exit

# Step 1: Fetch latest M1 data
echo "[STEP 1] Fetching latest M1 data..."
./fetch_today_m1.sh

# Step 2: Resample M1 to higher timeframes
echo "[STEP 2] Resampling M1 to HTF datasets..."
./resample_today.sh

# Step 3: Analyze today (optional: prepare journal, log setup)
echo "[STEP 3] Launching today's analysis & objectives..."
./analyze_today.sh

# Completion message
echo "[MASTER LAUNCH COMPLETE] Zanzibar v5.1 Daily Boot Completed at $(date)" >> logs/master_launch_log.txt

# Optional: Notify success
echo "✅ Zanzibar v5.1 Daily Launch Sequence Completed Successfully!"
