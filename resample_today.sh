#!/bin/bash
# resample_today.sh
# Zanzibar v5.1 Shell Script
# Version: 5.1.0
# Description: Automates M1 to HTF resampling for all tracked symbols.

# Navigate to project directory
cd /path/to/zt_v5.1/ || exit

# Run resampler
python3 core/resample_m1_to_htf_parallel.py

# Optional: Log timestamp
echo "[INFO] Resample completed at $(date)" >> logs/resample_log.txt