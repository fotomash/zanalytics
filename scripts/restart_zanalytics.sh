#!/bin/bash
# restart_zanalytics.sh - Restart ZANALYTICS system

echo "🔄 RESTARTING ZANALYTICS SYSTEM"
echo "==============================="

# Stop all services
./stop_zanalytics.sh

echo ""
echo "⏳ Waiting 3 seconds..."
sleep 3

# Start all services
./start_zanalytics_full.sh
