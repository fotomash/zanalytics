#!/bin/bash
# ZANFLOW Multi-Asset Tick Analysis Quick Start

echo "🚀 Starting ZANFLOW Multi-Asset Tick Analysis System"
echo "=================================================="

# Check Redis
echo "1. Checking Redis..."
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
    sleep 2
fi
echo "✅ Redis running"

# Start API Server
echo -e "\n2. Starting Multi-Asset Processor..."
python multi_asset_tick_processor.py > processor.log 2>&1 &
PROC_PID=$!
sleep 3

if ps -p $PROC_PID > /dev/null; then
    echo "✅ Processor started (PID: $PROC_PID)"
else
    echo "❌ Failed to start processor"
    exit 1
fi

# Start Dashboard
echo -e "\n3. Starting Dashboard..."
streamlit run multi_asset_tick_dashboard.py > dashboard.log 2>&1 &
DASH_PID=$!
sleep 5

if ps -p $DASH_PID > /dev/null; then
    echo "✅ Dashboard started (PID: $DASH_PID)"
    echo "   Open: http://localhost:8501"
else
    echo "❌ Failed to start dashboard"
fi

echo -e "\n=================================================="
echo "✅ System Running!"
echo ""
echo "Next steps:"
echo "1. Install EA in MT5: ZANFLOW_MultiAsset_TickAnalyzer_EA.mq5"
echo "2. Attach to any chart"
echo "3. Configure symbols to monitor"
echo "4. Watch manipulation detection in dashboard"
echo ""
echo "Logs:"
echo "- Processor: tail -f processor.log"
echo "- Dashboard: tail -f dashboard.log"
echo ""
echo "Press Ctrl+C to stop all components"

# Save PIDs
echo $PROC_PID > .processor.pid
echo $DASH_PID > .dashboard.pid

# Trap Ctrl+C
trap cleanup INT

cleanup() {
    echo -e "\nStopping services..."
    kill $PROC_PID $DASH_PID 2>/dev/null
    rm -f .processor.pid .dashboard.pid
    echo "✅ Stopped"
    exit 0
}

# Keep running
while true; do
    sleep 1
done
