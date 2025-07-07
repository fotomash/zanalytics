#!/bin/bash
# ZANFLOW Multi-Asset Tick Analyzer System Startup

echo "🚀 Starting ZANFLOW Multi-Asset Tick Analysis System"

# Check if Redis is installed
if command -v redis-server >/dev/null 2>&1; then
    echo "✅ Redis found"

    # Start Redis in the background (if not already running)
    redis-server --daemonize yes
    echo "✅ Redis server started"
else
    echo "⚠️ Redis not found. System will use memory storage instead."
    echo "   For better performance, install Redis:"
    echo "   macOS: brew install redis"
    echo "   Windows: Download from GitHub"
fi

# Check if Python is installed
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "❌ Python not found. Please install Python 3.7+ first."
    exit 1
fi

# Check for required Python packages
$PYTHON -c "import flask, pandas" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Missing required Python packages. Installing now..."
    $PYTHON -m pip install flask pandas streamlit

    # Try to install redis, but continue if it fails
    $PYTHON -m pip install redis || echo "⚠️ Redis Python client not installed. Will use memory storage."
fi

# Start the processor in the background
echo "🔄 Starting tick processor..."
$PYTHON multi_asset_tick_processor_v2.py > processor.log 2>&1 &
PROCESSOR_PID=$!
echo "✅ Tick processor started (PID: $PROCESSOR_PID)"

# Wait a moment for the processor to initialize
sleep 2

# Start the dashboard
echo "📊 Starting dashboard..."
$PYTHON -m streamlit run multi_asset_tick_dashboard_v2.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "✅ Dashboard started (PID: $DASHBOARD_PID)"

echo ""
echo "✨ ZANFLOW Multi-Asset Tick Analysis System is now running!"
echo "   Dashboard: http://localhost:8501"
echo "   Processor API: http://localhost:5000"
echo ""
echo "🔍 To monitor logs:"
echo "   Processor: tail -f processor.log"
echo "   Dashboard: tail -f dashboard.log"
echo ""
echo "⚠️ To stop the system:"
echo "   kill $PROCESSOR_PID $DASHBOARD_PID"
echo "   redis-cli shutdown (if Redis is running)"
echo ""
