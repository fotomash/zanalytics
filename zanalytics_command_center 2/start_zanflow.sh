#!/bin/bash
# ZANFLOW Simple Startup Script

echo "🚀 Starting ZANFLOW Services..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    exit 1
fi

# Check required packages
echo "📦 Checking dependencies..."
python3 -c "import streamlit" 2>/dev/null || echo "⚠️  Missing: streamlit"
python3 -c "import flask" 2>/dev/null || echo "⚠️  Missing: flask"
python3 -c "import pandas" 2>/dev/null || echo "⚠️  Missing: pandas"

# Fix Python paths
echo "🔧 Fixing Python paths..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -exec touch {}/__init__.py \; 2>/dev/null

# Start API
echo "🌐 Starting API Service..."
python3 zanalytics_api_service.py &
API_PID=$!
sleep 3

# Start Dashboard
echo "📊 Starting Dashboard..."
streamlit run dashboards/Home.py &
DASH_PID=$!
sleep 3

echo "✅ Services started!"
echo "   API: http://localhost:5010 (PID: $API_PID)"
echo "   Dashboard: http://localhost:8501 (PID: $DASH_PID)"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait and cleanup
trap "kill $API_PID $DASH_PID 2>/dev/null; echo 'Services stopped.'" EXIT
wait
