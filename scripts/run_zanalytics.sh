#!/bin/bash
# ZAnalytics Run Script

echo "Starting ZAnalytics System..."

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found!"
    exit 1
fi

# Run startup checks
python3 zanalytics_startup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Starting Analysis Orchestrator..."
    python3 -m core.orchestrator
else
    echo "Startup checks failed!"
    exit 1
fi
