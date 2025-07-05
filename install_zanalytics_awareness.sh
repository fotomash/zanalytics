
#!/bin/bash
# install_zanalytics_awareness.sh - Install ZANALYTICS Data Awareness System

echo "🚀 Installing ZANALYTICS Data Awareness System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Install required packages
echo "📦 Installing required packages..."
pip3 install pandas watchdog

# Create directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p exports
mkdir -p signals
mkdir -p analysis_output
mkdir -p logs

# Set permissions
chmod +x start_zanalytics_awareness.py
chmod +x dashboard.py

echo "✅ Installation complete!"
echo ""
echo "🎯 QUICK START:"
echo "   1. Run: python3 start_zanalytics_awareness.py"
echo "   2. In another terminal: python3 dashboard.py"
echo "   3. Drop CSV/JSON files in data/ or exports/ directories"
echo "   4. Watch your ZANALYTICS agents come alive!"
echo ""
echo "📁 Copy your ZANALYTICS files to this directory"
echo "📊 Monitor the dashboard for real-time activity"
