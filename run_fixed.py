#!/usr/bin/env python3
"""
ZANFLOW v12 - Unified Startup Script
Fixed version with proper import checking
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ZanflowRunner:
    def __init__(self):
        self.processes = []
        self.running = True

    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("📦 Checking dependencies...")

        required_packages = {
            'streamlit': 'streamlit',
            'flask': 'flask',
            'flask_cors': 'flask-cors',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'plotly': 'plotly',
            'yaml': 'pyyaml',
            'requests': 'requests',
            'sklearn': 'scikit-learn',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn'
        }

        missing = []
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"✅ {package_name} installed")
            except ImportError:
                print(f"❌ {package_name} missing")
                missing.append(package_name)

        if missing:
            print(f"\n⚠️  Missing packages: {', '.join(missing)}")
            print(f"Install with: pip install {' '.join(missing)}")
            return False

        return True

    def ensure_directories(self):
        """Ensure all required directories exist"""
        print("\n📁 Ensuring directory structure...")

        directories = [
            'data/market',
            'data/indicators', 
            'data/analysis',
            'knowledge/strategies',
            'knowledge/strategies/backups',
            'commands/queue',
            'logs',
            'dashboards/pages',
            'core/validation',
            'core/components',
            'analysis',
            'agents'
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Ensured: {directory}")

    def start_api(self):
        """Start the Flask API server"""
        print("\n🚀 Starting API server...")

        # Create a simple API starter if api.py doesn't exist
        if not os.path.exists('api.py'):
            api_content = """
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'ZANFLOW v12 API'})

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'version': 'v12',
        'services': {
            'api': 'active',
            'dashboard': 'pending'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
"""
            with open('api.py', 'w') as f:
                f.write(api_content)

        process = subprocess.Popen(
            [sys.executable, 'api.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(process)
        return process

    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        print("\n🎨 Starting Dashboard...")

        # Create a simple dashboard if it doesn't exist
        if not os.path.exists('dashboards/main_dashboard.py'):
            os.makedirs('dashboards', exist_ok=True)
            dashboard_content = """
import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="ZANFLOW v12 Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🌊 ZANFLOW v12 - Trading Intelligence Platform")

# Check API status
try:
    response = requests.get('http://localhost:5000/api/status')
    if response.status_code == 200:
        status = response.json()
        st.success(f"✅ API Status: {status['status']}")
    else:
        st.error("❌ API is not responding")
except:
    st.warning("⚠️ API connection failed - make sure the API server is running")

# Main dashboard content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Active Strategies", "3", "+1")

with col2:
    st.metric("Total Signals", "127", "+12")

with col3:
    st.metric("Win Rate", "67.3%", "+2.1%")

st.markdown("---")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Analysis", "🎯 Strategies", "📊 Performance", "⚙️ Settings"])

with tab1:
    st.header("Market Analysis")
    st.info("Market analysis components will be loaded here")

with tab2:
    st.header("Active Strategies")
    strategies = pd.DataFrame({
        'Strategy': ['London Kill Zone', 'MIDAS Curve', 'Wyckoff Analysis'],
        'Status': ['Active', 'Active', 'Monitoring'],
        'Performance': ['+12.3%', '+8.7%', '+5.2%']
    })
    st.dataframe(strategies)

with tab3:
    st.header("Performance Metrics")
    st.info("Performance charts will be displayed here")

with tab4:
    st.header("System Settings")
    st.info("Configuration options will be available here")
"""
            with open('dashboards/main_dashboard.py', 'w') as f:
                f.write(dashboard_content)

        process = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'dashboards/main_dashboard.py', '--server.port', '8501'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(process)
        return process

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n⏹️  Shutting down ZANFLOW...")
        self.running = False
        for process in self.processes:
            process.terminate()
        sys.exit(0)

    def run(self):
        """Main run method"""
        print("🌊 ZANFLOW v12 - Starting up...")
        print("=" * 50)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Ensure directories
        self.ensure_directories()

        # Check dependencies
        if not self.check_dependencies():
            print("\n❌ Please install missing dependencies first")
            return

        # Start services
        print("\n✅ All dependencies satisfied!")

        # Start API
        api_process = self.start_api()
        time.sleep(2)  # Give API time to start

        # Start Dashboard
        dashboard_process = self.start_dashboard()

        print("\n✅ ZANFLOW v12 is running!")
        print("\n📍 Access points:")
        print("   - Dashboard: http://localhost:8501")
        print("   - API: http://localhost:5000")
        print("\nPress Ctrl+C to stop all services")

        # Keep running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.signal_handler(None, None)

if __name__ == "__main__":
    runner = ZanflowRunner()
    runner.run()
