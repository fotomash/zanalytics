#!/usr/bin/env python3
"""
Ultimate Strategy System Deployment Script
One-click deployment for the complete XANA-ready trading system
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path
import threading
import argparse
import json

class SystemDeployer:
    """Deploys and manages the complete trading system"""

    def __init__(self, data_dir: str = "./data", api_port: int = 8000, streamlit_port: int = 8501):
        self.data_dir = Path(data_dir)
        self.api_port = api_port
        self.streamlit_port = streamlit_port
        self.processes = {}

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        required_packages = [
            'fastapi', 'uvicorn', 'streamlit', 'requests', 
            'pandas', 'numpy', 'plotly', 'pathlib'
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            print(f"❌ Missing packages: {', '.join(missing)}")
            print(f"Install with: pip install {' '.join(missing)}")
            return False

        print("✅ All dependencies satisfied")
        return True

    def check_scripts(self):
        """Check if all required scripts exist"""
        required_scripts = [
            'ultimate_strategy_merger.py',
            'ultimate_strategy_api.py', 
            '🏠 Home.py',
            'ncOS_ultimate_microstructure_analyzer_DEFAULTS.py'
        ]

        missing = []
        for script in required_scripts:
            if not Path(script).exists():
                missing.append(script)

        if missing:
            print(f"❌ Missing scripts: {', '.join(missing)}")
            return False

        print("✅ All scripts found")
        return True

    def start_api_server(self):
        """Start the FastAPI server"""
        print(f"🚀 Starting API server on port {self.api_port}...")

        cmd = [
            sys.executable, 'ultimate_strategy_api.py',
            '--host', '0.0.0.0',
            '--port', str(self.api_port),
            '--data-dir', str(self.data_dir)
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes['api'] = process

        # Wait a moment for startup
        time.sleep(3)

        if process.poll() is None:
            print(f"✅ API server started (PID: {process.pid})")
            return True
        else:
            print("❌ API server failed to start")
            return False

    def start_streamlit_dashboard(self):
        """Start the Streamlit dashboard"""
        print(f"📊 Starting Streamlit dashboard on port {self.streamlit_port}...")

        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            '🏠 Home.py',
            '--server.port', str(self.streamlit_port),
            '--server.headless', 'true'
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes['streamlit'] = process

        # Wait a moment for startup
        time.sleep(5)

        if process.poll() is None:
            print(f"✅ Dashboard started (PID: {process.pid})")
            return True
        else:
            print("❌ Dashboard failed to start")
            return False

    def start_ngrok(self, service: str = "api"):
        """Start ngrok tunnel"""
        port = self.api_port if service == "api" else self.streamlit_port

        print(f"🌐 Starting ngrok tunnel for {service} on port {port}...")

        cmd = ['ngrok', 'http', str(port)]

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes[f'ngrok_{service}'] = process

            time.sleep(3)

            if process.poll() is None:
                print(f"✅ ngrok tunnel started for {service}")
                print(f"🔗 Check ngrok dashboard: http://127.0.0.1:4040")
                return True
            else:
                print(f"❌ ngrok failed to start for {service}")
                return False

        except FileNotFoundError:
            print("❌ ngrok not found. Install from https://ngrok.com/")
            return False

    def run_analyzer(self, symbol: str = "XAUUSD"):
        """Run the ncOS analyzer to generate data"""
        print(f"📈 Running ncOS analyzer for {symbol}...")

        cmd = [
            sys.executable, 'ncOS_ultimate_microstructure_analyzer_DEFAULTS.py',
            '--directory', str(self.data_dir),
            '--output_dir', str(self.data_dir)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ ncOS analyzer completed successfully")
                return True
            else:
                print(f"❌ ncOS analyzer failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("⏰ ncOS analyzer timed out (5 minutes)")
            return False
        except Exception as e:
            print(f"❌ ncOS analyzer error: {e}")
            return False

    def run_merger(self, symbol: str = "XAUUSD"):
        """Run the strategy merger"""
        print(f"🔄 Running strategy merger for {symbol}...")

        cmd = [
            sys.executable, 'ultimate_strategy_merger.py',
            '--data-dir', str(self.data_dir),
            '--symbol', symbol
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("✅ Strategy merger completed")
                return True
            else:
                print(f"❌ Strategy merger failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Merger error: {e}")
            return False

    def deploy_full_system(self, enable_ngrok: bool = True, run_analysis: bool = True):
        """Deploy the complete system"""
        print("🚀 DEPLOYING ULTIMATE STRATEGY SYSTEM")
        print("=" * 50)

        # 1. Check dependencies
        if not self.check_dependencies():
            return False

        # 2. Check scripts
        if not self.check_scripts():
            return False

        # 3. Run analysis if requested
        if run_analysis:
            if not self.run_analyzer():
                print("⚠️ Analysis failed, but continuing with deployment...")

            if not self.run_merger():
                print("⚠️ Merger failed, but continuing with deployment...")

        # 4. Start API server
        if not self.start_api_server():
            return False

        # 5. Start dashboard
        if not self.start_streamlit_dashboard():
            return False

        # 6. Start ngrok if requested
        if enable_ngrok:
            self.start_ngrok("api")
            # Optionally expose dashboard too
            # self.start_ngrok("streamlit")

        # 7. Display access information
        self.display_access_info(enable_ngrok)

        return True

    def display_access_info(self, ngrok_enabled: bool):
        """Display system access information"""
        print("\n🎉 SYSTEM DEPLOYED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📊 Dashboard: http://localhost:{self.streamlit_port}")
        print(f"🔗 API: http://localhost:{self.api_port}")
        print(f"📚 API Docs: http://localhost:{self.api_port}/docs")

        if ngrok_enabled:
            print(f"🌐 ngrok Dashboard: http://127.0.0.1:4040")
            print("🔗 Public API URL: Check ngrok dashboard")

        print(f"📁 Data Directory: {self.data_dir.absolute()}")
        print("\n🎯 KEY ENDPOINTS:")
        print(f"  • Consolidated Data: /summary/consolidated")
        print(f"  • Tick Window: /microstructure/tick-window")
        print(f"  • Entry Signals: /signals/entry")
        print("\n💡 USAGE:")
        print("  • Open dashboard in browser for visual interface")
        print("  • Use API endpoints for ChatGPT/bot integration")
        print("  • Run analyzer regularly to update data")
        print("\n⏹️  Press Ctrl+C to stop all services")

    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")

        for name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"  Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("✅ All services stopped")

    def monitor_services(self):
        """Monitor running services"""
        try:
            while True:
                time.sleep(30)

                # Check if services are still running
                for name, process in self.processes.items():
                    if process and process.poll() is not None:
                        print(f"⚠️ Service {name} stopped unexpectedly")

        except KeyboardInterrupt:
            self.stop_all_services()

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Ultimate Strategy System Deployer')

    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--api-port', type=int, default=8000, help='API server port')
    parser.add_argument('--dashboard-port', type=int, default=8501, help='Dashboard port')
    parser.add_argument('--no-ngrok', action='store_true', help='Disable ngrok tunnel')
    parser.add_argument('--no-analysis', action='store_true', help='Skip running analysis')
    parser.add_argument('--symbol', default='XAUUSD', help='Symbol to analyze')

    # Quick start modes
    parser.add_argument('--quick', action='store_true', help='Quick start (API + Dashboard only)')
    parser.add_argument('--api-only', action='store_true', help='Start API server only')
    parser.add_argument('--dashboard-only', action='store_true', help='Start dashboard only')

    args = parser.parse_args()

    # Create deployer
    deployer = SystemDeployer(
        data_dir=args.data_dir,
        api_port=args.api_port,
        streamlit_port=args.dashboard_port
    )

    # Handle different modes
    if args.api_only:
        print("🚀 Starting API server only...")
        if deployer.check_dependencies() and deployer.start_api_server():
            print(f"✅ API running at http://localhost:{args.api_port}")
            try:
                deployer.monitor_services()
            except KeyboardInterrupt:
                deployer.stop_all_services()

    elif args.dashboard_only:
        print("📊 Starting dashboard only...")
        if deployer.check_dependencies() and deployer.start_streamlit_dashboard():
            print(f"✅ Dashboard running at http://localhost:{args.dashboard_port}")
            try:
                deployer.monitor_services()
            except KeyboardInterrupt:
                deployer.stop_all_services()

    else:
        # Full deployment
        if deployer.deploy_full_system(
            enable_ngrok=not args.no_ngrok,
            run_analysis=not args.no_analysis
        ):
            try:
                deployer.monitor_services()
            except KeyboardInterrupt:
                deployer.stop_all_services()

if __name__ == "__main__":
    main()
