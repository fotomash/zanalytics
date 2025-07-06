#!/usr/bin/env python3
"""
ZANFLOW Master Run Script
Handles all startup, module fixing, and service orchestration
"""

import os
import sys
import subprocess
import time
import json
import yaml
from pathlib import Path
import signal
import threading

class ZanflowMaster:
    def __init__(self):
        self.project_root = Path.cwd()
        self.processes = {}
        self.running = True

    def setup_environment(self):
        """Setup Python environment and fix imports"""
        print("🔧 Setting up environment...")

        # Add to Python path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        # Create all __init__.py files
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            if any(f.endswith('.py') for f in files):
                init_file = os.path.join(root, '__init__.py')
                if not os.path.exists(init_file):
                    Path(init_file).touch()
                    print(f"✅ Created {os.path.relpath(init_file)}")

        # Fix common import issues
        self.fix_imports()

    def fix_imports(self):
        """Fix import statements across the project"""
        import_mappings = {
            # Core imports
            'from core.data_manager import': 'from core.data_manager import',
            'from core.orchestration.analysis_orchestrator import': 'from core.orchestration.analysis_orchestrator import',
            'from core from core from core from core from core import data_manager': 'from core from core from core from core from core from core import data_manager',

            # Analysis imports
            'from analysis.wyckoff_analysis import': 'from analysis.wyckoff_analysis import',
            'from analysis.midas_analysis import': 'from analysis.midas_analysis import',

            # Config imports
            'from config import orchestrator_config': 'from config import orchestrator_config',
            'from config import orchestrator_config': 'from config import orchestrator_config',

            # Dashboard imports
            'from dashboards.config.dashboard_config import': 'from dashboards.config.dashboard_config import',
            'from dashboards.utils from dashboards.utils from dashboards.utils from dashboards.utils from dashboards.utils import dashboard_utils': 'from dashboards.utils from dashboards.utils from dashboards.utils from dashboards.utils from dashboards.utils from dashboards.utils import dashboard_utils',
        }

        fixed_files = []

        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            if any(skip in root for skip in ['.git', '__pycache__', 'venv', '.env']):
                continue

            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()

                        original_content = content
                        for old_import, new_import in import_mappings.items():
                            if old_import in content:
                                content = content.replace(old_import, new_import)

                        if content != original_content:
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(content)
                            fixed_files.append(os.path.relpath(filepath))
                    except Exception as e:
                        print(f"⚠️  Could not process {filepath}: {e}")

        if fixed_files:
            print(f"✅ Fixed imports in {len(fixed_files)} files")
            for f in fixed_files[:5]:  # Show first 5
                print(f"   - {f}")
            if len(fixed_files) > 5:
                print(f"   ... and {len(fixed_files) - 5} more")

    def ensure_data_structure(self):
        """Ensure proper data structure exists"""
        print("📁 Ensuring data structure...")

        # Required directories
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
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Ensured: {directory}")

        # Create data manifest if missing
        manifest_path = self.project_root / 'data' / 'data_manifest.yml'
        if not manifest_path.exists():
            manifest_content = {
                'version': '1.0',
                'data_sources': {
                    'market_data': {
                        'path': 'data/market',
                        'format': 'csv',
                        'description': 'Market price data',
                        'schema': {
                            'timestamp': 'datetime',
                            'open': 'float',
                            'high': 'float',
                            'low': 'float',
                            'close': 'float',
                            'volume': 'int'
                        }
                    },
                    'indicators': {
                        'path': 'data/indicators',
                        'format': 'json',
                        'description': 'Technical indicators'
                    },
                    'analysis_results': {
                        'path': 'data/analysis',
                        'format': 'json',
                        'description': 'Analysis outputs'
                    }
                },
                'quality_checks': {
                    'market_data': {
                        'null_check': True,
                        'range_check': True,
                        'timestamp_order': True
                    }
                }
            }

            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_content, f, default_flow_style=False)
            print("✅ Created data_manifest.yml")

    def check_dependencies(self):
        """Check and install required dependencies"""
        print("📦 Checking dependencies...")

        required_packages = [
            'streamlit',
            'flask',
            'flask-cors',
            'pandas',
            'numpy',
            'plotly',
            'pyyaml',
            'requests',
            'scikit-learn',
            'matplotlib',
            'seaborn'
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package} installed")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} missing")

        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + ' '.join(missing_packages))
            return False

        return True

    def start_service(self, name, command, cwd=None, env=None):
        """Start a service in the background"""
        try:
            if cwd is None:
                cwd = self.project_root

            if env is None:
                env = os.environ.copy()
                env['PYTHONPATH'] = str(self.project_root)

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd),
                env=env,
                shell=isinstance(command, str)
            )

            self.processes[name] = process
            print(f"✅ Started {name} (PID: {process.pid})")

            # Start output monitoring thread
            threading.Thread(
                target=self.monitor_output,
                args=(name, process),
                daemon=True
            ).start()

            return True

        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False

    def monitor_output(self, name, process):
        """Monitor service output"""
        while self.running and process.poll() is None:
            line = process.stdout.readline()
            if line:
                print(f"[{name}] {line.decode().strip()}")

    def start_all_services(self):
        """Start all ZANFLOW services"""
        print("\n🚀 Starting ZANFLOW services...")

        # 1. API Service
        api_script = self.project_root / 'zanalytics_api_service.py'
        if api_script.exists():
            self.start_service(
                'API Service',
                [sys.executable, str(api_script)]
            )
        else:
            print("⚠️  API service not found, creating minimal version...")
            self.create_minimal_api()
            self.start_service(
                'API Service',
                [sys.executable, 'zanalytics_api_service.py']
            )

        time.sleep(3)  # Wait for API to start

        # 2. Dashboard
        dashboard_home = self.project_root / 'dashboards' / 'Home.py'
        if dashboard_home.exists():
            self.start_service(
                'Dashboard',
                ['streamlit', 'run', str(dashboard_home), '--server.port', '8501']
            )
        else:
            print("⚠️  Dashboard not found, creating basic structure...")
            self.create_basic_dashboard()
            self.start_service(
                'Dashboard',
                ['streamlit', 'run', 'dashboards/Home.py', '--server.port', '8501']
            )

        time.sleep(3)  # Wait for dashboard to start

        # 3. Optional: Analysis Orchestrator
        orchestrator = self.project_root / 'core' / 'orchestration' / 'analysis_orchestrator.py'
        if orchestrator.exists():
            self.start_service(
                'Orchestrator',
                [sys.executable, str(orchestrator)]
            )

        # 4. Optional: Scheduling Agent
        scheduler = self.project_root / 'agents' / 'scheduling_agent.py'
        if scheduler.exists():
            self.start_service(
                'Scheduler',
                [sys.executable, str(scheduler)]
            )

    def create_minimal_api(self):
        """Create minimal API if missing"""
        api_code = """from flask import Flask, jsonify, request
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/strategies')
def list_strategies():
    return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)
"""
        with open('zanalytics_api_service.py', 'w') as f:
            f.write(api_code)

    def create_basic_dashboard(self):
        """Create basic dashboard structure"""
        os.makedirs('dashboards/pages', exist_ok=True)

        home_code = """import streamlit as st

st.set_page_config(page_title="ZANFLOW", page_icon="🚀", layout="wide")

st.title("🚀 ZANFLOW Trading Intelligence")
st.markdown("Welcome to your trading command center!")

st.sidebar.success("Select a page above.")
"""

        with open('dashboards/Home.py', 'w') as f:
            f.write(home_code)

    def show_status(self):
        """Show status of all services"""
        print("\n" + "=" * 60)
        print("📊 ZANFLOW STATUS")
        print("=" * 60)

        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"✅ {name}: Running (PID: {process.pid})")
            else:
                print(f"❌ {name}: Stopped")

        print("\n🌐 Access Points:")
        print("   Dashboard: http://localhost:8501")
        print("   API: http://localhost:5010")
        print("   API Health: http://localhost:5010/health")
        print("\n💡 Commands:")
        print("   Press 's' for status")
        print("   Press 'r' to restart services")
        print("   Press 'q' or Ctrl+C to quit")
        print("=" * 60)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown gracefully"""
        print("\n🛑 Shutting down ZANFLOW...")
        self.running = False

        for name, process in self.processes.items():
            if process.poll() is None:
                process.terminate()
                print(f"   Stopped {name}")

        print("✅ Shutdown complete")
        sys.exit(0)

    def run(self):
        """Main run loop"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        print("🚀 ZANFLOW Master Controller")
        print("=" * 60)

        # Setup steps
        self.setup_environment()
        self.ensure_data_structure()

        if not self.check_dependencies():
            print("\n❌ Please install missing dependencies first")
            return

        # Start services
        self.start_all_services()

        # Show status
        self.show_status()

        # Interactive loop
        while self.running:
            try:
                cmd = input("\n> ").strip().lower()

                if cmd == 'q':
                    self.handle_shutdown(None, None)
                elif cmd == 's':
                    self.show_status()
                elif cmd == 'r':
                    print("Restarting services...")
                    for name, process in list(self.processes.items()):
                        process.terminate()
                        time.sleep(1)
                    self.start_all_services()
                    self.show_status()
                else:
                    print("Unknown command. Use 's' for status, 'r' to restart, 'q' to quit")

            except KeyboardInterrupt:
                self.handle_shutdown(None, None)
            except EOFError:
                # Handle when running in background
                time.sleep(1)

if __name__ == "__main__":
    master = ZanflowMaster()
    master.run()
