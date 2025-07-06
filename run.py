#!/usr/bin/env python3
"""
ZAnalytics Unified Startup Script
Main entry point for all ZAnalytics services.
"""
import os
import sys
import argparse
import asyncio
import logging
import signal
import subprocess
from pathlib import Path
from typing import List, Optional
import yaml

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.orchestrator import get_orchestrator


class ZAnalyticsLauncher:
    """Main launcher for ZAnalytics services."""

    def __init__(self):
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.services = {}
        self.processes = []

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('ZAnalytics')

    def _load_config(self) -> dict:
        """Load main configuration."""
        config_path = project_root / 'orchestrator_config.yaml'
        if not config_path.exists():
            self.logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def start_service(self, service: str, args: Optional[List[str]] = None):
        """Start a specific service."""
        self.logger.info(f"Starting {service} service...")

        if service == 'orchestrator':
            self._start_orchestrator()
        elif service == 'api':
            self._start_api()
        elif service == 'dashboard':
            self._start_dashboard()
        elif service == 'all':
            self._start_all_services()
        else:
            self.logger.error(f"Unknown service: {service}")
            sys.exit(1)

    def _start_orchestrator(self):
        """Start the analysis orchestrator."""
        async def run_orchestrator():
            orchestrator = get_orchestrator()

            def signal_handler(sig, frame):
                self.logger.info("Received shutdown signal")
                asyncio.create_task(orchestrator.stop())

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            try:
                await orchestrator.start()
            except Exception as e:
                self.logger.error(f"Orchestrator error: {e}")
                raise

        asyncio.run(run_orchestrator())

    def _start_api(self):
        """Start the API server."""
        if not self.config.get('api', {}).get('enabled', True):
            self.logger.info("API service is disabled in configuration")
            return

        api_config = self.config['api']
        host = api_config.get('host', '0.0.0.0')
        port = api_config.get('port', 8000)

        api_path = project_root / 'api' / 'server.py'
        if not api_path.exists():
            self._create_basic_api_server()

        cmd = [
            sys.executable,
            '-m', 'uvicorn',
            'api.server:app',
            '--host', host,
            '--port', str(port),
            '--reload' if os.getenv('ENV') == 'development' else '--workers=4'
        ]

        process = subprocess.Popen(cmd)
        self.processes.append(process)
        self.logger.info(f"API server started on {host}:{port}")

    def _start_dashboard(self):
        """Start the dashboard."""
        if not self.config.get('dashboard', {}).get('enabled', True):
            self.logger.info("Dashboard service is disabled in configuration")
            return

        dashboard_config = self.config['dashboard']
        host = dashboard_config.get('host', 'localhost')
        port = dashboard_config.get('port', 8501)

        dashboard_files = list(project_root.glob('*dashboard*.py'))
        if not dashboard_files:
            dashboard_files = list((project_root / 'pages').glob('*.py'))

        if dashboard_files:
            dashboard_file = dashboard_files[0]
        else:
            self.logger.error("No dashboard file found")
            return

        cmd = [
            sys.executable,
            '-m', 'streamlit', 'run',
            str(dashboard_file),
            '--server.address', host,
            '--server.port', str(port),
            '--theme.base', dashboard_config.get('theme', 'dark')
        ]

        process = subprocess.Popen(cmd)
        self.processes.append(process)
        self.logger.info(f"Dashboard started on {host}:{port}")

    def _start_all_services(self):
        """Start all enabled services."""
        if self.config.get('orchestrator', {}).get('enabled', True):
            import threading
            orchestrator_thread = threading.Thread(
                target=self._start_orchestrator,
                daemon=True
            )
            orchestrator_thread.start()

        self._start_api()
        self._start_dashboard()

        try:
            for process in self.processes:
                process.wait()
        except KeyboardInterrupt:
            self.logger.info("Shutting down services...")
            for process in self.processes:
                process.terminate()

    def _create_basic_api_server(self):
        """Create a basic API server if none exists."""
        api_dir = project_root / 'api'
        api_dir.mkdir(exist_ok=True)
        server_path = api_dir / 'server.py'
        if not server_path.exists():
            server_path.write_text(_BASIC_API_SERVER)
            (api_dir / '__init__.py').write_text('')

    def list_services(self):
        """List available services and their status."""
        services = {
            'orchestrator': {
                'enabled': self.config.get('orchestrator', {}).get('enabled', True),
                'status': 'configured' if self.config.get('orchestrator') else 'not configured'
            },
            'api': {
                'enabled': self.config.get('api', {}).get('enabled', True),
                'host': self.config.get('api', {}).get('host', '0.0.0.0'),
                'port': self.config.get('api', {}).get('port', 8000)
            },
            'dashboard': {
                'enabled': self.config.get('dashboard', {}).get('enabled', True),
                'host': self.config.get('dashboard', {}).get('host', 'localhost'),
                'port': self.config.get('dashboard', {}).get('port', 8501)
            }
        }

        print("ZAnalytics Services:")
        print("=" * 50)
        for service, info in services.items():
            print(f"\n{service.upper()}:")
            for key, value in info.items():
                print(f"  {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ZAnalytics Unified Launcher')
    parser.add_argument(
        '--service',
        choices=['orchestrator', 'api', 'dashboard', 'all'],
        default='all',
        help='Service to start (default: all)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available services'
    )
    parser.add_argument(
        '--config',
        default='orchestrator_config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    launcher = ZAnalyticsLauncher()

    if args.list:
        launcher.list_services()
    else:
        launcher.start_service(args.service)


_BASIC_API_SERVER = """'''\nZAnalytics API Server\nAuto-generated basic API implementation.\n'''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from core.orchestrator import get_orchestrator, AnalysisRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='ZAnalytics API',
    description='API for ZAnalytics trading analysis system',
    version='2.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class AnalysisRequestModel(BaseModel):
    symbol: str
    timeframe: str
    analysis_type: str = 'combined'
    parameters: Optional[Dict[str, Any]] = {}

class AnalysisResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime

orchestrator = get_orchestrator()

@app.get('/')
async def root():
    return {'name': 'ZAnalytics API', 'version': '2.0.0', 'status': 'running'}

@app.get('/health')
async def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now(),
        'services': {
            'orchestrator': orchestrator.is_running,
            'engines': orchestrator.get_engine_status()
        }
    }

@app.post('/analyze', response_model=AnalysisResponse)
async def analyze(request: AnalysisRequestModel):
    try:
        analysis_request = AnalysisRequest(
            request_id=f"{request.symbol}_{request.timeframe}_{datetime.now().timestamp()}",
            symbol=request.symbol,
            timeframe=request.timeframe,
            analysis_type=request.analysis_type,
            parameters=request.parameters
        )
        request_id = orchestrator.submit_request(analysis_request)
        result = orchestrator.get_result(request_id, timeout=30)
        if result:
            return AnalysisResponse(
                request_id=request_id,
                status='completed' if result.success else 'failed',
                result=result.result_data if result.success else None,
                error=result.error,
                timestamp=result.timestamp
            )
        return AnalysisResponse(
            request_id=request_id,
            status='timeout',
            error='Analysis request timed out',
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f'Analysis error: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/metrics')
async def get_metrics():
    return orchestrator.get_metrics()

@app.get('/symbols')
async def get_symbols():
    return {'symbols': orchestrator.config['orchestrator']['analysis']['symbols']}

@app.get('/timeframes')
async def get_timeframes():
    return {'timeframes': orchestrator.config['orchestrator']['analysis']['timeframes']}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
"""

if __name__ == '__main__':
    main()
