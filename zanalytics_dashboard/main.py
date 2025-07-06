#!/usr/bin/env python3
"""
🚀 Ultimate Trading System - Main Launcher
Coordinates all components of the trading ecosystem
"""

import asyncio
import threading
import time
import logging
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystemOrchestrator:
    """Main system coordinator that manages all components"""

    def __init__(self):
        self.components = {}
        self.running = False
        self.config = self.load_config()

    def load_config(self):
        """Load system configuration"""
        try:
            with open('config/settings.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.create_default_config()

    def create_default_config(self):
        """Create default configuration"""
        config = {
            "data_pipeline": {
                "refresh_interval": 60,
                "data_path": "./data",
                "timeframes": ["1M", "5M", "15M", "1H", "4H", "1D"]
            },
            "api_server": {
                "websocket_port": 8765,
                "rest_port": 8080,
                "host": "localhost"
            },
            "dashboard": {
                "port": 8050,
                "debug": True,
                "auto_refresh": 5
            },
            "zanalytics": {
                "enabled": True,
                "agents": ["htf", "micro", "risk", "macro"],
                "analysis_interval": 300
            },
            "trading": {
                "paper_trading": True,
                "risk_per_trade": 0.02,
                "max_positions": 5
            }
        }

        # Save default config
        Path('config').mkdir(exist_ok=True)
        with open('config/settings.json', 'w') as f:
            json.dump(config, f, indent=2)

        return config

    async def start_data_pipeline(self):
        """Start the data enrichment pipeline"""
        logger.info("🔄 Starting Data Pipeline...")

        from data_pipeline.data_enricher import DataEnricher
        enricher = DataEnricher(self.config['data_pipeline'])

        # Run enrichment in background
        def run_enrichment():
            while self.running:
                try:
                    enricher.process_all_data()
                    logger.info("✅ Data enrichment cycle completed")
                    time.sleep(self.config['data_pipeline']['refresh_interval'])
                except Exception as e:
                    logger.error(f"❌ Data pipeline error: {e}")
                    time.sleep(30)

        thread = threading.Thread(target=run_enrichment, daemon=True)
        thread.start()
        self.components['data_pipeline'] = thread

    async def start_api_server(self):
        """Start WebSocket and REST API servers"""
        logger.info("🌐 Starting API Server...")

        from api_server.websocket_server import WebSocketServer
        from api_server.rest_endpoints import RestAPI

        # Start WebSocket server
        ws_server = WebSocketServer(
            host=self.config['api_server']['host'],
            port=self.config['api_server']['websocket_port']
        )

        # Start REST API
        rest_api = RestAPI(
            host=self.config['api_server']['host'],
            port=self.config['api_server']['rest_port']
        )

        # Run servers
        ws_task = asyncio.create_task(ws_server.start())
        rest_task = asyncio.create_task(rest_api.start())

        self.components['websocket'] = ws_task
        self.components['rest_api'] = rest_task

    async def start_zanalytics(self):
        """Start zAnalytics AI agents"""
        if not self.config['zanalytics']['enabled']:
            return

        logger.info("🧠 Starting zAnalytics...")

        from zanalytics.zanalytics_adapter import ZAnalyticsSystem

        za_system = ZAnalyticsSystem(self.config['zanalytics'])

        def run_agents():
            while self.running:
                try:
                    za_system.run_analysis_cycle()
                    logger.info("🤖 zAnalytics cycle completed")
                    time.sleep(self.config['zanalytics']['analysis_interval'])
                except Exception as e:
                    logger.error(f"❌ zAnalytics error: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=run_agents, daemon=True)
        thread.start()
        self.components['zanalytics'] = thread

    async def start_dashboard(self):
        """Start the web dashboard"""
        logger.info("📊 Starting Dashboard...")

        from dashboard.app import create_dash_app

        app = create_dash_app(self.config['dashboard'])

        def run_dashboard():
            app.run_server(
                host='0.0.0.0',
                port=self.config['dashboard']['port'],
                debug=self.config['dashboard']['debug']
            )

        thread = threading.Thread(target=run_dashboard, daemon=True)
        thread.start()
        self.components['dashboard'] = thread

    async def start_trading_engine(self):
        """Start the trading signal generator and risk manager"""
        logger.info("🤖 Starting Trading Engine...")

        from trading_engine.signal_generator import SignalGenerator
        from trading_engine.risk_manager import RiskManager

        signal_gen = SignalGenerator(self.config['trading'])
        risk_mgr = RiskManager(self.config['trading'])

        def run_trading():
            while self.running:
                try:
                    # Generate signals
                    signals = signal_gen.generate_signals()

                    # Apply risk management
                    managed_signals = risk_mgr.process_signals(signals)

                    # Broadcast signals via WebSocket
                    self.broadcast_signals(managed_signals)

                    logger.info(f"📡 Generated {len(managed_signals)} trading signals")
                    time.sleep(60)  # Check every minute

                except Exception as e:
                    logger.error(f"❌ Trading engine error: {e}")
                    time.sleep(30)

        thread = threading.Thread(target=run_trading, daemon=True)
        thread.start()
        self.components['trading_engine'] = thread

    def broadcast_signals(self, signals):
        """Broadcast trading signals to all connected clients"""
        # This would send signals via WebSocket to dashboard and other clients
        pass

    async def start_all_components(self):
        """Start all system components"""
        logger.info("🚀 Starting Ultimate Trading System...")

        # Create necessary directories
        for dir_name in ['logs', 'data/processed', 'data/cache', 'config']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

        self.running = True

        # Start components in order
        await self.start_data_pipeline()
        await self.start_api_server()
        await self.start_zanalytics()
        await self.start_trading_engine()
        await self.start_dashboard()

        logger.info("✅ All components started successfully!")
        logger.info(f"📊 Dashboard: http://localhost:{self.config['dashboard']['port']}")
        logger.info(f"🌐 WebSocket: ws://localhost:{self.config['api_server']['websocket_port']}")
        logger.info(f"🔗 REST API: http://localhost:{self.config['api_server']['rest_port']}")

    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("🛑 Shutting down system...")
        self.running = False

        # Cancel async tasks
        for name, component in self.components.items():
            if hasattr(component, 'cancel'):
                component.cancel()
                logger.info(f"✅ Stopped {name}")

        logger.info("🔴 System shutdown complete")

async def main():
    """Main entry point"""
    orchestrator = TradingSystemOrchestrator()

    try:
        await orchestrator.start_all_components()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal...")
        await orchestrator.shutdown()
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        await orchestrator.shutdown()

if __name__ == "__main__":
    print("🚀 Ultimate Trading System")
    print("="*50)
    print("Starting all components...")
    print("Press Ctrl+C to shutdown")
    print("="*50)

    asyncio.run(main())
