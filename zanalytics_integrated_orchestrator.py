#!/usr/bin/env python3
"""
ZAnalytics Integrated Orchestrator
Main orchestration system that loads and coordinates all components
"""

import os
import sys
import json
import time
import logging
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import importlib.util
import traceback
import pandas as pd
import numpy as np
import asyncio
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComponentLoader:
    """Dynamically loads ZAnalytics components"""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.components = {}
        self.modules = {}

    def load_module(self, module_name: str, file_path: str) -> Optional[Any]:
        """Load a Python module from file"""
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.modules[module_name] = module
                logger.info(f"Loaded module: {module_name}")
                return module
        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {str(e)}")
            return None

    def load_all_components(self):
        """Load all ZAnalytics components"""
        component_map = {
            'data_pipeline': 'zanalytics_data_pipeline.py',
            'integration': 'zanalytics_integration.py',
            'signal_generator': 'zanalytics_signal_generator.py',
            'llm_formatter': 'zanalytics_llm_formatter.py',
            'dashboard': 'dashboard/app.py',
            'backtester': 'zanalytics_backtester.py',
            'advanced_analytics': 'zanalytics_advanced_analytics.py',
            'market_monitor': 'zanalytics_market_monitor.py',
            'llm_framework': 'zanalytics_llm_framework.py'
        }

        # Also try to load the original analyzers if they exist
        optional_components = {
            'microstructure_analyzer': 'zanflow_microstructure_analyzer.py',
            'ncOS_analyzer': 'ncOS_ultimate_microstructure_analyzer.py',
            'smc_converter': 'convert_final_enhanced_smc_ULTIMATE.py'
        }

        # Load required components
        for name, filename in component_map.items():
            file_path = self.base_path / filename
            if file_path.exists():
                self.load_module(name, str(file_path))
            else:
                logger.warning(f"Component file not found: {filename}")

        # Load optional components
        for name, filename in optional_components.items():
            file_path = self.base_path / filename
            if file_path.exists():
                self.load_module(name, str(file_path))


class ZAnalyticsIntegratedOrchestrator:
    """
    Integrated orchestrator that actually uses the loaded components
    """

    def __init__(self, config_path: str = "config/orchestrator_config.json"):
        self.config = self._load_config(config_path)
        self.component_loader = ComponentLoader()
        self.components = {}
        self.data_store = {}
        self.workflows = {}
        self.scheduler_thread = None
        self.running = False
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load orchestrator configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            logger.warning("Config file not found, using defaults")
            return {
                "update_interval": 300,
                "max_workers": 4,
                "symbols": ["BTC/USD", "ETH/USD"],
                "timeframes": ["1h", "4h", "1d"]
            }

    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing ZAnalytics Integrated Orchestrator...")

        # Load all component modules
        self.component_loader.load_all_components()

        # Initialize component instances where possible
        self._initialize_components()

        # Register workflows
        self._register_workflows()

        logger.info("Orchestrator initialized successfully")

    def _initialize_components(self):
        """Initialize component instances from loaded modules"""
        modules = self.component_loader.modules

        # Data Pipeline
        if 'data_pipeline' in modules:
            try:
                pipeline_class = getattr(modules['data_pipeline'], 'DataProcessingPipeline', None)
                if pipeline_class:
                    self.components['data_pipeline'] = pipeline_class()
                    logger.info("Initialized DataProcessingPipeline")
            except Exception as e:
                logger.error(f"Failed to initialize data pipeline: {e}")

        # Integration Hub
        if 'integration' in modules:
            try:
                integration_class = getattr(modules['integration'], 'ZAnalyticsIntegration', None)
                if integration_class:
                    self.components['integration'] = integration_class()
                    logger.info("Initialized ZAnalyticsIntegration")
            except Exception as e:
                logger.error(f"Failed to initialize integration: {e}")

        # Signal Generator
        if 'signal_generator' in modules:
            try:
                signal_class = getattr(modules['signal_generator'], 'TradingSignalGenerator', None)
                if signal_class:
                    self.components['signal_generator'] = signal_class()
                    logger.info("Initialized TradingSignalGenerator")
            except Exception as e:
                logger.error(f"Failed to initialize signal generator: {e}")

        # Advanced Analytics
        if 'advanced_analytics' in modules:
            try:
                analytics_class = getattr(modules['advanced_analytics'], 'AdvancedTradingAnalytics', None)
                if analytics_class:
                    self.components['advanced_analytics'] = analytics_class()
                    logger.info("Initialized AdvancedTradingAnalytics")
            except Exception as e:
                logger.error(f"Failed to initialize advanced analytics: {e}")

        # Market Monitor
        if 'market_monitor' in modules:
            try:
                monitor_class = getattr(modules['market_monitor'], 'MarketMonitor', None)
                if monitor_class:
                    self.components['market_monitor'] = monitor_class()
                    logger.info("Initialized MarketMonitor")
            except Exception as e:
                logger.error(f"Failed to initialize market monitor: {e}")

        # LLM Framework
        if 'llm_framework' in modules:
            try:
                llm_class = getattr(modules['llm_framework'], 'ZAnalyticsLLMFramework', None)
                if llm_class:
                    self.components['llm_framework'] = llm_class()
                    logger.info("Initialized ZAnalyticsLLMFramework")
            except Exception as e:
                logger.error(f"Failed to initialize LLM framework: {e}")

        # Register internal dashboard updater
        self.components['update_dashboard'] = self.update_dashboard

    def _register_workflows(self):
        """Register analysis workflows"""
        self.workflows = {
            'market_analysis': self._market_analysis_workflow,
            'comprehensive_analysis': self._comprehensive_analysis_workflow,
            'signal_generation': self._signal_generation_workflow,
            'risk_assessment': self._risk_assessment_workflow
        }

    def _market_analysis_workflow(self, symbol: str = "BTC/USD") -> Dict[str, Any]:
        """Execute market analysis workflow"""
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'status': 'started'
        }

        try:
            # Fetch market data
            if 'data_pipeline' in self.components:
                pipeline = self.components['data_pipeline']
                data = pipeline.fetch_market_data(symbol)
                if data is not None:
                    results['market_data'] = {
                        'rows': len(data),
                        'latest_price': float(data['close'].iloc[-1]) if len(data) > 0 else None
                    }
                    self.data_store[f'{symbol}_data'] = data

            # Run integration analysis
            if 'integration' in self.components and f'{symbol}_data' in self.data_store:
                integration = self.components['integration']
                analysis = integration.analyze(
                    self.data_store[f'{symbol}_data'],
                    symbol=symbol
                )
                results['integration_analysis'] = analysis

            # Generate signals
            if 'signal_generator' in self.components and f'{symbol}_data' in self.data_store:
                signal_gen = self.components['signal_generator']
                signals = signal_gen.generate_signals(
                    self.data_store[f'{symbol}_data']
                )
                results['signals'] = signals

            results['status'] = 'completed'

        except Exception as e:
            logger.error(f"Market analysis workflow failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)

        return results

    def _comprehensive_analysis_workflow(self) -> Dict[str, Any]:
        """Execute comprehensive multi-symbol analysis"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'symbols': self.config.get('symbols', ['BTC/USD'])
        }

        symbol_results = {}
        for symbol in results['symbols']:
            symbol_results[symbol] = self._market_analysis_workflow(symbol)

        results['symbol_analysis'] = symbol_results

        # Cross-market analysis if available
        if 'advanced_analytics' in self.components:
            try:
                analytics = self.components['advanced_analytics']
                # Prepare combined data
                all_data = {}
                for symbol in results['symbols']:
                    if f'{symbol}_data' in self.data_store:
                        all_data[symbol] = self.data_store[f'{symbol}_data']

                if all_data:
                    # Run correlation analysis
                    correlations = analytics.calculate_correlation_matrix(all_data)
                    results['correlations'] = correlations

            except Exception as e:
                logger.error(f"Cross-market analysis failed: {e}")

        return results

    def _signal_generation_workflow(self) -> Dict[str, Any]:
        """Generate trading signals for all symbols"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'signals': {}
        }

        for symbol in self.config.get('symbols', ['BTC/USD']):
            if f'{symbol}_data' in self.data_store and 'signal_generator' in self.components:
                try:
                    signal_gen = self.components['signal_generator']
                    signals = signal_gen.generate_signals(
                        self.data_store[f'{symbol}_data']
                    )
                    results['signals'][symbol] = signals
                except Exception as e:
                    logger.error(f"Signal generation failed for {symbol}: {e}")

        return results

    def _risk_assessment_workflow(self) -> Dict[str, Any]:
        """Assess portfolio risk"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': {}
        }

        if 'advanced_analytics' in self.components:
            try:
                analytics = self.components['advanced_analytics']

                for symbol in self.config.get('symbols', ['BTC/USD']):
                    if f'{symbol}_data' in self.data_store:
                        data = self.data_store[f'{symbol}_data']

                        # Calculate risk metrics
                        risk_metrics = analytics.calculate_risk_metrics(
                            data['close'].pct_change().dropna()
                        )
                        results['risk_metrics'][symbol] = risk_metrics

            except Exception as e:
                logger.error(f"Risk assessment failed: {e}")

        return results

    async def update_dashboard(self, data: Dict[str, Any]) -> None:
        """Forward results to the dashboard websocket API"""
        ws_url = self.config.get("dashboard_ws_url", "ws://localhost:5010/ws")
        message = {
            "type": "analysis_update",
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            async with websockets.connect(ws_url) as websocket:
                await websocket.send(json.dumps(message))
        except Exception as e:
            logger.warning(f"Dashboard update failed: {e}")

    def run_workflow(self, workflow_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a named workflow"""
        if workflow_name not in self.workflows:
            return {'error': f'Unknown workflow: {workflow_name}'}

        try:
            result = self.workflows[workflow_name](**kwargs)

            # Save results
            result_file = self.results_dir / f"{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            # Forward results to dashboard if enabled
            if self.config.get('dashboard_enabled', False):
                try:
                    asyncio.run(self.update_dashboard(result))
                except Exception as e:
                    logger.warning(f"Failed to push results to dashboard: {e}")

            return result

        except Exception as e:
            logger.error(f"Workflow {workflow_name} failed: {e}")
            return {'error': str(e), 'traceback': traceback.format_exc()}

    def schedule_tasks(self):
        """Schedule periodic tasks"""
        # Market analysis every 5 minutes
        schedule.every(5).minutes.do(
            lambda: self.run_workflow('market_analysis')
        )

        # Comprehensive analysis every hour
        schedule.every().hour.do(
            lambda: self.run_workflow('comprehensive_analysis')
        )

        # Risk assessment every 30 minutes
        schedule.every(30).minutes.do(
            lambda: self.run_workflow('risk_assessment')
        )

    def _run_scheduler(self):
        """Run the scheduler in a separate thread"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)

    def start(self):
        """Start the orchestrator"""
        self.running = True

        # Schedule tasks
        self.schedule_tasks()

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        logger.info("Orchestrator started")

        # Run initial analysis
        self.run_workflow('comprehensive_analysis')

    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Orchestrator stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'running': self.running,
            'components': list(self.components.keys()),
            'loaded_modules': list(self.component_loader.modules.keys()),
            'workflows': list(self.workflows.keys()),
            'data_store_keys': list(self.data_store.keys()),
            'scheduled_jobs': [str(job) for job in schedule.jobs]
        }


def main():
    """Main entry point"""
    orchestrator = ZAnalyticsIntegratedOrchestrator()

    try:
        # Initialize
        orchestrator.initialize()

        # Start orchestrator
        orchestrator.start()

        # Print status
        print("\nOrchestrator Status:")
        print(json.dumps(orchestrator.get_status(), indent=2))

        print("\nOrchestrator is running. Press Ctrl+C to stop.")

        # Keep running
        while True:
            time.sleep(60)
            # Print status every minute
            status = orchestrator.get_status()
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Active components: {len(status['components'])}")

    except KeyboardInterrupt:
        print("\nStopping orchestrator...")
        orchestrator.stop()
        print("Orchestrator stopped successfully")

    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
