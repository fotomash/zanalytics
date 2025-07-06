# zanalytics_adapter.py - Bridge between Data Flow and ZANALYTICS Agents
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

# Import ZANALYTICS components (adjust imports based on your structure)
try:
    from agent_initializer import initialize_agents
    from agent_microstrategist import MicroStrategistAgent
    from agent_macroanalyser import MacroAnalyzerAgent
    from agent_riskmanager import RiskManagerAgent
    from agent_tradejournalist import TradeJournalistAgent
    from core.orchestrator import AnalysisOrchestrator
except ImportError as e:
    logging.warning(f"ZANALYTICS import warning: {e}")

from data_flow_manager import DataFlowManager, DataFlowEvent

class ZAnalyticsDataBridge:
    """
    Bridges real-time data flow with ZANALYTICS agents
    Makes your app 'aware' of incoming data and triggers appropriate analysis
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "zanalytics_config.json"
        self.config = self._load_config(config_path)
        self.agents = {}
        self.active_symbols = set()
        self.data_flow_manager = None
        self.smc_orchestrator = None
        self.analysis_cache = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._initialize_components()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration for the bridge"""
        default_config = {
            "watch_directories": ["./", "./data", "./exports"],
            "agents": {
                "micro_strategist": {"active": True, "symbols": ["EURUSD", "GBPUSD", "XAUUSD"]},
                "macro_analyzer": {"active": True, "symbols": ["EURUSD", "GBPUSD", "XAUUSD"]},
                "risk_manager": {"active": True, "symbols": ["EURUSD", "GBPUSD", "XAUUSD"]},
                "trade_journalist": {"active": True, "symbols": ["EURUSD", "GBPUSD", "XAUUSD"]},
                "smc_orchestrator": {"active": True, "symbols": ["EURUSD", "GBPUSD", "XAUUSD"]}
            },
            "data_triggers": {
                "new_csv_data": ["process_ohlc", "update_indicators"],
                "new_analysis_data": ["update_agents", "generate_signals"],
                "tick_data": ["microstructure_analysis", "spread_analysis"]
            },
            "real_time": {
                "enable": True,
                "update_interval": 1.0,  # seconds
                "batch_size": 100
            }
        }

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            self.logger.info(f"Config file {config_path} not found, using defaults")
            return default_config

    def _initialize_components(self):
        """Initialize ZANALYTICS components"""
        try:
            # Initialize agents
            self.agents = initialize_agents(self.config)
            self.logger.info("ZANALYTICS agents initialized")

            # Initialize SMC orchestrator if enabled
            if self.config.get("agents", {}).get("smc_orchestrator", {}).get("active", False):
                self.smc_orchestrator = AnalysisOrchestrator()
                self.logger.info("SMC Orchestrator initialized")

        except Exception as e:
            self.logger.error(f"Error initializing ZANALYTICS components: {e}")

    async def start_data_awareness(self):
        """Start the data flow awareness system"""
        # Initialize data flow manager
        self.data_flow_manager = DataFlowManager(
            watch_directories=self.config["watch_directories"],
            zanalytics_callback=self._on_data_flow_event
        )

        # Start monitoring
        self.data_flow_manager.start_monitoring()

        # Subscribe agents to relevant data streams
        self._setup_agent_subscriptions()

        self.logger.info("🚀 ZANALYTICS Data Awareness System STARTED")
        self.logger.info(f"📁 Monitoring directories: {self.config['watch_directories']}")
        self.logger.info(f"🤖 Active agents: {list(self.agents.keys())}")

        # Keep the system running
        try:
            while True:
                await asyncio.sleep(1)
                await self._periodic_health_check()
        except KeyboardInterrupt:
            await self.stop_data_awareness()

    async def stop_data_awareness(self):
        """Stop the data awareness system"""
        if self.data_flow_manager:
            self.data_flow_manager.stop_monitoring()
        self.logger.info("🛑 ZANALYTICS Data Awareness System STOPPED")

    def _setup_agent_subscriptions(self):
        """Subscribe agents to relevant data streams"""
        for agent_name, agent_config in self.config.get("agents", {}).items():
            if agent_config.get("active", False):
                symbols = agent_config.get("symbols", [])
                for symbol in symbols:
                    # Subscribe to multiple timeframes
                    for timeframe in ["M1", "M5", "M15", "H1", "H4", "D1", "TICK"]:
                        self.data_flow_manager.subscribe_agent(
                            symbol, timeframe, 
                            lambda event, agent=agent_name: asyncio.create_task(
                                self._notify_agent(agent, event)
                            )
                        )

    async def _on_data_flow_event(self, event: DataFlowEvent):
        """Handle data flow events and trigger appropriate analysis"""
        self.logger.info(f"📊 Data Event: {event.symbol} {event.timeframe} - {event.event_type}")

        # Add to active symbols
        self.active_symbols.add(event.symbol)

        # Trigger appropriate actions based on event type
        triggers = self.config.get("data_triggers", {}).get(event.event_type, [])

        for trigger in triggers:
            try:
                await self._execute_trigger(trigger, event)
            except Exception as e:
                self.logger.error(f"Error executing trigger {trigger}: {e}")

    async def _execute_trigger(self, trigger: str, event: DataFlowEvent):
        """Execute specific trigger actions"""
        if trigger == "process_ohlc":
            await self._process_ohlc_data(event)
        elif trigger == "update_indicators":
            await self._update_indicators(event)
        elif trigger == "update_agents":
            await self._update_all_agents(event)
        elif trigger == "generate_signals":
            await self._generate_signals(event)
        elif trigger == "microstructure_analysis":
            await self._microstructure_analysis(event)
        elif trigger == "spread_analysis":
            await self._spread_analysis(event)

    async def _process_ohlc_data(self, event: DataFlowEvent):
        """Process OHLC data through ZANALYTICS"""
        if event.source == 'csv' and event.file_path:
            try:
                # Load the data
                df = pd.read_csv(event.file_path)

                # Create analysis context
                context = {
                    "symbol": event.symbol,
                    "timeframe": event.timeframe,
                    "data": df,
                    "timestamp": event.timestamp,
                    "source": "real_time_csv"
                }

                # Run SMC analysis if orchestrator is available
                if self.smc_orchestrator:
                    smc_result = await self._run_smc_analysis(context)
                    context["smc_analysis"] = smc_result

                # Cache the analysis
                cache_key = f"{event.symbol}_{event.timeframe}"
                self.analysis_cache[cache_key] = context

                self.logger.info(f"✅ Processed OHLC data for {event.symbol} {event.timeframe}")

            except Exception as e:
                self.logger.error(f"Error processing OHLC data: {e}")

    async def _run_smc_analysis(self, context: Dict) -> Dict:
        """Run SMC analysis on the data"""
        try:
            if self.smc_orchestrator:
                result = self.smc_orchestrator.analyze_market_structure(
                    context["data"], 
                    context["symbol"], 
                    context["timeframe"]
                )
                return result
        except Exception as e:
            self.logger.error(f"SMC analysis error: {e}")
        return {}

    async def _update_indicators(self, event: DataFlowEvent):
        """Update technical indicators"""
        # This would integrate with your indicator calculation system
        self.logger.info(f"🔄 Updating indicators for {event.symbol} {event.timeframe}")

    async def _update_all_agents(self, event: DataFlowEvent):
        """Update all active agents with new data"""
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'update_data'):
                    await agent.update_data(event)
                self.logger.info(f"🤖 Updated {agent_name} with new data")
            except Exception as e:
                self.logger.error(f"Error updating {agent_name}: {e}")

    async def _generate_signals(self, event: DataFlowEvent):
        """Generate trading signals based on new analysis"""
        cache_key = f"{event.symbol}_{event.timeframe}"
        if cache_key in self.analysis_cache:
            context = self.analysis_cache[cache_key]

            # Generate signal summary
            signal_summary = {
                "symbol": event.symbol,
                "timeframe": event.timeframe,
                "timestamp": datetime.now().isoformat(),
                "analysis_available": True,
                "smc_signals": context.get("smc_analysis", {}),
                "data_quality": "good" if len(context.get("data", [])) > 100 else "limited"
            }

            # Save signal to file
            signal_file = f"signals_{event.symbol}_{event.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(signal_file, 'w') as f:
                json.dump(signal_summary, f, indent=2)

            self.logger.info(f"📈 Generated signals for {event.symbol} {event.timeframe}")

    async def _microstructure_analysis(self, event: DataFlowEvent):
        """Perform microstructure analysis on tick data"""
        if event.timeframe == "TICK" and event.file_path:
            try:
                # Load tick data
                df = pd.read_csv(event.file_path)

                # Basic microstructure metrics
                if 'bid' in df.columns and 'ask' in df.columns:
                    df['spread'] = df['ask'] - df['bid']
                    df['mid_price'] = (df['bid'] + df['ask']) / 2

                    microstructure_metrics = {
                        "avg_spread": df['spread'].mean(),
                        "spread_volatility": df['spread'].std(),
                        "tick_count": len(df),
                        "price_range": df['mid_price'].max() - df['mid_price'].min(),
                        "analysis_timestamp": datetime.now().isoformat()
                    }

                    # Cache microstructure data
                    cache_key = f"{event.symbol}_microstructure"
                    self.analysis_cache[cache_key] = microstructure_metrics

                    self.logger.info(f"🔬 Microstructure analysis completed for {event.symbol}")

            except Exception as e:
                self.logger.error(f"Microstructure analysis error: {e}")

    async def _spread_analysis(self, event: DataFlowEvent):
        """Analyze spread patterns"""
        self.logger.info(f"📊 Spread analysis for {event.symbol}")

    async def _notify_agent(self, agent_name: str, event: DataFlowEvent):
        """Notify specific agent about data update"""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            try:
                if hasattr(agent, 'on_data_update'):
                    await agent.on_data_update(event)
                self.logger.debug(f"Notified {agent_name} about {event.symbol} update")
            except Exception as e:
                self.logger.error(f"Error notifying {agent_name}: {e}")

    async def _periodic_health_check(self):
        """Periodic health check of the system"""
        # Check data flow status
        if self.data_flow_manager:
            status = self.data_flow_manager.get_data_status()
            if status['active_streams'] > 0:
                self.logger.debug(f"💗 Health check: {status['active_streams']} active data streams")

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "active_symbols": list(self.active_symbols),
            "active_agents": list(self.agents.keys()),
            "analysis_cache_size": len(self.analysis_cache),
            "data_flow_status": self.data_flow_manager.get_data_status() if self.data_flow_manager else None
        }
        return status

# Main execution function
async def main():
    """Main function to start ZANALYTICS data awareness"""
    bridge = ZAnalyticsDataBridge()
    await bridge.start_data_awareness()

if __name__ == "__main__":
    asyncio.run(main())
