# Fixed agent_initializer.py
# Replace the entire contents of this file

import logging
# Fix imports - remove 'agents.' prefix since files are in root
from agent_microstrategist import MicroStrategistAgent
from agent_macroanalyser import MacroAnalyzerAgent
from agent_riskmanager import RiskManagerAgent
from agent_tradejournalist import TradeJournalistAgent
try:
    from advanced_smc_orchestrator import AdvancedSMCOrchestrator
except ImportError:
    AdvancedSMCOrchestrator = None

logger = logging.getLogger(__name__)

def initialize_agents(registry, config):
    """Initialize all configured agents"""
    agent_config = config.get('agents', {})
    
    # Initialize Micro Strategist
    if agent_config.get('micro_strategist', {}).get('active', False):
        micro = MicroStrategistAgent()
        for symbol in agent_config['micro_strategist'].get('symbols', []):
            registry.register_agent(f"micro_{symbol}", micro)
        logger.info("MicroStrategist agents initialized")
    
    # Initialize Macro Analyzer
    if agent_config.get('macro_analyzer', {}).get('active', False):
        macro = MacroAnalyzerAgent()
        for symbol in agent_config['macro_analyzer'].get('symbols', []):
            registry.register_agent(f"macro_{symbol}", macro)
        logger.info("MacroAnalyzer agents initialized")
    
    # Initialize Risk Manager
    if agent_config.get('risk_manager', {}).get('active', False):
        risk = RiskManagerAgent()
        for symbol in agent_config['risk_manager'].get('symbols', []):
            registry.register_agent(f"risk_{symbol}", risk)
        logger.info("RiskManager agents initialized")
    
    # Initialize Trade Journalist
    if agent_config.get('trade_journalist', {}).get('active', False):
        journalist = TradeJournalistAgent()
        for symbol in agent_config['trade_journalist'].get('symbols', []):
            registry.register_agent(f"journal_{symbol}", journalist)
        logger.info("TradeJournalist agents initialized")
    
    # Initialize SMC Orchestrator if available
    if AdvancedSMCOrchestrator and agent_config.get('smc_orchestrator', {}).get('active', False):
        smc = AdvancedSMCOrchestrator()
        for symbol in agent_config['smc_orchestrator'].get('symbols', []):
            registry.register_agent(f"smc_{symbol}", smc)
        logger.info("SMC Orchestrator agents initialized")
