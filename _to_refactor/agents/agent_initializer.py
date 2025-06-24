# agent_initializer.py

from agents.agent_microstrategist import MicroStrategistAgent
from agents.agent_macroanalyser import MacroAnalyzerAgent
from agents.agent_riskmanager import RiskManagerAgent
from agents.agent_tradejournalist import TradeJournalistAgent
from agents.agent_htfanalyst import HTFAnalystAgent
from agents.agent_reputationauditor import ReputationAuditorAgent
from core.agent_semanticdss import SemanticDecisionSupportAgent


def initialize_agents(config):
    """Initialize all AI agents with the shared context from config."""
    shared_context = {
        "wyckoff_result": config.get("wyckoff_result"),
        "micro_context": config.get("micro_context"),
        "indicator_profiles": config.get("indicator_profiles"),
        "macro_snapshot": config.get("macro_snapshot"),
        "scalp_signal": config.get("scalp_signal"),
        "symbol": config.get("symbol"),
    }

    agents = {}
    if config.get("agents", {}).get("micro_strategist", {}).get("active", True):
        agents["micro_strategist"] = MicroStrategistAgent("micro", shared_context, None)
    if config.get("agents", {}).get("macro_analyzer", {}).get("active", True):
        agents["macro_analyzer"] = MacroAnalyzerAgent(shared_context)
    if config.get("agents", {}).get("risk_manager", {}).get("active", True):
        agents["risk_manager"] = RiskManagerAgent(context=shared_context)
    if config.get("agents", {}).get("trade_journalist", {}).get("active", True):
        agents["trade_journalist"] = TradeJournalistAgent(context=shared_context)
    if config.get("agents", {}).get("htf_phase_analyst", {}).get("active", True):
        agents["htf_phase_analyst"] = HTFAnalystAgent("htf", shared_context, None)
    if config.get("agents", {}).get("reputation_auditor", {}).get("active", True):
        agents["reputation_auditor"] = ReputationAuditorAgent(context=shared_context)
    if config.get("agents", {}).get("semantic_dss", {}).get("active", True):
        agents["semantic_dss"] = SemanticDecisionSupportAgent(context=shared_context)
    return agents
