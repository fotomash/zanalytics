# __init__.py – ZANALYTICS root package initializer (V10)
# CTO-ready: exposes core AI agent interface layer for modular orchestrators

__version__ = "5.2.1"
__author__ = "ZANALYTICS AI Stack"

import logging
logging.getLogger(__name__).info("ZANALYTICS package initialized.")

# Expose core agent APIs
from .agent_initializer import initialize_agents
from .agent_microstrategist import MicroStrategistAgent
from .agent_macroanalyser import MacroAnalyzerAgent
from .agent_riskmanager import RiskManagerAgent
from .agent_tradejournalist import TradeJournalistAgent
from .agent_wyckoffspecialist import WyckoffSpecialistAgent
from .agent_semanticdss import SemanticDecisionSupportAgent

__all__ = [
    "initialize_agents",
    "MicroStrategistAgent",
    "MacroAnalyzerAgent",
    "RiskManagerAgent",
    "TradeJournalistAgent",
    "WyckoffSpecialistAgent",
    "SemanticDecisionSupportAgent"
]

# YAML-first orchestration:
# All agents above are controlled via YAML profiles located in /profiles/.
# This supports declarative configuration for deterministic and testable execution logic across V10/V11.
