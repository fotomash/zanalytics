# __init__.py – ZANALYTICS root package initializer (V10)
# CTO-ready: exposes core AI agent interface layer for modular orchestrators

__version__ = "5.2.1"
__author__ = "ZANALYTICS AI Stack"

import logging
import os
from .logger import setup_logging

# Initialize logging. In lightweight test mode we avoid loading the JSON
# configuration and simply set up basic logging to STDOUT.
if os.environ.get("ZANALYTICS_TEST_MODE") == "1":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    )
else:
    setup_logging()
    logging.getLogger(__name__).info("ZANALYTICS package initialized.")

# During lightweight test runs we avoid importing heavy modules that may
# introduce additional dependencies. Set `ZANALYTICS_TEST_MODE=1` in the
# environment to enable this behavior.
if os.environ.get("ZANALYTICS_TEST_MODE") != "1":
    try:
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
            "SemanticDecisionSupportAgent",
        ]
    except Exception:
        __all__ = []
else:
    __all__ = []


# YAML-first orchestration:
# All agents above are controlled via YAML profiles located in /profiles/.
# This supports declarative configuration for deterministic and testable execution logic across V10/V11.
