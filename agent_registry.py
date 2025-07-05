# Agent registry module
"""Registry for runtime agents"""

from typing import Any, Dict
import inspect

# Public exports
__all__ = ["AgentRegistry", "AGENT_PROFILES"]

# Retain legacy agent profiles for backward compatibility
AGENT_PROFILES = {
    "Bożenka": {
        "role": "signal validator",
        "source": "M1/M5 Wyckoff CHoCH/Spring/BOS",
        "filters": ["spread", "tick_ret", "volume"],
        "trigger_logic": "evaluate_microstructure_phase_trigger",
        "active": True,
    },
    "Stefania": {
        "role": "reputation auditor / trust scoring",
        "source": "past entries + score feedback",
        "filters": ["phase", "trust_score", "journal"],
        "trigger_logic": "score_contextual_trust",
        "active": True,
    },
    "Lusia": {
        "role": "semantic confluence engine",
        "source": "indicators + DSS + divergence + config tags",
        "filters": ["DSS slope", "EMA polarity", "BB compression"],
        "trigger_logic": "evaluate_indicator_alignment",
        "active": True,
    },
    "Zdzisiek": {
        "role": "compliance/risk monitor",
        "source": "spread + volatility + tick clusters",
        "filters": ["ATR", "spread", "max risk"],
        "trigger_logic": "evaluate_risk_profile",
        "active": True,
    },
    "Rysiek": {
        "role": "HTF phase tracker",
        "source": "H1/H4 Wyckoff schematic + phase confidence",
        "filters": ["AR", "ST", "Spring", "LPS", "SOS"],
        "trigger_logic": "evaluate_wyckoff_phase_context",
        "active": True,
    },
}

class AgentRegistry:
    """Simple runtime registry for agent instances."""

    def __init__(self) -> None:
        self._agents: Dict[str, Any] = {}

    def register_agent(self, agent_id: str, agent: Any) -> None:
        """Register an agent instance under a unique identifier."""
        self._agents[agent_id] = agent

    def get_all_agents(self) -> Dict[str, Any]:
        """Return mapping of all registered agents."""
        return self._agents

    async def broadcast_event(self, event: Any) -> None:
        """Dispatch event to all agents implementing ``process_event``."""
        for agent in self._agents.values():
            handler = getattr(agent, "process_event", None)
            if handler is None:
                continue
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
