"""Minimal agent framework for dynamic registration."""
from typing import Callable, Dict, Type
from schema.models import Intent, AgentResult

# Global registry mapping agent keys to classes
agent_registry: Dict[str, Type["BaseAgent"]] = {}


def register_agent(agent_key: str) -> Callable[[Type["BaseAgent"]], Type["BaseAgent"]]:
    """Decorator to register an agent class under a specific key."""

    def decorator(cls: Type["BaseAgent"]) -> Type["BaseAgent"]:
        agent_registry[agent_key] = cls
        cls.agent_key = agent_key
        return cls

    return decorator


class BaseAgent:
    """Lightweight base class for pluggable agents."""

    framework_version: str = "1.0"
    agent_registry = agent_registry

    def handle_intent(self, intent: Intent, memory) -> AgentResult:  # pragma: no cover - abstract
        raise NotImplementedError
