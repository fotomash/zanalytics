# BaseAgent

The `BaseAgent` abstract class defines the core contract and common utilities for all agents in the Copilot Framework.

## Overview

All agent modules should subclass `BaseAgent` and register via `@register_agent`. This pattern guarantees automatic discovery, routing, and consistent integration across domains.

## Class Definition

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
from core.user_memory import UserMemory
from core.models import Intent, AgentResult

class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Version field helps with compatibility management.
    """

    def __init__(self, name: str, version: str = "1.0"):
        """
        Initialize the agent with a unique name identifier.
        """
        self.name = name
        self.version = version

    @abstractmethod
    def handle_intent(self, intent: Intent, memory: UserMemory) -> AgentResult:
        """
        Process the incoming Intent and current UserMemory.
        
        Args:
            intent (Intent): Structured intent payload.
            memory (UserMemory): User-specific contextual memory.

        Returns:
            AgentResult: Result containing response, memory updates, and any triggers.
        """
        pass
```

## Registration

Agents must register themselves to be discoverable by the orchestrator:

```python
from core.agent import BaseAgent, register_agent

@register_agent("example")
class SampleAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="example")
        self.secure = True  # enforce secure mode by default

    def handle_intent(self, intent: Intent, memory: UserMemory) -> AgentResult:
        try:
            # Custom logic here
            return AgentResult(
                message="Processed by SampleAgent",
                updates={},
                triggers=[]
            )
        except Exception as e:
            return AgentResult(
                message=f"Error in {self.name}: {str(e)}",
                updates={},
                triggers=[]
            )
```

The decorator:
- Adds the agent class to the global registry
- Associates the `business_type` or `agent_key` with this handler

## Best Practices

- Follow SOLID principles to keep logic maintainable and testable.
- Keep each agent’s logic focused and domain-specific.
- Use the shared `UserMemory` interface to persist or retrieve user context.
- Emit structured `AgentResult` objects for consistent downstream processing.
- Include error handling and validation in each agent.
- Implement schema validation early in `handle_intent`.
- Leverage the built-in logging hooks for traceability.

## Extending BaseAgent

Consider extending `BaseAgent` with common utilities to streamline agent development and ensure consistency, such as:
- Message templating engines
- Input validation decorators
- Structured logging and metrics collectors
- Authentication/authorization checks
