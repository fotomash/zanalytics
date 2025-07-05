#!/usr/bin/env python3
"""
NCOS v24 - Multi-Agent Orchestrator
The primary coordinator for all agent activities. This module manages agent
registration, task queuing, and selects the optimal execution strategy
(e.g., parallel, collaborative) for incoming tasks.
"""

import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger
from collections import deque

from ncos.core.base import BaseComponent, AgentResponse
from ncos.core.memory.manager import MemoryManager
# Import specific strategies and agent base class when they are created
# from ncos.core.strategies.adaptive import AdaptiveStrategy
# from ncos.agents.base_agent import BaseAgent

class Task:
    """A data class representing a single unit of work for an agent."""
    def __init__(self, task_id: str, task_type: str, data: Any, priority: int = 5):
        self.task_id = task_id
        self.task_type = task_type
        self.data = data
        self.priority = priority
        self.status = "pending"
        self.result: Optional[AgentResponse] = None

class MultiAgentOrchestrator(BaseComponent):
    """
    Orchestrates task execution across a dynamic pool of agents, using an
    adaptive strategy to choose the best execution method.
    """

    def __init__(self, config: Dict[str, Any], memory_manager: MemoryManager):
        super().__init__(config)
        self.memory_manager = memory_manager
        self.agents: Dict[str, Any] = {}  # Will hold BaseAgent instances
        self.task_queue = asyncio.Queue()
        self.execution_strategy = None # Will hold an AdaptiveStrategy instance

    async def initialize(self) -> bool:
        """
        Initializes the orchestrator, loads agents, and sets up the
        adaptive execution strategy.
        """
        self.logger.info("Initializing Multi-Agent Orchestrator...")
        
        # In a real implementation, you would dynamically load agent plugins here
        # For now, we'll assume agents are registered manually.
        await self._load_agents_from_config()
        
        # Initialize the adaptive execution strategy
        # self.execution_strategy = AdaptiveStrategy(self)
        # self.logger.info(f"Execution strategy loaded: {self.execution_strategy.__class__.__name__}")

        self.is_initialized = True
        self.logger.success("Multi-Agent Orchestrator is operational.")
        return True

    async def _load_agents_from_config(self):
        """Loads and registers agents defined in the configuration."""
        agents_config = self.config.get("agents", [])
        self.logger.info(f"Found {len(agents_config)} agent configurations to load.")
        # This is a placeholder for a more robust plugin loading system.
        # For each agent config, you would import its class, instantiate it,
        # and register it.
        #
        # from ncos.agents.some_agent import SomeAgent
        # for agent_conf in agents_config:
        #     agent_instance = SomeAgent(agent_conf)
        #     await self.register_agent(agent_instance)
        pass

    async def register_agent(self, agent: Any): # Replace Any with BaseAgent later
        """
        Registers a new agent with the orchestrator, making it available for tasks.
        
        Args:
            agent: An instance of a class that inherits from BaseAgent.
        """
        if agent.agent_id in self.agents:
            self.logger.warning(f"Agent '{agent.agent_id}' is already registered. Overwriting.")
        
        await agent.initialize()
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Agent '{agent.agent_id}' of type '{agent.__class__.__name__}' has been registered.")

    async def process(self, data: Dict[str, Any]) -> Any:
        """
        Main entry point for handling incoming tasks. It determines the correct
        agents and execution strategy for the task.
        """
        if not self.is_initialized:
            raise RuntimeError("Orchestrator is not initialized.")

        task_type = data.get("task_type", "general")
        task_data = data.get("payload", {})
        
        # 1. Identify agents capable of handling this task
        capable_agents = self._find_capable_agents(task_type)
        if not capable_agents:
            msg = f"No capable agents found for task type: '{task_type}'"
            self.logger.error(msg)
            return {"status": "failed", "error": msg}

        # 2. Use the adaptive strategy to execute the task
        # This is a placeholder for the strategy execution logic
        self.logger.info(f"Found {len(capable_agents)} agents for task '{task_type}'. Delegating to execution strategy.")
        
        # In the full implementation:
        # return await self.execution_strategy.execute(data, capable_agents)

        # For now, we'll just process with the first capable agent as a fallback
        first_agent = self.agents[capable_agents[0]]
        return await first_agent.process(task_data)

    def _find_capable_agents(self, task_type: str) -> List[str]:
        """Finds all registered agents that can handle a given task type."""
        capable = []
        for agent_id, agent in self.agents.items():
            if agent.can_handle(task_type):
                capable.append(agent_id)
        return capable

    async def get_system_status(self) -> Dict[str, Any]:
        """Gathers and returns the status of the orchestrator and its agents."""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_status()

        return {
            "orchestrator_status": "running" if self.is_initialized else "stopped",
            "registered_agents_count": len(self.agents),
            "pending_tasks": self.task_queue.qsize(),
            "agent_statuses": agent_statuses
        }

    async def cleanup(self):
        """Cleans up all registered agents."""
        self.logger.info("Cleaning up registered agents...")
        for agent in self.agents.values():
            await agent.cleanup()
        await super().cleanup()
