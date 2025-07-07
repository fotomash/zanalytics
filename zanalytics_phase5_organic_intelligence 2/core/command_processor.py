"""
Command Processor - Integrates the Organic Intelligence Loop with existing Zanalytics components.
Processes commands from the queue and coordinates with the Action Dispatcher.
"""

import asyncio
import json
from typing import Dict, Any, Optional
import redis
from loguru import logger
from datetime import datetime

from core.dispatcher.action_dispatcher import get_dispatcher
from core.agents.scheduling_agent import SchedulingAgent
from core.agents.london_killzone_agent import LondonKillzoneAgentFactory


class CommandProcessor:
    """
    Main processor that runs continuously, processing commands from various sources:
    - LLM responses
    - Scheduled triggers
    - User actions
    - Market events
    """

    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.dispatcher = get_dispatcher(redis_client, config)
        self.running = False
        self.tasks = []

        # Agent registry
        self.agent_factories = {
            "LondonKillzone_SMC_v1": LondonKillzoneAgentFactory
        }

    async def start(self):
        """Start the command processor."""
        self.running = True
        logger.info("Starting Command Processor...")

        # Start processing tasks
        self.tasks.append(asyncio.create_task(self._process_command_queue()))
        self.tasks.append(asyncio.create_task(self._process_agent_tasks()))
        self.tasks.append(asyncio.create_task(self._process_scheduled_tasks()))

        logger.success("Command Processor started")

    async def stop(self):
        """Stop the command processor."""
        logger.info("Stopping Command Processor...")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)

        logger.info("Command Processor stopped")

    async def _process_command_queue(self):
        """Process commands from the main command queue."""
        while self.running:
            try:
                # Get command from queue (blocking pop with timeout)
                command_data = self.redis.brpop("zanalytics:command_queue", timeout=1)

                if command_data:
                    _, command_json = command_data
                    command = json.loads(command_json)

                    # Process through dispatcher
                    await self.dispatcher.process_command(command)

            except Exception as e:
                logger.error(f"Error processing command: {e}")
                await asyncio.sleep(1)

    async def _process_agent_tasks(self):
        """Process agent execution tasks."""
        while self.running:
            try:
                # Check each agent queue
                for agent_name, factory_class in self.agent_factories.items():
                    task_data = self.redis.rpop(f"zanalytics:agent_tasks:{agent_name}")

                    if task_data:
                        task = json.loads(task_data)
                        await self._execute_agent_task(agent_name, task, factory_class)

                await asyncio.sleep(0.5)  # Small delay between checks

            except Exception as e:
                logger.error(f"Error processing agent task: {e}")
                await asyncio.sleep(1)

    async def _execute_agent_task(self, agent_name: str, task: Dict[str, Any], factory_class):
        """Execute a specific agent task."""
        logger.info(f"Executing agent task: {agent_name} - {task['task_id']}")

        try:
            # Get manifest from task context
            manifest = task['context'].get('manifest', {})

            if not manifest:
                # Load manifest from disk
                manifest_path = f"knowledge/strategies/{agent_name}.yml"
                import yaml
                with open(manifest_path, 'r') as f:
                    manifest = yaml.safe_load(f)

            # Create agent instance
            agent = factory_class.create(self.redis, manifest)

            # Execute agent
            result = await agent.execute(task['context'])

            # Log result
            logger.success(f"Agent {agent_name} completed: {result.get('trade_ideas', [])} trade ideas found")

        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {e}")

    async def _process_scheduled_tasks(self):
        """Process scheduled follow-up tasks."""
        while self.running:
            try:
                # Get current timestamp
                now = datetime.utcnow().timestamp()

                # Get all scheduled tasks due for execution
                due_tasks = self.redis.zrangebyscore(
                    "zanalytics:scheduled_tasks",
                    0,
                    now,
                    withscores=False
                )

                for task_json in due_tasks:
                    task = json.loads(task_json)

                    # Create command from scheduled task
                    command = {
                        "request_id": f"scheduled_{task['id']}_{int(now)}",
                        "action_type": task['action'],
                        "payload": task['params'],
                        "metadata": {
                            "source": "scheduler",
                            "scheduled_id": task['id']
                        }
                    }

                    # Process command
                    await self.dispatcher.process_command(command)

                    # Remove from scheduled tasks
                    self.redis.zrem("zanalytics:scheduled_tasks", task_json)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error processing scheduled tasks: {e}")
                await asyncio.sleep(5)


class OrganicIntelligenceOrchestrator:
    """
    Main orchestrator that coordinates all components of the Organic Intelligence Loop.
    """

    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.command_processor = CommandProcessor(redis_client, config)
        self.scheduling_agent = SchedulingAgent(redis_client, config=config)

    async def start(self):
        """Start all components of the Organic Intelligence system."""
        logger.info("Starting Organic Intelligence Orchestrator...")

        # Initialize scheduling agent
        await self.scheduling_agent.initialize()

        # Start command processor
        await self.command_processor.start()

        logger.success("Organic Intelligence Orchestrator started successfully")

    async def stop(self):
        """Stop all components."""
        logger.info("Stopping Organic Intelligence Orchestrator...")

        # Stop scheduling agent
        await self.scheduling_agent.shutdown()

        # Stop command processor
        await self.command_processor.stop()

        logger.info("Organic Intelligence Orchestrator stopped")

    async def process_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Process a response from an LLM, extracting and executing any commands.

        Args:
            llm_response: Raw text response from LLM

        Returns:
            Processing result
        """
        try:
            # Try to parse as JSON first
            if llm_response.strip().startswith('{'):
                command = json.loads(llm_response)

                # Ensure it has required fields
                if all(key in command for key in ['request_id', 'action_type', 'payload']):
                    return await self.command_processor.dispatcher.process_command(command)

            # If not valid JSON or missing fields, extract commands from text
            commands = self._extract_commands_from_text(llm_response)

            results = []
            for command in commands:
                result = await self.command_processor.dispatcher.process_command(command)
                results.append(result)

            return {
                "status": "success",
                "commands_processed": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            return {"status": "error", "message": str(e)}

    def _extract_commands_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structured commands from unstructured text.
        This is a simplified version - in production, you might use NLP or regex patterns.
        """
        commands = []

        # Look for common command patterns
        if "trade setup" in text.lower() or "trade idea" in text.lower():
            # Extract trade information
            commands.append({
                "request_id": f"extracted_{datetime.utcnow().timestamp()}",
                "action_type": "LOG_JOURNAL_ENTRY",
                "payload": {
                    "type": "TradeIdea",
                    "source": "LLM_Extraction",
                    "content": text
                }
            })

        if "alert" in text.lower() or "notify" in text.lower():
            commands.append({
                "request_id": f"extracted_notify_{datetime.utcnow().timestamp()}",
                "action_type": "NOTIFY_USER",
                "payload": {
                    "level": "medium",
                    "message": text
                }
            })

        return commands
