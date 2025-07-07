"""
Scheduling Agent - The system's autonomous timekeeper.
Monitors time and market conditions to trigger strategies and analyses.
"""

import asyncio
import json
import yaml
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import redis
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

class SchedulingAgent:
    """
    Master scheduling agent that reads strategy manifests and triggers
    other agents based on their defined schedules and conditions.
    """

    def __init__(self, 
                 redis_client: redis.Redis,
                 manifest_dir: str = "knowledge/strategies",
                 config: Optional[Dict[str, Any]] = None):
        self.redis = redis_client
        self.manifest_dir = Path(manifest_dir)
        self.config = config or {}
        self.scheduler = AsyncIOScheduler(timezone=timezone.utc)
        self.active_strategies: Dict[str, Dict] = {}
        self.running_agents: Dict[str, asyncio.Task] = {}

    async def initialize(self):
        """Initialize the scheduling agent and load all strategies."""
        logger.info("Initializing Scheduling Agent...")

        # Load all strategy manifests
        await self._load_strategy_manifests()

        # Set up schedules
        self._setup_schedules()

        # Start the scheduler
        self.scheduler.start()

        logger.success(f"Scheduling Agent initialized with {len(self.active_strategies)} active strategies")

    async def _load_strategy_manifests(self):
        """Load all strategy manifests from the manifest directory."""
        if not self.manifest_dir.exists():
            logger.warning(f"Manifest directory not found: {self.manifest_dir}")
            return

        for manifest_file in self.manifest_dir.glob("*.yml"):
            try:
                with open(manifest_file, 'r') as f:
                    manifest = yaml.safe_load(f)

                if manifest.get("enabled", False):
                    strategy_id = manifest["strategy_id"]
                    self.active_strategies[strategy_id] = manifest
                    logger.info(f"Loaded strategy: {strategy_id}")

            except Exception as e:
                logger.error(f"Failed to load manifest {manifest_file}: {e}")

    def _setup_schedules(self):
        """Set up scheduled jobs for all active strategies."""
        for strategy_id, manifest in self.active_strategies.items():
            schedule_config = manifest.get("schedule", {})

            if schedule_config.get("trigger_type") == "time_based":
                self._setup_time_based_schedule(strategy_id, schedule_config)
            elif schedule_config.get("trigger_type") == "condition_based":
                self._setup_condition_based_schedule(strategy_id, schedule_config)

    def _setup_time_based_schedule(self, strategy_id: str, schedule: Dict[str, Any]):
        """Set up time-based scheduling for a strategy."""
        time_window = schedule.get("utc_time_window", {})
        start_time = time_window.get("start", "00:00")
        end_time = time_window.get("end", "23:59")
        frequency = schedule.get("frequency", "every_1_minute")
        days = schedule.get("days", ["monday", "tuesday", "wednesday", "thursday", "friday"])

        # Convert day names to cron format
        day_map = {
            "monday": "mon", "tuesday": "tue", "wednesday": "wed",
            "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun"
        }
        cron_days = ",".join([day_map[d] for d in days])

        # Parse frequency
        if frequency == "every_1_minute":
            minute_interval = "*/1"
        elif frequency == "every_5_minutes":
            minute_interval = "*/5"
        elif frequency == "every_15_minutes":
            minute_interval = "*/15"
        else:
            minute_interval = "0"  # Once per hour

        # Create cron trigger for the time window
        start_hour, start_minute = map(int, start_time.split(":"))
        end_hour, end_minute = map(int, end_time.split(":"))

        # Schedule job
        self.scheduler.add_job(
            func=self._trigger_strategy,
            trigger=CronTrigger(
                day_of_week=cron_days,
                hour=f"{start_hour}-{end_hour}",
                minute=minute_interval
            ),
            args=[strategy_id],
            id=f"{strategy_id}_schedule",
            name=f"Trigger {strategy_id}",
            replace_existing=True
        )

        logger.info(f"Scheduled {strategy_id} to run {frequency} on {cron_days} between {start_time}-{end_time} UTC")

    def _setup_condition_based_schedule(self, strategy_id: str, schedule: Dict[str, Any]):
        """Set up condition-based monitoring for a strategy."""
        # For condition-based strategies, we set up a monitoring loop
        check_interval = schedule.get("check_interval_seconds", 60)

        self.scheduler.add_job(
            func=self._check_conditions,
            trigger="interval",
            seconds=check_interval,
            args=[strategy_id],
            id=f"{strategy_id}_condition_check",
            name=f"Check conditions for {strategy_id}",
            replace_existing=True
        )

    async def _trigger_strategy(self, strategy_id: str):
        """Trigger a specific strategy."""
        manifest = self.active_strategies.get(strategy_id)
        if not manifest:
            logger.error(f"Strategy {strategy_id} not found")
            return

        logger.info(f"SCHEDULER: Triggering strategy: {strategy_id}")

        # Check if agent is already running
        if strategy_id in self.running_agents and not self.running_agents[strategy_id].done():
            logger.warning(f"Strategy {strategy_id} is already running, skipping trigger")
            return

        # Check pre-conditions
        if not await self._check_preconditions(manifest):
            logger.info(f"Pre-conditions not met for {strategy_id}, skipping trigger")
            return

        # Create command to trigger the specialist agent
        command = {
            "request_id": f"scheduled_{strategy_id}_{datetime.utcnow().isoformat()}",
            "action_type": "TRIGGER_AGENT_ANALYSIS",
            "payload": {
                "agent_name": strategy_id,
                "mission": f"Execute {manifest['strategy_name']}",
                "context": {
                    "manifest": manifest,
                    "trigger_time": datetime.utcnow().isoformat(),
                    "trigger_source": "scheduler"
                }
            },
            "metadata": {
                "source": "SchedulingAgent",
                "priority": "high"
            }
        }

        # Queue the command
        self.redis.lpush("zanalytics:command_queue", json.dumps(command))

        # Start monitoring task
        task = asyncio.create_task(self._monitor_agent_execution(strategy_id))
        self.running_agents[strategy_id] = task

    async def _check_conditions(self, strategy_id: str):
        """Check conditions for condition-based strategies."""
        manifest = self.active_strategies.get(strategy_id)
        if not manifest:
            return

        conditions = manifest.get("schedule", {}).get("conditions", [])

        for condition in conditions:
            if not await self._evaluate_condition(condition):
                return  # All conditions must be met

        # All conditions met, trigger strategy
        await self._trigger_strategy(strategy_id)

    async def _check_preconditions(self, manifest: Dict[str, Any]) -> bool:
        """Check if pre-conditions for a strategy are met."""
        pre_conditions = manifest.get("pre_conditions", [])

        for condition in pre_conditions:
            check_type = condition.get("check")

            if check_type == "market_open":
                if not await self._is_market_open(manifest.get("risk_params", {}).get("allowed_pairs", [])):
                    return False

            elif check_type == "no_high_impact_news":
                params = condition.get("params", {})
                if await self._has_high_impact_news(
                    lookback=params.get("lookback_minutes", -30),
                    lookahead=params.get("lookahead_minutes", 30)
                ):
                    return False

        return True

    async def _is_market_open(self, pairs: List[str]) -> bool:
        """Check if the market is open for the given pairs."""
        # Simplified check - in production, would check actual market hours
        now = datetime.now(timezone.utc)

        # Forex is closed on weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Could add more sophisticated checks here
        return True

    async def _has_high_impact_news(self, lookback: int, lookahead: int) -> bool:
        """Check for high-impact news events."""
        # In production, this would query a news calendar API
        # For now, return False (no high-impact news)
        return False

    async def _evaluate_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        condition_type = condition.get("type")
        params = condition.get("params", {})

        if condition_type == "price_above":
            symbol = params.get("symbol")
            level = params.get("level")
            current_price = await self._get_current_price(symbol)
            return current_price > level

        elif condition_type == "volatility_above":
            symbol = params.get("symbol")
            threshold = params.get("threshold")
            current_vol = await self._get_volatility(symbol)
            return current_vol > threshold

        # Add more condition types as needed
        return False

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol from Redis."""
        price_key = f"zanalytics:prices:{symbol}"
        price_data = self.redis.get(price_key)
        if price_data:
            return json.loads(price_data).get("price", 0.0)
        return 0.0

    async def _get_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol."""
        vol_key = f"zanalytics:volatility:{symbol}"
        vol_data = self.redis.get(vol_key)
        if vol_data:
            return json.loads(vol_data).get("value", 0.0)
        return 0.0

    async def _monitor_agent_execution(self, strategy_id: str):
        """Monitor the execution of a triggered agent."""
        logger.info(f"Monitoring execution of {strategy_id}")

        # Wait for agent to complete (with timeout)
        timeout = 300  # 5 minutes
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).seconds < timeout:
            # Check if agent has reported completion
            status_key = f"zanalytics:agent_status:{strategy_id}"
            status = self.redis.get(status_key)

            if status:
                status_data = json.loads(status)
                if status_data.get("status") in ["completed", "failed"]:
                    logger.info(f"Agent {strategy_id} finished with status: {status_data['status']}")
                    break

            await asyncio.sleep(5)  # Check every 5 seconds

    async def shutdown(self):
        """Gracefully shutdown the scheduling agent."""
        logger.info("Shutting down Scheduling Agent...")

        # Cancel all running agent tasks
        for task in self.running_agents.values():
            if not task.done():
                task.cancel()

        # Shutdown scheduler
        self.scheduler.shutdown()

        logger.info("Scheduling Agent shutdown complete")
