"""Asynchronous scheduling agent for triggering strategy agents."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
import yaml


class SchedulingAgent:
    """Master scheduling agent that monitors manifests and triggers agents."""

    def __init__(self, redis_client: redis.Redis, manifest_dir: str = "knowledge/strategies", config: Optional[Dict[str, Any]] = None) -> None:
        self.redis = redis_client
        self.manifest_dir = Path(manifest_dir)
        self.config = config or {}
        self.scheduler = AsyncIOScheduler(timezone=timezone.utc)
        self.active_strategies: Dict[str, Dict[str, Any]] = {}
        self.running_agents: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        logger.info("Initializing Scheduling Agent...")
        await self._load_strategy_manifests()
        self._setup_schedules()
        self.scheduler.start()
        logger.info(f"Scheduling Agent initialized with {len(self.active_strategies)} strategies")

    async def shutdown(self) -> None:
        logger.info("Shutting down Scheduling Agent...")
        for task in list(self.running_agents.values()):
            if not task.done():
                task.cancel()
        self.scheduler.shutdown()

    # ------------------------------------------------------------------
    # Manifest loading and scheduling
    # ------------------------------------------------------------------

    async def _load_strategy_manifests(self) -> None:
        if not self.manifest_dir.exists():
            logger.warning(f"Manifest directory not found: {self.manifest_dir}")
            return
        for manifest_file in self.manifest_dir.glob("*.yml"):
            try:
                with open(manifest_file, "r") as f:
                    manifest = yaml.safe_load(f)
                if manifest.get("enabled", False):
                    strategy_id = manifest["strategy_id"]
                    self.active_strategies[strategy_id] = manifest
                    logger.debug(f"Loaded strategy {strategy_id}")
            except Exception as exc:  # pragma: no cover - file errors
                logger.error(f"Failed to load {manifest_file}: {exc}")

    def _setup_schedules(self) -> None:
        for strategy_id, manifest in self.active_strategies.items():
            schedule_cfg = manifest.get("schedule", {})
            if schedule_cfg.get("trigger_type") == "time_based":
                self._setup_time_based_schedule(strategy_id, schedule_cfg)

    def _setup_time_based_schedule(self, strategy_id: str, schedule: Dict[str, Any]) -> None:
        time_window = schedule.get("utc_time_window", {"start": "00:00", "end": "23:59"})
        days = schedule.get("days", ["monday", "tuesday", "wednesday", "thursday", "friday"])
        frequency = schedule.get("frequency", "every_1_minute")
        day_map = {
            "monday": "mon",
            "tuesday": "tue",
            "wednesday": "wed",
            "thursday": "thu",
            "friday": "fri",
            "saturday": "sat",
            "sunday": "sun",
        }
        cron_days = ",".join(day_map[d] for d in days)
        minute_interval = {
            "every_1_minute": "*/1",
            "every_5_minutes": "*/5",
            "every_15_minutes": "*/15",
        }.get(frequency, "0")
        start_hour, _ = map(int, time_window.get("start", "00:00").split(":"))
        end_hour, _ = map(int, time_window.get("end", "23:59").split(":"))
        trigger = CronTrigger(day_of_week=cron_days, hour=f"{start_hour}-{end_hour}", minute=minute_interval)
        self.scheduler.add_job(
            self._trigger_strategy,
            trigger=trigger,
            args=[strategy_id],
            id=f"{strategy_id}_schedule",
            name=f"Trigger {strategy_id}",
            replace_existing=True,
        )
        logger.debug(f"Scheduled {strategy_id} with {frequency} on {cron_days}")

    async def _trigger_strategy(self, strategy_id: str) -> None:
        manifest = self.active_strategies.get(strategy_id)
        if not manifest:
            return
        logger.info(f"SCHEDULER: Triggering {strategy_id}")
        if strategy_id in self.running_agents and not self.running_agents[strategy_id].done():
            logger.warning(f"{strategy_id} already running")
            return
        command = {
            "request_id": f"scheduled_{strategy_id}_{datetime.utcnow().isoformat()}",
            "action_type": "TRIGGER_AGENT_ANALYSIS",
            "payload": {
                "agent_name": strategy_id,
                "mission": f"Execute {manifest['strategy_name']}",
                "context": {
                    "manifest": manifest,
                    "trigger_time": datetime.utcnow().isoformat(),
                    "trigger_source": "scheduler",
                },
            },
            "metadata": {"source": "SchedulingAgent"},
        }
        self.redis.lpush("zanalytics:command_queue", json.dumps(command))
        task = asyncio.create_task(self._monitor_agent_execution(strategy_id))
        self.running_agents[strategy_id] = task

    async def _monitor_agent_execution(self, strategy_id: str) -> None:
        timeout = 60
        start = datetime.utcnow()
        while (datetime.utcnow() - start).seconds < timeout:
            status_key = f"zanalytics:agent_status:{strategy_id}"
            status = self.redis.get(status_key)
            if status:
                status_data = json.loads(status)
                if status_data.get("status") in {"completed", "failed"}:
                    break
            await asyncio.sleep(1)

    # Helper methods for condition checks (simplified for tests)
    async def _is_market_open(self, pairs: List[str]) -> bool:
        now = datetime.now(timezone.utc)
        return now.weekday() < 5

    async def _has_high_impact_news(self, lookback: int, lookahead: int) -> bool:
        return False
