import os
import yaml
import importlib
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

class SchedulingAgent:
    """Master scheduling agent that loads strategy manifests and schedules specialist agents."""

    def __init__(self, strategies_dir: str = "knowledge/strategies"):
        self.strategies_dir = strategies_dir
        self.scheduler = BlockingScheduler(timezone='UTC')
        print("SCHEDULER: Initializing Scheduling Agent...")

    def load_and_schedule_strategies(self):
        """Parse manifests and create scheduled jobs."""
        print(f"SCHEDULER: Scanning for strategy manifests in '{self.strategies_dir}'...")
        if not os.path.isdir(self.strategies_dir):
            print(f"SCHEDULER: No strategies directory found at '{self.strategies_dir}'")
            return
        for filename in os.listdir(self.strategies_dir):
            if filename.endswith(".yml") or filename.endswith(".yaml"):
                filepath = os.path.join(self.strategies_dir, filename)
                with open(filepath, "r") as f:
                    manifest = yaml.safe_load(f)
                schedule_cfg = manifest.get("schedule")
                if schedule_cfg and schedule_cfg.get("trigger_type") == "time_based":
                    self._create_job_from_manifest(manifest)

    def _create_job_from_manifest(self, manifest: dict):
        """Create an APScheduler job from a manifest."""
        strategy_id = manifest.get("strategy_id", "unknown_strategy")
        schedule_cfg = manifest["schedule"]
        try:
            agent_module = importlib.import_module(manifest["agent_module"])
            agent_class = getattr(agent_module, manifest["agent_class"])

            window = (
                schedule_cfg.get("utc_time_window")
                or schedule_cfg.get("trigger_window")
            )
            if not window:
                raise ValueError("Schedule is missing a time window definition")
            start = datetime.strptime(window["start"], "%H:%M")
            end = datetime.strptime(window["end"], "%H:%M")
            minutes = schedule_cfg.get("frequency", {}).get("value", 1)
            trigger = CronTrigger(
                hour=f"{start.hour}-{end.hour}",
                minute=f"*/{minutes}",
                timezone=schedule_cfg.get("timezone", "UTC"),
            )
            self.scheduler.add_job(
                self.run_agent_mission,
                trigger=trigger,
                args=[agent_class, manifest],
                id=strategy_id,
                name=f"Run {strategy_id}",
                replace_existing=True,
            )
            print(f"SCHEDULER: ✅ Successfully scheduled '{strategy_id}'.")
        except Exception as e:
            print(f"SCHEDULER: ❌ ERROR - Could not schedule '{strategy_id}': {e}")

    @staticmethod
    def run_agent_mission(agent_class, manifest: dict):
        """Instantiate and execute the specialist agent."""
        print(f"--- MISSION TRIGGERED [{datetime.utcnow().isoformat()}] ---")
        print(f"  Agent: {manifest['agent_class']}")
        print(f"  Strategy: {manifest['strategy_name']}")
        agent = agent_class(manifest)
        if hasattr(agent, 'execute_workflow'):
            agent.execute_workflow()
        print("--- MISSION COMPLETE ---")

    def start(self):
        self.load_and_schedule_strategies()
        print("\nSCHEDULER: All jobs scheduled. System is now in proactive monitoring mode.")
        print("Scheduler is running... Press Ctrl+C to exit.")
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            print("\nSCHEDULER: Shutting down.")
            self.scheduler.shutdown()

if __name__ == '__main__':
    scheduler = SchedulingAgent()
    scheduler.start()
