import json

import pytest

from core.agents.scheduling_agent import SchedulingAgent


class FakeRedis:
    def __init__(self):
        self.lists = {}
        self.storage = {}

    def lpush(self, name: str, value: str) -> None:
        self.lists.setdefault(name, []).insert(0, value)

    def get(self, key: str):
        return self.storage.get(key)

    def publish(self, *args, **kwargs):
        pass

    def hset(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_scheduling_agent_triggers_time_based(tmp_path):
    # create manifest directory
    manifest = {
        "strategy_id": "test_strategy",
        "strategy_name": "Test Strategy",
        "enabled": True,
        "schedule": {
            "trigger_type": "time_based",
            "utc_time_window": {"start": "00:00", "end": "23:59"},
            "frequency": "every_1_minute",
        },
    }
    man_dir = tmp_path / "manifests"
    man_dir.mkdir()
    (man_dir / "test.yml").write_text(json.dumps(manifest))

    r = FakeRedis()
    agent = SchedulingAgent(r, manifest_dir=str(man_dir))
    await agent.initialize()

    # there should be one scheduled job
    jobs = agent.scheduler.get_jobs()
    assert jobs

    # trigger the job manually to avoid waiting
    await agent._trigger_strategy("test_strategy")

    # verify command queued
    queued = r.lists.get("zanalytics:command_queue")
    assert queued and json.loads(queued[0])["payload"]["agent_name"] == "test_strategy"

    await agent.shutdown()
