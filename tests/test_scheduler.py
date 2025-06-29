import os
os.environ["ZANALYTICS_TEST_MODE"] = "1"

import schedule

from zanalytics_orchestrator import Scheduler, Task

class DummyExecutor:
    async def execute_task(self, task):
        return {"status": "success"}

class DummyWorkflowEngine:
    def __init__(self):
        self.executor = DummyExecutor()

def test_weekly_task_registered():
    schedule.clear()
    scheduler = Scheduler(DummyWorkflowEngine())
    task = Task(
        name="weekly_task",
        module="mod",
        method="method",
        params={},
        schedule="weekly",
    )
    scheduler.schedule_task(task)
    assert "weekly_task" in scheduler.scheduled_tasks
    assert any(job.unit == "weeks" for job in schedule.jobs)
