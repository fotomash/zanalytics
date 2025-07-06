import pytest

from core.dispatcher.action_dispatcher import ActionDispatcher, get_dispatcher


class FakeRedis:
    def __init__(self):
        self.storage = {}
        self.published = []
        self.lists = {}

    def hset(self, name: str, key: str, value: str) -> None:
        self.storage.setdefault(name, {})[key] = value

    def lpush(self, name: str, value: str) -> None:
        self.lists.setdefault(name, []).insert(0, value)

    def publish(self, channel: str, message: str) -> None:
        self.published.append((channel, message))

    def get(self, key: str):
        return self.storage.get(key)


@pytest.mark.asyncio
async def test_dispatcher_logs_journal_entry():
    r = FakeRedis()
    dispatcher = get_dispatcher(r, {})
    cmd = {
        "request_id": "1",
        "action_type": "LOG_JOURNAL_ENTRY",
        "payload": {"content": "hello"},
    }
    result = await dispatcher.process_command(cmd)
    assert result["status"] == "success"
    assert r.storage["zanalytics:journal"]["1"]
    assert r.published
    from core.dispatcher import action_dispatcher as mod
    mod._dispatcher_instance = None


@pytest.mark.asyncio
async def test_dispatcher_unknown_action():
    r = FakeRedis()
    dispatcher = ActionDispatcher(r, {})
    cmd = {
        "request_id": "2",
        "action_type": "MISSING",
        "payload": {},
    }
    result = await dispatcher.process_command(cmd)
    assert result["status"] == "error"
