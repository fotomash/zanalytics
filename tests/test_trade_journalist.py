import pytest
from agents.agent_tradejournalist import TradeJournalistAgent
from schemas.agent_profile_schemas import JournalingConfig


def test_journalist_records_confluence_path():
    agent = TradeJournalistAgent()
    steps = ["bias", "sweep", "bos", "fvg_touch"]
    entry = agent.log_decision(
        {
            "symbol": "EURUSD",
            "entry_signal": "long",
            "trigger": "setup",
            "confidence": 0.9,
            "reason": "test",
            "phase_context": None,
        },
        macro_result={"bias": "bull"},
        confluence_path=steps,
    )
    assert entry["confluence_path"] == steps
    assert agent.get_latest_entry() == entry


def test_journaling_config_accepts_confluence_path():
    cfg = JournalingConfig(confluence_path=["bias", "sweep"])
    assert cfg.confluence_path == ["bias", "sweep"]
