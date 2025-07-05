import asyncio
import json
import sys
from pathlib import Path

import yaml

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import orchestrator as orch_mod


@pytest.mark.asyncio
async def test_orchestrator_dynamic_loading_and_run(tmp_path, monkeypatch):
    """Ensure AnalysisOrchestrator loads modules and writes user context."""
    # Create dummy strategy module in tmp directory
    module_path = tmp_path / "dummy_mod.py"
    module_path.write_text(
        "def dummy_strategy(prompt):\n    return {'echo': prompt}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    # Write mock config pointing to dummy strategy
    config = {
        "orchestrators": {
            "dummy": {"module": "dummy_mod", "callable": "dummy_strategy"}
        },
        "default_orchestrator": "dummy",
    }
    config_file = tmp_path / "zsi_config.yaml"
    config_file.write_text(yaml.safe_dump(config))

    # Redirect USER_CONTEXT_PATH into tmp directory
    user_ctx = tmp_path / "data" / "user_context.json"
    monkeypatch.setattr(orch_mod, "USER_CONTEXT_PATH", user_ctx)

    orch = orch_mod.AnalysisOrchestrator(config_path=str(config_file))

    strategy = orch.select_strategy("dummy")
    assert callable(strategy)

    result = await orch.run({"orchestrator": "dummy", "prompt": "hello"})
    assert result == {"echo": "hello"}

    assert user_ctx.is_file()
    saved = json.loads(user_ctx.read_text())
    assert saved["last_result"] == result
