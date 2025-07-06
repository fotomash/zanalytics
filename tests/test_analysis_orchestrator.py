import asyncio
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import orchestrator as orch_mod
import pandas as pd


def test_orchestrator_dynamic_loading_and_run(tmp_path, monkeypatch):
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

    result = asyncio.run(orch.run({"orchestrator": "dummy", "prompt": "hello"}))
    assert result == {"echo": "hello"}

    assert user_ctx.is_file()
    saved = json.loads(user_ctx.read_text())
    assert saved["last_result"] == result


def test_run_full_enrichment_returns_unified_bars():
    orch = orch_mod.AnalysisOrchestrator(config_path="missing.yaml")

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [10, 20, 30],
        }
    )

    bars = orch.run_full_enrichment(df, {"pipeline": False})

    assert isinstance(bars, list)
    assert len(bars) == len(df)
    assert all(isinstance(b, orch_mod.UnifiedAnalyticsBar) for b in bars)
