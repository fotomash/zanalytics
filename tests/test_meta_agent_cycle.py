import json
import sys
from pathlib import Path
import pandas as pd
import types

# Provide lightweight sklearn.metrics stub to satisfy MetaAgent import
sklearn_stub = types.ModuleType("sklearn")
metrics_stub = types.ModuleType("metrics")
def _mcc(y_true=None, y_pred=None):
    return 0.0
metrics_stub.matthews_corrcoef = _mcc
sklearn_stub.metrics = metrics_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.metrics", metrics_stub)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zanflow_v12_intelligence_evolution.core.meta_agent.optimization_engine import MetaAgent

CONFIG_DIR = ROOT / "zanflow_v12_intelligence_evolution" / "configs"


def create_stub_journal(path: Path):
    entries = []
    start = pd.Timestamp("2024-01-01")
    for i in range(30):
        entries.append({
            "trade_id": f"T{i}",
            "timestamp": (start + pd.Timedelta(hours=i)).isoformat(),
            "symbol": "EURUSD",
            "timeframe": "M5",
            "path_signature": "bias->bos" if i < 25 else "bias->sweep->bos",
            "r_multiple": 1.0 if i % 3 == 0 else -1.0,
            "executed": True,
            "final_maturity_score": 0.7 + 0.01 * i,
            "rejection_reason": "low_score" if i % 7 == 0 else None,
            "simulated_r": -0.5 if i % 7 == 0 else None,
        })
    path.write_text(json.dumps(entries, indent=2))


def test_meta_agent_run_cycle(tmp_path):
    journal = tmp_path / "journal.json"
    create_stub_journal(journal)

    agent = MetaAgent(
        config_path=str(CONFIG_DIR / "meta_agent_config.yaml"),
        journal_path=str(journal),
    )
    report = agent.run_weekly_analysis()

    assert report["report_id"].startswith("WEEKLY_")
    assert "confluence_paths" in report["detailed_analysis"]
    assert isinstance(report["recommendations"], list)

    report_file = Path("reports") / f"meta_analysis_{report['report_id']}.json"
    assert report_file.is_file()
