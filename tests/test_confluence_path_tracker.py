import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zanflow_v12_intelligence_evolution.core.confluence.path_tracker import ConfluencePathTracker


def test_path_lifecycle(tmp_path):
    journal = tmp_path / "journal.json"
    tracker = ConfluencePathTracker(journal_path=str(journal))

    tracker.start_path("T1", "EURUSD", "M5")
    tracker.add_event("T1", "BIAS_CONFIRMED", 0.8, 1500)
    tracker.add_event("T1", "BOS_CONFIRMED", 0.85, 1200)

    outcome = {"executed": True, "r_multiple": 1.5}
    completed = tracker.complete_path("T1", 0.9, outcome)

    assert "T1" not in tracker.active_paths

    sig = completed.get_path_signature()
    stats = tracker.path_statistics[sig]
    assert stats["occurrences"] == 1
    assert stats["executions"] == 1
    assert stats["wins"] == 1
    assert stats["total_r"] == 1.5

    data = json.loads(journal.read_text())
    assert data[0]["trade_id"] == "T1"
    assert data[0]["outcome"]["r_multiple"] == 1.5
