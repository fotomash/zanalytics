import json
from pathlib import Path

from agents.optimization_meta import OptimizationMetaAgent


def create_sample_logs(tmp_path: Path) -> Path:
    journal = tmp_path / "journal"
    journal.mkdir()
    data = (
        "timestamp,result,confluence_path,rejected\n"
        "2024-01-01T00:00:00Z,Win,\"['bias','bos']\",False\n"
        "2024-01-02T00:00:00Z,Loss,\"['bias','bos']\",True\n"
        "2024-01-03T00:00:00Z,Loss,\"['bias','sweep','bos']\",True\n"
        "2024-01-04T00:00:00Z,Win,\"['bias','sweep','bos']\",False\n"
        "2024-01-05T00:00:00Z,Loss,\"['bias']\",True\n"
    )
    (journal / "trade_log.csv").write_text(data)
    return journal


def test_metrics_and_report_generation(tmp_path: Path):
    journal = create_sample_logs(tmp_path)
    agent = OptimizationMetaAgent(journal_dir=str(journal))
    report_json = agent.generate_report()
    report = json.loads(report_json)

    assert report["confluence_win_rates"]["bias->bos"] == 0.5
    assert report["rejection_effectiveness"] == 1.0
    assert "entry_threshold" in report["suggestions"]

    # Markdown output smoke test
    md = agent.generate_report(as_markdown=True)
    assert md.startswith("# Optimization Meta Report")
