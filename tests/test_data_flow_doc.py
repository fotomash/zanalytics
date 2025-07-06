from pathlib import Path


def test_system_data_flow_doc_exists():
    doc_path = Path(__file__).resolve().parents[1] / "docs" / "System_Data_Flow.md"
    assert doc_path.is_file(), "System_Data_Flow.md should exist in docs/"
    content = doc_path.read_text().strip()
    assert content.startswith("# ZAnalytics System Data Flow"), "Doc should start with heading"
