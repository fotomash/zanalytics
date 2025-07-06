import pytest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("risk_manager", ROOT / "risk_manager.py")
risk_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk_module)
calculate_risk = risk_module.calculate_risk


def test_equal_entry_and_stop_raises():
    with pytest.raises(ValueError):
        calculate_risk(100, 100, 120)


def test_calculate_risk_normal():
    rr, risk, reward = calculate_risk(101, 100, 106)
    assert rr == pytest.approx(6 / 1)
    assert risk == 1
    assert reward == 5
