import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zanflow_v12_intelligence_evolution.core.risk.adaptive_risk_manager import AdaptiveRiskManager

CONFIG_DIR = ROOT / "zanflow_v12_intelligence_evolution" / "configs"


def test_calculate_position_size():
    rm = AdaptiveRiskManager(config_path=str(CONFIG_DIR / "adaptive_risk_config.yaml"))
    profile = rm.calculate_position_size(
        symbol="EURUSD",
        maturity_score=0.9,
        stop_distance_pips=20,
        account_balance=100000,
        current_conditions={"killzone_active": True, "high_impact_news": False, "volatility": 0.8},
    )

    assert profile.risk_percent == 1.2
    assert profile.position_size == 75.0
