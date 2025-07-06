import pandas as pd
from agent_riskmanager import RiskManagerAgent


def test_risk_manager_recommends_tiered_risk():
    data = pd.DataFrame([
        {"SPREAD": 0.3, "RET": 0.0002},
        {"SPREAD": 0.3, "RET": 0.0002},
    ])
    context = {"micro_context": data, "maturity_score": 0.75}
    agent = RiskManagerAgent(context)

    cfg = {
        "base_risk_pct": 1.0,
        "score_risk_tiers": {"0.8": 0.5, "0.6": 1.0, "0.4": 1.5},
    }
    res = agent.evaluate_risk_profile(cfg)
    assert res["recommended_risk_pct"] == 1.0
