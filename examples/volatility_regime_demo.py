"""Demonstrate volatility regime detection using AnalysisOrchestrator."""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

from core.orchestrator import AnalysisOrchestrator
import core.advanced_smc_orchestrator as adv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

orch = AnalysisOrchestrator()
run_strategy = orch.select_strategy("advanced_smc")


def make_data():
    idx = pd.date_range("2024-01-01", periods=40, freq="H", tz="UTC")
    data = {
        "Open": np.ones(len(idx)),
        "High": np.ones(len(idx)) + 0.001,
        "Low": np.ones(len(idx)) - 0.001,
        "Close": np.ones(len(idx)),
    }
    return pd.DataFrame(data, index=idx)


def run_variant(vol_cfg):
    df = make_data()
    all_tf = {"m1": df}
    profile = {
        "structure_timeframe": "m1",
        "poi_timeframe": "m1",
        "poi_tap_check_timeframe": "m1",
        "confirmation_timeframe": "m1",
        "execution_timeframe": "m1",
        "risk_model_config": {},
        "poi_tap_config": {},
        "volatility_config": vol_cfg,
    }
    adv.analyze_market_structure = lambda df: {"htf_bias": "Bullish"}
    adv.find_and_validate_smc_pois = lambda *a, **k: [
        {"range": [0.999, 1.0], "type": "Bullish", "timestamp": df.index[0], "source_tf": "m1"}
    ]
    adv.check_poi_tap_smc = lambda **kw: {"is_tapped": True, "tap_time": df.index[-1]}
    adv.confirm_smc_entry = lambda **kw: {
        "confirmation_status": True,
        "ltf_poi_timestamp": df.index[-1],
        "ltf_poi_range": [1.0, 1.0],
        "mitigated_htf_poi": {"type": "Bullish"},
    }
    adv.execute_smc_entry = lambda **kw: {"entry_confirmed": True}
    adv.load_strategy_profile = lambda variant: profile
    res = run_strategy(all_tf, "demo", df.index[-1])
    print(f"Volatility regime: {res['context'].get('volatility_regime')}")


if __name__ == "__main__":
    quiet_cfg = {
        "quiet_atr_threshold_pct": 100,
        "quiet_bbw_threshold_pct": 100,
        "explosive_atr_threshold_pct": 200,
        "explosive_bbw_threshold_pct": 200,
        "atr_period": 14,
        "bb_period": 20,
        "atr_ma_period": 10,
        "bbw_ma_period": 10,
    }
    expl_cfg = {
        "quiet_atr_threshold_pct": 0.01,
        "quiet_bbw_threshold_pct": 0.01,
        "explosive_atr_threshold_pct": 0.5,
        "explosive_bbw_threshold_pct": 0.5,
        "atr_period": 14,
        "bb_period": 20,
        "atr_ma_period": 10,
        "bbw_ma_period": 10,
    }
    print("-- Quiet config --")
    run_variant(quiet_cfg)
    print("-- Explosive config --")
    run_variant(expl_cfg)
