import os
import sys
from pathlib import Path

import pandas as pd

os.environ["ZANALYTICS_TEST_MODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.smc_enrichment_engine as _smc
import core.wyckoff_phase_engine as _wyckoff
import core.event_detector as _event
sys.modules['smc_enrichment_engine'] = _smc
sys.modules['wyckoff_phase_engine'] = _wyckoff
sys.modules['event_detector'] = _event

from core.analysis import marker_enrichment_engine as me
me.SMC_ENGINE_LOADED = True
me.tag_smc_zones = _smc.tag_smc_zones
me.WYCKOFF_ENGINE_LOADED = True
def _stub_wyckoff(df, **kwargs):
    df = df.copy()
    df["wyckoff_phase"] = "A"
    df["wyckoff_event"] = "test"
    return df
me.tag_wyckoff_phases = _stub_wyckoff
add_all_indicators = me.add_all_indicators
from core.liquidity_vwap_detector import LiquidityVWAPDetector


def make_df(include_volume=True):
    index = pd.date_range("2024-01-01", periods=5, freq="H")
    data = {
        "Open": [1, 2, 3, 4, 5],
        "High": [2, 3, 4, 5, 6],
        "Low": [0, 1, 2, 3, 4],
        "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
    }
    if include_volume:
        data["Volume"] = [10, 20, 30, 40, 50]
    return pd.DataFrame(data, index=index)


def test_enrichment_adds_columns():
    df = make_df()
    enriched = add_all_indicators(df, timeframe="H1", config={})
    for col in ["bos", "wyckoff_phase", "liq_sweep"]:
        assert col in enriched.columns
    assert len(enriched) == len(df)


def test_liquidity_vwap_detector():
    df = make_df()
    detector = LiquidityVWAPDetector()
    result = detector.detect_deviation_sweeps(df)
    assert "LiquiditySweep" in result.columns
    assert "SweepFlag" in result.columns
    assert len(result) == len(df)


def test_edge_cases():
    df_no_vol = make_df(include_volume=False)
    enriched = add_all_indicators(df_no_vol, timeframe="H1", config={})
    assert "Volume" in enriched.columns
    assert (enriched["Volume"] == 0).all()

    empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    empty_result = add_all_indicators(empty_df, timeframe="H1", config={})
    assert empty_result.empty
