import os
import sys
from pathlib import Path

import pandas as pd

# ensure test mode and import path
os.environ["ZANALYTICS_TEST_MODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# load and expose helper modules so marker_enrichment_engine can import them
import core.smc_enrichment_engine as _smc
import core.wyckoff_phase_engine as _wyckoff
import core.event_detector as _event
sys.modules['smc_enrichment_engine'] = _smc
sys.modules['wyckoff_phase_engine'] = _wyckoff
sys.modules['event_detector'] = _event

from core.analysis import marker_enrichment_engine as me

# activate indicator engines and stub wyckoff tagging
me.SMC_ENGINE_LOADED = True
me.tag_smc_zones = _smc.tag_smc_zones
me.WYCKOFF_ENGINE_LOADED = True

def _stub_wyckoff(df, **kwargs):
    df = df.copy()
    df["wyckoff_phase"] = "A"
    df["wyckoff_event"] = "test"
    return df

me.tag_wyckoff_phases = _stub_wyckoff

from core.analysis import pipeline as pl

# replace network heavy calls with simple stubs
pl.fetch_macro_context = lambda asset: {"risk_state": "test"}
pl.detect_inducement_from_structure = lambda df, tf, struct: {"status": True}
pl.add_all_indicators = me.add_all_indicators


def make_df():
    index = pd.date_range("2024-01-01", periods=5, freq="H")
    data = {
        "Open": [1, 2, 3, 4, 5],
        "High": [2, 3, 4, 5, 6],
        "Low": [0, 1, 2, 3, 4],
        "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
        "Volume": [10, 20, 30, 40, 50],
    }
    return pd.DataFrame(data, index=index)


def test_pipeline_interface_columns_and_results():
    df = make_df()
    config = {
        "markers": {"timeframe": "H1"},
        # pass a non-empty dict so the option evaluates True
        "liquidity_vwap": {"std_window": 2},
        "liquidity_smc": {"timeframe": "H1", "structure_data": {}},
        "macro": {"asset_symbol": "TEST"},
    }
    pipeline = pl.EnrichmentPipeline(config)
    result = pipeline.apply(df)

    for col in ["bos", "wyckoff_phase", "LiquiditySweep"]:
        assert col in result.columns
    assert len(result) == len(df)

    assert pipeline.results.get("liquidity_smc") == {"status": True}
    assert pipeline.results.get("macro") == {"risk_state": "test"}
