import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.analysis.marker_enrichment_engine import add_all_indicators


def test_add_all_indicators_returns_df():
    df = pd.DataFrame({
        "Open": [1.0],
        "High": [1.0],
        "Low": [1.0],
        "Close": [1.0],
        "Volume": [100],
    }, index=pd.date_range("2020-01-01", periods=1, freq="D"))

    assert callable(add_all_indicators)

    result = add_all_indicators(df, timeframe="D", config={})
    assert isinstance(result, pd.DataFrame)
