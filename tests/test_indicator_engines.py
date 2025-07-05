import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bollinger_engine import add_bollinger_bands
from dss_engine import add_dss
from fractal_engine import add_fractals
from vwap_engine import add_vwap
from divergence_engine import add_rsi_divergence


def make_df(rows: int = 30):
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {
        "Open": range(rows),
        "High": [x + 1 for x in range(rows)],
        "Low": [x - 1 for x in range(rows)],
        "Close": range(rows),
        "Volume": [x + 1 for x in range(rows)],
    }
    return pd.DataFrame(data, index=index)


def test_bollinger_engine_columns():
    df = make_df()
    result = add_bollinger_bands(df.copy())
    assert {"bb_upper", "bb_lower", "bb_mid"}.issubset(result.columns)


def test_dss_engine_columns():
    df = make_df()
    result = add_dss(df.copy())
    assert {"dss_k", "dss_d"}.issubset(result.columns)


def test_fractal_engine_columns():
    df = make_df()
    result = add_fractals(df.copy())
    assert {"fractal_high", "fractal_low"}.issubset(result.columns)


def test_vwap_engine_column():
    df = make_df()
    result = add_vwap(df.copy(), tf="D1")
    assert "vwap" in result.columns


def test_divergence_engine_columns():
    df = make_df()
    result = add_rsi_divergence(df.copy())
    assert "rsi_divergence" in result.columns
    assert "rsi" in result.columns
