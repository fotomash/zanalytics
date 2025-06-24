import pytest
from core import advanced_stoploss_lots_engine as engine


def test_get_pip_point_value_non_usd_account(monkeypatch):
    """Point value should convert using provided FX rate."""
    def fake_rate(pair):
        assert pair == "JPYEUR"
        return 0.006

    monkeypatch.setattr(engine, "get_live_fx_rate", fake_rate)
    value, decimals = engine.get_pip_point_value("USDJPY", account_currency="EUR")
    assert decimals == 3
    assert pytest.approx(0.6, rel=1e-3) == value


def test_get_pip_point_value_same_currency(monkeypatch):
    def fake_rate(pair):
        raise AssertionError("should not call get_live_fx_rate")

    monkeypatch.setattr(engine, "get_live_fx_rate", fake_rate)
    value, decimals = engine.get_pip_point_value("EURUSD", account_currency="USD")
    assert decimals == 5
    assert value == 1.0

