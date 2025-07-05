import importlib
import importlib.util
import os
import sys
import pytest

if importlib.util.find_spec("ncos") is None:
    pytest.skip("ncos package missing", allow_module_level=True)


def reload_zanalytics(monkeypatch, test_mode):
    if test_mode is None:
        monkeypatch.delenv("ZANALYTICS_TEST_MODE", raising=False)
    else:
        monkeypatch.setenv("ZANALYTICS_TEST_MODE", test_mode)
    if "zanalytics" in sys.modules:
        del sys.modules["zanalytics"]
    return importlib.import_module("zanalytics")


def test_init_without_test_mode(monkeypatch):
    za = reload_zanalytics(monkeypatch, None)
    assert "initialize_agents" in za.__all__
    assert "WyckoffSpecialistAgent" in za.__all__


def test_init_with_test_mode(monkeypatch):
    za = reload_zanalytics(monkeypatch, "1")
    assert za.__all__ == []

