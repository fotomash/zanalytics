# tests/conftest.py - Set test mode immediately
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure test mode is active for all imports
os.environ['ZANALYTICS_TEST_MODE'] = '1'

import pytest


def pytest_configure(config):
    """Pytest configuration hook."""
    pass


def pytest_unconfigure(config):
    """Cleanup after tests."""
    os.environ.pop('ZANALYTICS_TEST_MODE', None)
