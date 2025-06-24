# tests/conftest.py - Set test mode immediately
import os

# Ensure test mode is active for all imports
os.environ['ZANALYTICS_TEST_MODE'] = '1'

import pytest


def pytest_configure(config):
    """Pytest configuration hook."""
    pass


def pytest_unconfigure(config):
    """Cleanup after tests."""
    os.environ.pop('ZANALYTICS_TEST_MODE', None)
