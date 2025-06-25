# tests/conftest.py - Set test mode immediately
import os
import pytest

# Set test mode BEFORE any imports
os.environ['ZANALYTICS_TEST_MODE'] = '1'

def pytest_configure(config):
    """pytest configuration hook"""
    # Test mode already set above
    pass

def pytest_unconfigure(config):
    """Clean up after tests"""
    if 'ZANALYTICS_TEST_MODE' in os.environ:
        del os.environ['ZANALYTICS_TEST_MODE']

