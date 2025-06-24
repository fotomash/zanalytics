import os

def pytest_configure():
    os.environ.setdefault("ZANALYTICS_TEST_MODE", "1")
