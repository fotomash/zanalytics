import os

# Ensure lightweight test mode is enabled before the package is imported
os.environ.setdefault("ZANALYTICS_TEST_MODE", "1")

def pytest_configure():
    """Additional pytest configuration can be added here."""
    pass
