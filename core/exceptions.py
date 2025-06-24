class DataUnavailableError(Exception):
    """Raised when requested data cannot be retrieved."""


class InvalidSignalError(Exception):
    """Raised when a generated trading signal is invalid."""
