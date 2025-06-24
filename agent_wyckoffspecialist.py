class WyckoffSpecialistAgent:
    """Minimal Wyckoff specialist agent.

    This agent inspects microstructure Wyckoff phase information from the
    provided context and exposes a simple analysis method.
    """

    def __init__(self, context=None):
        self.context = context or {}
        self.symbol = self.context.get("symbol", "XAUUSD")
        # Expected structure: {"phase": "A", "confidence": 0.0}
        self.micro_wyckoff = self.context.get("micro_wyckoff", {})

    def analyze(self):
        """Return the latest microstructure Wyckoff phase information."""
        if not isinstance(self.micro_wyckoff, dict) or not self.micro_wyckoff:
            return {
                "symbol": self.symbol,
                "phase": None,
                "confidence": 0.0,
                "reason": "No microstructure Wyckoff data",
            }

        return {
            "symbol": self.symbol,
            "phase": self.micro_wyckoff.get("phase"),
            "confidence": self.micro_wyckoff.get("confidence", 0.0),
            "reason": "Latest microstructure context",
        }
