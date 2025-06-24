"""Example implementation of a ZanFlow strategy."""

from typing import Dict, Any
from .base_strategy import BaseStrategy, StrategySignal


class ZanFlowStrategy(BaseStrategy):
    """Simple momentum based strategy placeholder."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("zanflow", config)

    def evaluate(self, data: Dict[str, Any]) -> StrategySignal:
        price = data.get("price", 0)
        momentum = data.get("momentum", 0)
        action = "BUY" if momentum > 0 else "SELL"
        confidence = min(abs(momentum), 1.0)
        return StrategySignal("entry", data.get("symbol", "UNKNOWN"), action, confidence)
