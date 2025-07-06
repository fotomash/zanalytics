"""Risk management utilities for evaluating volatility and spread."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RiskManagerAgent:
    """Evaluate microstructure data to classify current trading risk."""

    context: Dict[str, Any] = field(default_factory=dict)
    symbol: str = field(init=False)
    micro_context: Any = field(init=False)
    confidence: float = field(init=False, default=0.0)
    spread: float | None = field(init=False, default=None)
    ret: float | None = field(init=False, default=None)
    tick_count: int = field(init=False, default=0)
    maturity_score: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.symbol = self.context.get("symbol", "XAUUSD")
        self.micro_context = self.context.get("micro_context", {})
        self.confidence = self.context.get("confidence", 0.0)
        self.tick_count = (
            len(self.micro_context) if hasattr(self.micro_context, "__len__") else 0
        )
        self.maturity_score = self.context.get("maturity_score")
        if hasattr(self.micro_context, "empty") and not self.micro_context.empty:
            last = self.micro_context.iloc[-1]
            self.spread = last.get("SPREAD")
            self.ret = last.get("RET")

    def evaluate_risk_profile(self, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Classify risk level and compute recommended risk percent."""
        result = {
            "symbol": self.symbol,
            "risk": "medium",
            "volatility": "normal",
            "spread": self.spread,
            "ret": self.ret,
            "reason": "",
        }

        if self.spread is None or self.ret is None:
            result["reason"] = "No tick context available"
            return result

        if self.spread > 0.5:
            result["risk"] = "high"
            result["reason"] += "Spread above 0.5. "
        elif self.spread < 0.2:
            result["risk"] = "low"
            result["reason"] += "Spread under 0.2. "

        if abs(self.ret) > 0.001:
            result["volatility"] = "spiky"
            result["risk"] = "high"
            result["reason"] += "High volatility detected. "
        elif abs(self.ret) < 0.0003:
            result["volatility"] = "compressed"
            if result["risk"] != "high":
                result["risk"] = "low"
            result["reason"] += "Low volatility zone. "

        # Adjust recommended risk based on maturity score
        if config:
            base_pct = float(config.get("base_risk_pct", 1.0))
            tiers: Dict[str, float] = config.get("score_risk_tiers", {})
            recommended = base_pct
            if self.maturity_score is not None and tiers:
                # sort thresholds descending
                for th, pct in sorted(((float(k), v) for k, v in tiers.items()), reverse=True):
                    if self.maturity_score >= th:
                        recommended = pct
                        break
            result["recommended_risk_pct"] = recommended

        return result

