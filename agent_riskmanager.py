"""Risk management utilities for evaluating volatility and spread."""

from typing import Any, Dict


class RiskManagerAgent:
    """Basic risk evaluation based on microstructure context."""

    def __init__(self, context: Dict[str, Any] | None = None) -> None:
        self.context = context or {}
        self.symbol = self.context.get("symbol", "XAUUSD")
        self.micro_context = self.context.get("micro_context", {})
        self.confidence = self.context.get("confidence", 0.0)
        self.maturity_score = self.context.get("maturity_score")
        self.spread = None
        self.ret = None
        self.tick_count = (
            len(self.micro_context) if hasattr(self.micro_context, "__len__") else 0
        )

        if hasattr(self.micro_context, "empty") and not self.micro_context.empty:
            last = self.micro_context.iloc[-1]
            self.spread = last.get("SPREAD")
            self.ret = last.get("RET")

    def evaluate_risk_profile(self, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Return a simplified risk classification and risk percent."""
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

        if config:
            base_pct = float(config.get("base_risk_pct", 1.0))
            tiers: Dict[str, float] = config.get("score_risk_tiers", {})
            recommended = base_pct
            if self.maturity_score is not None and tiers:
                for th, pct in sorted(((float(k), v) for k, v in tiers.items()), reverse=True):
                    if self.maturity_score >= th:
                        recommended = pct
                        break
            result["recommended_risk_pct"] = recommended

        return result

