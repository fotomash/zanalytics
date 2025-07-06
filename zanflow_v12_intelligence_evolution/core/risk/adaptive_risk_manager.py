"""
Adaptive Risk Manager
====================
Dynamically adjusts position sizing based on maturity scores and market conditions.
Implements multiple risk curves and intelligent position sizing algorithms.
"""

import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging


@dataclass
class RiskProfile:
    """Risk parameters for a specific maturity score"""
    maturity_score: float
    risk_percent: float
    position_size: float
    stop_distance: float


class RiskCurve:
    """Base class for risk curves"""

    def calculate_risk(self, maturity_score: float) -> float:
        raise NotImplementedError


class SteppedRiskCurve(RiskCurve):
    """Stepped risk curve with discrete levels"""

    def __init__(self, levels: List[Dict[str, float]]):
        self.levels = sorted(levels, key=lambda x: x["maturity_min"], reverse=True)

    def calculate_risk(self, maturity_score: float) -> float:
        for level in self.levels:
            if level["maturity_min"] <= maturity_score <= level["maturity_max"]:
                return level["risk_percent"]
        return 0.0


class LinearRiskCurve(RiskCurve):
    """Linear interpolation between min and max risk"""

    def __init__(self, min_maturity: float, max_maturity: float,
                 min_risk: float, max_risk: float):
        self.min_maturity = min_maturity
        self.max_maturity = max_maturity
        self.min_risk = min_risk
        self.max_risk = max_risk

    def calculate_risk(self, maturity_score: float) -> float:
        if maturity_score < self.min_maturity:
            return 0.0
        if maturity_score > self.max_maturity:
            return self.max_risk

        # Linear interpolation
        score_range = self.max_maturity - self.min_maturity
        risk_range = self.max_risk - self.min_risk
        normalized_score = (maturity_score - self.min_maturity) / score_range

        return self.min_risk + (normalized_score * risk_range)


class ExponentialRiskCurve(RiskCurve):
    """Exponential risk curve for aggressive scaling"""

    def __init__(self, base: float, min_maturity: float, 
                 max_risk: float, scaling_factor: float):
        self.base = base
        self.min_maturity = min_maturity
        self.max_risk = max_risk
        self.scaling_factor = scaling_factor

    def calculate_risk(self, maturity_score: float) -> float:
        if maturity_score < self.min_maturity:
            return 0.0

        # Exponential scaling
        adjusted_score = (maturity_score - self.min_maturity) * self.scaling_factor
        risk = (self.base ** adjusted_score - 1) / (self.base - 1) * self.max_risk

        return min(risk, self.max_risk)


class AdaptiveRiskManager:
    """
    Intelligent risk management system that adapts position sizing
    based on maturity scores, market conditions, and account state.
    """

    def __init__(self, config_path: str = "adaptive_risk_config.yaml"):
        self.config = self._load_config(config_path)
        self.risk_curves = self._initialize_risk_curves()
        self.current_curve = "conservative"
        self.position_tracker = {}
        self.daily_risk_used = 0.0
        self.last_reset = datetime.now().date()
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(self, 
                              symbol: str,
                              maturity_score: float,
                              stop_distance_pips: float,
                              account_balance: float,
                              current_conditions: Optional[Dict[str, Any]] = None) -> RiskProfile:
        """
        Calculate optimal position size based on maturity score and conditions.

        Args:
            symbol: Trading symbol
            maturity_score: Current maturity score (0-1)
            stop_distance_pips: Stop loss distance in pips
            account_balance: Current account balance
            current_conditions: Market conditions (killzone, news, etc.)

        Returns:
            RiskProfile with position sizing details
        """
        # Reset daily risk if new day
        self._check_daily_reset()

        # Get base risk percentage from curve
        base_risk = self.risk_curves[self.current_curve].calculate_risk(maturity_score)

        # Apply condition-based adjustments
        adjusted_risk = self._apply_risk_adjustments(base_risk, current_conditions)

        # Check daily risk limit
        adjusted_risk = self._enforce_daily_risk_limit(adjusted_risk)

        # Calculate position size
        risk_amount = account_balance * (adjusted_risk / 100)

        # Adjust for correlation if multiple positions
        if self._check_correlation_adjustment(symbol):
            risk_amount *= 0.7  # Reduce risk for correlated positions

        # Calculate actual position size based on stop distance
        # This is simplified - in practice, you'd use proper pip values
        position_size = risk_amount / stop_distance_pips

        # Apply volatility adjustment
        if current_conditions and "volatility" in current_conditions:
            volatility_factor = self._calculate_volatility_adjustment(
                current_conditions["volatility"]
            )
            position_size *= volatility_factor

        # Create risk profile
        profile = RiskProfile(
            maturity_score=maturity_score,
            risk_percent=adjusted_risk,
            position_size=round(position_size, 2),
            stop_distance=stop_distance_pips
        )

        # Log the decision
        self.logger.info(
            f"Risk calculation for {symbol}: "
            f"Maturity={maturity_score:.2f}, "
            f"Risk={adjusted_risk:.2f}%, "
            f"Size={position_size:.2f}"
        )

        return profile

    def update_risk_curve(self, new_curve: str):
        """Switch to a different risk curve"""
        if new_curve in self.risk_curves:
            self.current_curve = new_curve
            self.logger.info(f"Switched to {new_curve} risk curve")
        else:
            raise ValueError(f"Unknown risk curve: {new_curve}")

    def register_position(self, symbol: str, risk_used: float):
        """Register a new position for tracking"""
        self.position_tracker[symbol] = {
            "risk": risk_used,
            "timestamp": datetime.now()
        }
        self.daily_risk_used += risk_used

    def close_position(self, symbol: str):
        """Remove a closed position from tracking"""
        if symbol in self.position_tracker:
            self.daily_risk_used -= self.position_tracker[symbol]["risk"]
            del self.position_tracker[symbol]

    def get_available_risk(self, account_balance: float) -> float:
        """Calculate remaining risk available for today"""
        max_daily_risk = self.config["adaptive_risk_config"]["overrides"]["max_daily_risk"]
        max_risk_amount = account_balance * (max_daily_risk / 100)

        return max(0, max_risk_amount - self.daily_risk_used)

    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate a comprehensive risk report"""
        return {
            "current_curve": self.current_curve,
            "active_positions": len(self.position_tracker),
            "daily_risk_used": self.daily_risk_used,
            "positions": dict(self.position_tracker),
            "timestamp": datetime.now().isoformat()
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_risk_curves(self) -> Dict[str, RiskCurve]:
        """Initialize risk curves from configuration"""
        curves = {}
        curve_configs = self.config["adaptive_risk_config"]["risk_curves"]

        for name, curve_config in curve_configs.items():
            if curve_config["curve_type"] == "stepped":
                curves[name] = SteppedRiskCurve(curve_config["levels"])
            elif curve_config["curve_type"] == "linear":
                curves[name] = LinearRiskCurve(
                    curve_config["min_maturity"],
                    curve_config["max_maturity"],
                    curve_config["min_risk"],
                    curve_config["max_risk"]
                )
            elif curve_config["curve_type"] == "exponential":
                curves[name] = ExponentialRiskCurve(
                    curve_config["base"],
                    curve_config["min_maturity"],
                    curve_config["max_risk"],
                    curve_config["scaling_factor"]
                )

        return curves

    def _apply_risk_adjustments(self, base_risk: float, 
                               conditions: Optional[Dict[str, Any]]) -> float:
        """Apply condition-based risk adjustments"""
        if not conditions:
            return base_risk

        adjusted_risk = base_risk
        overrides = self.config["adaptive_risk_config"]["overrides"]

        # Killzone multiplier
        if conditions.get("killzone_active", False):
            adjusted_risk *= overrides["killzone_multiplier"]

        # News event reduction
        if conditions.get("high_impact_news", False):
            adjusted_risk *= overrides["news_event_reduction"]

        return adjusted_risk

    def _enforce_daily_risk_limit(self, risk_percent: float) -> float:
        """Ensure daily risk limit is not exceeded"""
        max_daily = self.config["adaptive_risk_config"]["overrides"]["max_daily_risk"]

        if self.daily_risk_used + risk_percent > max_daily:
            return max(0, max_daily - self.daily_risk_used)

        return risk_percent

    def _check_correlation_adjustment(self, symbol: str) -> bool:
        """Check if correlation adjustment should be applied"""
        if not self.config["adaptive_risk_config"]["position_sizing"]["correlation_adjustment"]:
            return False

        # Simple correlation check - in practice, this would be more sophisticated
        base_currency = symbol[:3]
        for existing_symbol in self.position_tracker:
            if base_currency in existing_symbol:
                return True

        return False

    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate position size adjustment based on volatility"""
        if not self.config["adaptive_risk_config"]["position_sizing"]["volatility_adjustment"]:
            return 1.0

        # Inverse volatility scaling
        # Higher volatility = smaller position
        baseline_volatility = 1.0  # This would be configured
        return baseline_volatility / max(0.5, volatility)

    def _check_daily_reset(self):
        """Reset daily risk tracking if new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_risk_used = 0.0
            self.last_reset = current_date


# Usage Example
if __name__ == "__main__":
    # Initialize the adaptive risk manager
    risk_manager = AdaptiveRiskManager()

    # Calculate position size for a high-confidence setup
    conditions = {
        "killzone_active": True,
        "high_impact_news": False,
        "volatility": 0.8
    }

    risk_profile = risk_manager.calculate_position_size(
        symbol="EURUSD",
        maturity_score=0.88,
        stop_distance_pips=15,
        account_balance=100000,
        current_conditions=conditions
    )

    print(f"Position Size: {risk_profile.position_size} lots")
    print(f"Risk Percentage: {risk_profile.risk_percent}%")
