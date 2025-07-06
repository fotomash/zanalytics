"""
Intelligence Evolution Integration Module
========================================
Integrates the enhanced intelligence features into the existing ZANFLOW v12 system.
"""

import yaml
from typing import Dict, Any
from datetime import datetime
import logging

from core.confluence.path_tracker import ConfluencePathTracker
from core.risk.adaptive_risk_manager import AdaptiveRiskManager
from core.meta_agent.optimization_engine import MetaAgent


class IntelligenceIntegration:
    """
    Central integration point for all intelligence evolution features.
    Connects with existing ZANFLOW v12 components.
    """

    def __init__(self, base_config_path: str = "config/orchestrator_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.base_config = self._load_config(base_config_path)

        # Initialize enhanced components
        self.path_tracker = ConfluencePathTracker(
            journal_path=self.base_config.get("journal_path", "zbar_journal.json")
        )

        self.risk_manager = AdaptiveRiskManager(
            config_path="configs/adaptive_risk_config.yaml"
        )

        self.meta_agent = MetaAgent(
            config_path="configs/meta_agent_config.yaml",
            journal_path=self.base_config.get("journal_path", "zbar_journal.json")
        )

        self.logger.info("Intelligence Evolution components initialized")

    def enhance_agent_state(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the standard agent state with intelligence features.

        This method is called by agents to add confluence tracking
        and adaptive risk calculations to their state.
        """
        # Start confluence path tracking if new trade
        if agent_state.get("new_opportunity", False):
            trade_id = agent_state.get("trade_id", f"TRADE_{datetime.now().timestamp()}")
            symbol = agent_state.get("symbol", "UNKNOWN")
            timeframe = agent_state.get("timeframe", "M5")

            path = self.path_tracker.start_path(trade_id, symbol, timeframe)
            agent_state["confluence_path_id"] = trade_id
            agent_state["path_tracking_active"] = True

        # Add confluence events
        if agent_state.get("path_tracking_active", False) and "validation_event" in agent_state:
            event = agent_state["validation_event"]
            self.path_tracker.add_event(
                trade_id=agent_state["confluence_path_id"],
                event_type=event["type"],
                validation_score=event.get("score", 0.0),
                tick_volume=event.get("tick_volume", 0),
                metadata=event.get("metadata", {})
            )

            # Update agent state with path signature
            active_path = self.path_tracker.active_paths.get(agent_state["confluence_path_id"])
            if active_path:
                agent_state["current_confluence_path"] = active_path.get_event_sequence()
                agent_state["path_signature"] = active_path.get_path_signature()

        # Calculate adaptive risk if ready for entry
        if agent_state.get("ready_for_entry", False):
            maturity_score = agent_state.get("predictive_maturity_score", 0.65)

            conditions = {
                "killzone_active": agent_state.get("in_killzone", False),
                "high_impact_news": agent_state.get("news_risk", False),
                "volatility": agent_state.get("current_volatility", 1.0)
            }

            risk_profile = self.risk_manager.calculate_position_size(
                symbol=agent_state.get("symbol", "UNKNOWN"),
                maturity_score=maturity_score,
                stop_distance_pips=agent_state.get("stop_distance", 15),
                account_balance=agent_state.get("account_balance", 100000),
                current_conditions=conditions
            )

            agent_state["risk_profile"] = {
                "risk_percentage": risk_profile.risk_percent,
                "position_size": risk_profile.position_size,
                "risk_curve": self.risk_manager.current_curve
            }

        return agent_state

    def complete_trade_tracking(self, trade_id: str, outcome: Dict[str, Any], 
                               final_maturity_score: float):
        """
        Complete confluence path tracking for a finished trade.

        Called when a trade is closed or a setup is abandoned.
        """
        if trade_id in self.path_tracker.active_paths:
            self.path_tracker.complete_path(trade_id, final_maturity_score, outcome)
            self.logger.info(f"Completed tracking for {trade_id}")

    def get_path_recommendations(self) -> Dict[str, Any]:
        """
        Get real-time recommendations based on confluence path performance.

        Can be called by agents to prioritize certain setups.
        """
        top_paths = self.path_tracker.get_optimal_paths(metric="sharpe", top_n=5)

        recommendations = {
            "prioritize_paths": [path[0] for path in top_paths],
            "path_performance": {
                path[0]: {
                    "win_rate": path[1]["win_rate"],
                    "avg_r": path[1]["avg_r"],
                    "trades": path[1]["trades"]
                }
                for path in top_paths
            }
        }

        return recommendations

    def update_risk_curve(self, new_curve: str):
        """
        Update the active risk curve based on market conditions or performance.

        Args:
            new_curve: One of 'conservative', 'moderate', 'aggressive'
        """
        self.risk_manager.update_risk_curve(new_curve)
        self.logger.info(f"Updated risk curve to: {new_curve}")

    def run_optimization_cycle(self):
        """
        Run the meta-agent optimization cycle.

        This should be scheduled weekly or on-demand.
        """
        self.logger.info("Starting optimization cycle")

        # Run meta-agent analysis
        report = self.meta_agent.run_weekly_analysis()

        # Apply high-confidence recommendations automatically
        for rec in report.get("recommendations", []):
            if rec["confidence"] > "80%" and rec["component"] == "risk_manager":
                # Auto-apply risk adjustments
                self.logger.info(f"Auto-applying risk adjustment: {rec}")
                # Implementation would update configs

        return report

    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get current status of all intelligence components"""
        return {
            "confluence_tracking": {
                "active_paths": len(self.path_tracker.active_paths),
                "historical_paths": len(self.path_tracker.path_statistics),
                "top_performers": self.path_tracker.get_optimal_paths(top_n=3)
            },
            "risk_management": {
                "current_curve": self.risk_manager.current_curve,
                "active_positions": len(self.risk_manager.position_tracker),
                "daily_risk_used": f"{self.risk_manager.daily_risk_used:.1%}",
                "available_risk": self.risk_manager.get_available_risk(100000)
            },
            "optimization": {
                "last_analysis": self.meta_agent.analysis_results.get("generated_at", "Never"),
                "pending_recommendations": len(self.meta_agent.recommendations)
            }
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load base configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}


# Create API endpoints for the enhanced intelligence
class IntelligenceAPI:
    """API endpoints for intelligence features"""

    def __init__(self, integration: IntelligenceIntegration):
        self.integration = integration

    def enhance_state(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/intelligence/enhance-state"""
        return self.integration.enhance_agent_state(agent_state)

    def complete_tracking(self, trade_id: str, outcome: Dict[str, Any], 
                         maturity_score: float) -> Dict[str, Any]:
        """POST /api/intelligence/complete-tracking"""
        self.integration.complete_trade_tracking(trade_id, outcome, maturity_score)
        return {"status": "completed", "trade_id": trade_id}

    def get_recommendations(self) -> Dict[str, Any]:
        """GET /api/intelligence/recommendations"""
        return self.integration.get_path_recommendations()

    def update_risk_curve(self, curve: str) -> Dict[str, Any]:
        """PUT /api/intelligence/risk-curve"""
        self.integration.update_risk_curve(curve)
        return {"status": "updated", "new_curve": curve}

    def run_optimization(self) -> Dict[str, Any]:
        """POST /api/intelligence/optimize"""
        return self.integration.run_optimization_cycle()

    def get_status(self) -> Dict[str, Any]:
        """GET /api/intelligence/status"""
        return self.integration.get_intelligence_status()
