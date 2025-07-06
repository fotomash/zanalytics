"""
Meta-Agent for Learning and Optimization
========================================
An intelligent agent that analyzes system performance, learns from historical data,
and provides data-driven recommendations for continuous improvement.
"""

import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import scipy.stats as stats
from sklearn.metrics import matthews_corrcoef
import logging


@dataclass
class PerformanceMetrics:
    """Core performance metrics for analysis"""
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_r_multiple: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    expectancy: float


@dataclass
class OptimizationRecommendation:
    """Recommendation for system optimization"""
    component: str
    parameter: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: float
    rationale: str


class MetaAgent:
    """
    The learning and optimization meta-agent that analyzes system performance
    and provides data-driven recommendations for improvement.
    """

    def __init__(self, config_path: str = "meta_agent_config.yaml",
                 journal_path: str = "zbar_journal.json"):
        self.config = self._load_config(config_path)
        self.journal_path = journal_path
        self.logger = logging.getLogger(__name__)
        self.analysis_results = {}
        self.recommendations = []

    def run_weekly_analysis(self) -> Dict[str, Any]:
        """
        Execute the complete weekly analysis pipeline.

        Returns:
            Comprehensive analysis results and recommendations
        """
        self.logger.info("Starting weekly meta-agent analysis")

        # Load recent trading data
        trading_data = self._load_trading_data()

        # Run all enabled analysis modules
        if self.config["meta_agent_config"]["analysis_modules"]["confluence_path_analyzer"]["enabled"]:
            self.analysis_results["confluence_paths"] = self._analyze_confluence_paths(trading_data)

        if self.config["meta_agent_config"]["analysis_modules"]["rejection_analyzer"]["enabled"]:
            self.analysis_results["rejections"] = self._analyze_rejections(trading_data)

        if self.config["meta_agent_config"]["analysis_modules"]["maturity_correlation_analyzer"]["enabled"]:
            self.analysis_results["maturity_correlation"] = self._analyze_maturity_correlation(trading_data)

        if self.config["meta_agent_config"]["analysis_modules"]["risk_optimizer"]["enabled"]:
            self.analysis_results["risk_optimization"] = self._optimize_risk_parameters(trading_data)

        # Generate optimization recommendations
        self.recommendations = self._generate_recommendations()

        # Create comprehensive report
        report = self._create_analysis_report()

        # Execute configured actions
        self._execute_actions(report)

        return report

    def _analyze_confluence_paths(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance of different confluence paths"""
        self.logger.info("Analyzing confluence paths")

        # Group by path signature
        path_performance = defaultdict(lambda: {
            "trades": 0,
            "wins": 0,
            "total_r": 0.0,
            "max_r": -float('inf'),
            "min_r": float('inf'),
            "durations": []
        })

        min_samples = self.config["meta_agent_config"]["analysis_modules"]["confluence_path_analyzer"]["min_samples"]

        for _, trade in data.iterrows():
            if "path_signature" in trade and pd.notna(trade["path_signature"]):
                sig = trade["path_signature"]
                perf = path_performance[sig]

                perf["trades"] += 1
                if trade.get("r_multiple", 0) > 0:
                    perf["wins"] += 1

                r_multiple = trade.get("r_multiple", 0)
                perf["total_r"] += r_multiple
                perf["max_r"] = max(perf["max_r"], r_multiple)
                perf["min_r"] = min(perf["min_r"], r_multiple)

                # Calculate duration if available
                if "confluence_path" in trade:
                    path = trade["confluence_path"]
                    if isinstance(path, list) and len(path) > 1:
                        duration = (pd.to_datetime(path[-1]["timestamp"]) - 
                                  pd.to_datetime(path[0]["timestamp"])).total_seconds() / 60
                        perf["durations"].append(duration)

        # Calculate statistics for paths with sufficient samples
        analyzed_paths = {}
        for sig, perf in path_performance.items():
            if perf["trades"] >= min_samples:
                win_rate = perf["wins"] / perf["trades"]
                avg_r = perf["total_r"] / perf["trades"]

                # Calculate Sharpe-like metric
                if perf["trades"] > 1:
                    r_values = []  # Would need to store individual R values
                    sharpe_approx = avg_r * np.sqrt(perf["trades"]) / max(0.1, perf["max_r"] - perf["min_r"])
                else:
                    sharpe_approx = 0

                analyzed_paths[sig] = {
                    "trades": perf["trades"],
                    "win_rate": win_rate,
                    "avg_r": avg_r,
                    "sharpe_approximation": sharpe_approx,
                    "avg_duration_minutes": np.mean(perf["durations"]) if perf["durations"] else 0,
                    "performance_score": win_rate * avg_r  # Simple performance metric
                }

        # Identify top and bottom performers
        sorted_paths = sorted(analyzed_paths.items(), 
                            key=lambda x: x[1]["performance_score"], 
                            reverse=True)

        return {
            "total_unique_paths": len(path_performance),
            "analyzed_paths": len(analyzed_paths),
            "top_performers": dict(sorted_paths[:5]),
            "bottom_performers": dict(sorted_paths[-5:]),
            "path_details": analyzed_paths
        }

    def _analyze_rejections(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze effectiveness of rejection rules"""
        self.logger.info("Analyzing rejection effectiveness")

        rejection_stats = defaultdict(lambda: {
            "total_rejections": 0,
            "false_negatives": 0,  # Rejected setups that would have won
            "opportunity_cost_r": 0.0,
            "true_negatives": 0,   # Correctly rejected losing setups
            "savings_r": 0.0
        })

        # Analyze both executed and rejected trades
        for _, entry in data.iterrows():
            if "rejection_reason" in entry and pd.notna(entry["rejection_reason"]):
                reason = entry["rejection_reason"]
                stats = rejection_stats[reason]
                stats["total_rejections"] += 1

                # If we have simulated results for rejected trades
                if "simulated_r" in entry:
                    sim_r = entry["simulated_r"]
                    if sim_r > 0:
                        stats["false_negatives"] += 1
                        stats["opportunity_cost_r"] += sim_r
                    else:
                        stats["true_negatives"] += 1
                        stats["savings_r"] += abs(sim_r)

        # Calculate effectiveness metrics
        rejection_effectiveness = {}
        for reason, stats in rejection_stats.items():
            if stats["total_rejections"] > 0:
                effectiveness = stats["true_negatives"] / stats["total_rejections"]
                net_impact = stats["savings_r"] - stats["opportunity_cost_r"]

                rejection_effectiveness[reason] = {
                    "total_rejections": stats["total_rejections"],
                    "effectiveness_rate": effectiveness,
                    "false_negative_rate": stats["false_negatives"] / stats["total_rejections"],
                    "net_r_impact": net_impact,
                    "recommendation": "Keep" if net_impact > 0 else "Review"
                }

        return {
            "total_rejections": sum(s["total_rejections"] for s in rejection_stats.values()),
            "rejection_rules": rejection_effectiveness,
            "total_opportunity_cost": sum(s["opportunity_cost_r"] for s in rejection_stats.values()),
            "total_savings": sum(s["savings_r"] for s in rejection_stats.values())
        }

    def _analyze_maturity_correlation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation between maturity scores and outcomes"""
        self.logger.info("Analyzing maturity score correlations")

        # Filter for executed trades with maturity scores
        executed_trades = data[
            (data["executed"] == True) & 
            (data["final_maturity_score"].notna())
        ].copy()

        if len(executed_trades) < 10:
            return {"error": "Insufficient data for correlation analysis"}

        # Calculate correlations
        correlations = {}

        # Maturity vs R-multiple
        if "r_multiple" in executed_trades.columns:
            correlations["maturity_vs_r"] = {
                "pearson": executed_trades["final_maturity_score"].corr(
                    executed_trades["r_multiple"], method="pearson"
                ),
                "spearman": executed_trades["final_maturity_score"].corr(
                    executed_trades["r_multiple"], method="spearman"
                )
            }

        # Maturity score bins vs win rate
        bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        executed_trades["maturity_bin"] = pd.cut(
            executed_trades["final_maturity_score"], 
            bins=bins,
            labels=["0.0-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
        )

        bin_performance = {}
        for bin_label in executed_trades["maturity_bin"].unique():
            if pd.notna(bin_label):
                bin_trades = executed_trades[executed_trades["maturity_bin"] == bin_label]
                if len(bin_trades) > 0:
                    wins = (bin_trades["r_multiple"] > 0).sum()
                    bin_performance[str(bin_label)] = {
                        "trades": len(bin_trades),
                        "win_rate": wins / len(bin_trades),
                        "avg_r": bin_trades["r_multiple"].mean()
                    }

        # Analyze time decay effectiveness
        time_decay_analysis = self._analyze_time_decay(executed_trades)

        return {
            "correlations": correlations,
            "maturity_bin_performance": bin_performance,
            "time_decay_analysis": time_decay_analysis,
            "optimal_maturity_threshold": self._find_optimal_maturity_threshold(executed_trades)
        }

    def _optimize_risk_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize risk parameters using Kelly Criterion and other methods"""
        self.logger.info("Optimizing risk parameters")

        # Get recent performance metrics
        recent_trades = data[data["timestamp"] > (datetime.now() - timedelta(days=90))]

        if len(recent_trades) < 20:
            return {"error": "Insufficient recent data for risk optimization"}

        # Calculate win rate and average win/loss
        wins = recent_trades[recent_trades["r_multiple"] > 0]
        losses = recent_trades[recent_trades["r_multiple"] <= 0]

        win_rate = len(wins) / len(recent_trades)
        avg_win = wins["r_multiple"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["r_multiple"].mean()) if len(losses) > 0 else 1

        # Kelly Criterion calculation
        if avg_loss > 0:
            kelly_percentage = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_percentage = max(0, min(kelly_percentage, 0.25))  # Cap at 25%
        else:
            kelly_percentage = 0

        # Fractional Kelly (more conservative)
        fractional_kelly = kelly_percentage * 0.25  # Use 25% of Kelly

        # Analyze risk by maturity score bins
        risk_by_maturity = {}
        for bin_range in [(0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]:
            bin_trades = recent_trades[
                (recent_trades["final_maturity_score"] >= bin_range[0]) &
                (recent_trades["final_maturity_score"] < bin_range[1])
            ]

            if len(bin_trades) > 5:
                bin_wins = bin_trades[bin_trades["r_multiple"] > 0]
                bin_win_rate = len(bin_wins) / len(bin_trades)

                # Calculate optimal risk for this bin
                if len(bin_wins) > 0 and len(bin_trades) > len(bin_wins):
                    bin_avg_win = bin_wins["r_multiple"].mean()
                    bin_avg_loss = abs(bin_trades[bin_trades["r_multiple"] <= 0]["r_multiple"].mean())

                    bin_kelly = (bin_win_rate * bin_avg_win - (1 - bin_win_rate) * bin_avg_loss) / bin_avg_win
                    bin_kelly = max(0, min(bin_kelly, 0.25))
                else:
                    bin_kelly = 0

                risk_by_maturity[f"{bin_range[0]}-{bin_range[1]}"] = {
                    "trades": len(bin_trades),
                    "win_rate": bin_win_rate,
                    "optimal_risk": bin_kelly * 0.25  # Fractional Kelly
                }

        return {
            "overall_statistics": {
                "win_rate": win_rate,
                "avg_win_r": avg_win,
                "avg_loss_r": avg_loss,
                "profit_factor": (win_rate * avg_win) / ((1 - win_rate) * avg_loss) if avg_loss > 0 else 0
            },
            "kelly_criterion": {
                "full_kelly": kelly_percentage,
                "fractional_kelly": fractional_kelly,
                "recommended_max_risk": fractional_kelly
            },
            "risk_by_maturity": risk_by_maturity,
            "current_performance": self._calculate_performance_metrics(recent_trades)
        }

    def _generate_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Confluence path recommendations
        if "confluence_paths" in self.analysis_results:
            paths = self.analysis_results["confluence_paths"]
            if "top_performers" in paths and paths["top_performers"]:
                top_path = list(paths["top_performers"].items())[0]
                recommendations.append(OptimizationRecommendation(
                    component="confluence_validator",
                    parameter="priority_paths",
                    current_value="all",
                    recommended_value=list(paths["top_performers"].keys()),
                    expected_improvement=0.15,
                    confidence=0.85,
                    rationale=f"Top performing paths show {top_path[1]['avg_r']:.2f} avg R-multiple"
                ))

        # Rejection rule recommendations
        if "rejections" in self.analysis_results:
            rejections = self.analysis_results["rejections"]
            for rule, stats in rejections.get("rejection_rules", {}).items():
                if stats["recommendation"] == "Review" and stats["net_r_impact"] < -5:
                    recommendations.append(OptimizationRecommendation(
                        component="rejection_rules",
                        parameter=rule,
                        current_value="enabled",
                        recommended_value="disabled",
                        expected_improvement=abs(stats["net_r_impact"]) / 100,
                        confidence=0.75,
                        rationale=f"Rule has cost {abs(stats['net_r_impact']):.1f}R in opportunity"
                    ))

        # Risk optimization recommendations
        if "risk_optimization" in self.analysis_results:
            risk_opt = self.analysis_results["risk_optimization"]
            if "kelly_criterion" in risk_opt:
                current_risk = 0.01  # Assume 1% default
                recommended_risk = risk_opt["kelly_criterion"]["fractional_kelly"]

                if abs(recommended_risk - current_risk) > 0.002:
                    recommendations.append(OptimizationRecommendation(
                        component="risk_manager",
                        parameter="base_risk_percentage",
                        current_value=current_risk,
                        recommended_value=recommended_risk,
                        expected_improvement=(recommended_risk - current_risk) / current_risk,
                        confidence=0.80,
                        rationale="Optimized using Kelly Criterion on recent performance"
                    ))

        # Maturity threshold recommendations
        if "maturity_correlation" in self.analysis_results:
            maturity = self.analysis_results["maturity_correlation"]
            if "optimal_maturity_threshold" in maturity:
                optimal = maturity["optimal_maturity_threshold"]
                if optimal and optimal > 0.5:
                    recommendations.append(OptimizationRecommendation(
                        component="entry_validator",
                        parameter="min_maturity_score",
                        current_value=0.65,
                        recommended_value=optimal,
                        expected_improvement=0.10,
                        confidence=0.70,
                        rationale=f"Analysis shows better performance above {optimal:.2f} maturity"
                    ))

        return recommendations

    def _create_analysis_report(self) -> Dict[str, Any]:
        """Create comprehensive analysis report"""
        report = {
            "report_id": f"WEEKLY_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "generated_at": datetime.now().isoformat(),
            "analysis_period": {
                "start": (datetime.now() - timedelta(days=7)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "executive_summary": self._generate_executive_summary(),
            "detailed_analysis": self.analysis_results,
            "recommendations": [
                {
                    "component": r.component,
                    "parameter": r.parameter,
                    "current_value": r.current_value,
                    "recommended_value": r.recommended_value,
                    "expected_improvement": f"{r.expected_improvement:.1%}",
                    "confidence": f"{r.confidence:.0%}",
                    "rationale": r.rationale
                }
                for r in self.recommendations
            ],
            "next_steps": self._generate_next_steps()
        }

        return report

    def _generate_executive_summary(self) -> str:
        """Generate executive summary of findings"""
        summary_points = []

        # Confluence path insights
        if "confluence_paths" in self.analysis_results:
            paths = self.analysis_results["confluence_paths"]
            summary_points.append(
                f"Analyzed {paths['analyzed_paths']} unique confluence paths. "
                f"Top performers show significantly higher success rates."
            )

        # Rejection analysis insights
        if "rejections" in self.analysis_results:
            rejections = self.analysis_results["rejections"]
            net_impact = rejections.get("total_savings", 0) - rejections.get("total_opportunity_cost", 0)
            summary_points.append(
                f"Rejection rules filtered {rejections['total_rejections']} setups "
                f"with net impact of {net_impact:.1f}R."
            )

        # Risk optimization insights
        if "risk_optimization" in self.analysis_results:
            risk = self.analysis_results["risk_optimization"]
            if "kelly_criterion" in risk:
                summary_points.append(
                    f"Optimal risk per trade calculated at "
                    f"{risk['kelly_criterion']['fractional_kelly']:.1%} using Kelly Criterion."
                )

        return " ".join(summary_points)

    def _generate_next_steps(self) -> List[str]:
        """Generate actionable next steps"""
        steps = []

        if self.recommendations:
            steps.append("Review and implement high-confidence recommendations")

        if "confluence_paths" in self.analysis_results:
            steps.append("Focus on top-performing confluence paths in upcoming week")

        if "rejections" in self.analysis_results:
            steps.append("Audit rejection rules with negative net impact")

        steps.append("Monitor performance changes after implementing recommendations")

        return steps

    def _execute_actions(self, report: Dict[str, Any]):
        """Execute configured actions based on analysis"""
        output_config = self.config["meta_agent_config"]["output_actions"]

        # Save report
        if output_config["generate_reports"]:
            report_path = f"reports/meta_analysis_{report['report_id']}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Saved analysis report to {report_path}")

        # Update agent configurations (if enabled)
        if output_config["update_agent_configs"] and self.recommendations:
            self._apply_configuration_updates()

        # Send notifications (placeholder)
        if output_config["send_notifications"]:
            self.logger.info("Sending analysis notifications...")
            # Implementation would send email/slack/etc

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load meta-agent configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_trading_data(self) -> pd.DataFrame:
        """Load recent trading data from ZBAR journal"""
        try:
            with open(self.journal_path, 'r') as f:
                journal_data = json.load(f)

            # Convert to DataFrame for analysis
            df = pd.DataFrame(journal_data)

            # Convert timestamp strings to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df
        except Exception as e:
            self.logger.error(f"Error loading trading data: {e}")
            return pd.DataFrame()

    def _analyze_time_decay(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze effectiveness of time decay in maturity scores"""
        # Placeholder for time decay analysis
        return {
            "decay_effectiveness": "Requires temporal maturity score data",
            "optimal_decay_rate": 0.95
        }

    def _find_optimal_maturity_threshold(self, data: pd.DataFrame) -> float:
        """Find optimal maturity score threshold for entry"""
        if len(data) < 20:
            return 0.65  # Default

        # Test different thresholds
        best_threshold = 0.65
        best_score = 0

        for threshold in np.arange(0.5, 0.9, 0.05):
            filtered = data[data["final_maturity_score"] >= threshold]
            if len(filtered) > 5:
                win_rate = (filtered["r_multiple"] > 0).mean()
                avg_r = filtered["r_multiple"].mean()
                score = win_rate * avg_r * np.sqrt(len(filtered))  # Include sample size

                if score > best_score:
                    best_score = score
                    best_threshold = threshold

        return float(best_threshold)

    def _calculate_performance_metrics(self, data: pd.DataFrame) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if len(data) == 0:
            return None

        wins = data[data["r_multiple"] > 0]
        losses = data[data["r_multiple"] <= 0]

        metrics = PerformanceMetrics(
            total_trades=len(data),
            win_rate=len(wins) / len(data) if len(data) > 0 else 0,
            profit_factor=(wins["r_multiple"].sum() / abs(losses["r_multiple"].sum())) 
                         if len(losses) > 0 and losses["r_multiple"].sum() != 0 else 0,
            sharpe_ratio=data["r_multiple"].mean() / data["r_multiple"].std() 
                        if data["r_multiple"].std() > 0 else 0,
            max_drawdown=self._calculate_max_drawdown(data["r_multiple"]),
            avg_r_multiple=data["r_multiple"].mean(),
            avg_win=wins["r_multiple"].mean() if len(wins) > 0 else 0,
            avg_loss=losses["r_multiple"].mean() if len(losses) > 0 else 0,
            largest_win=wins["r_multiple"].max() if len(wins) > 0 else 0,
            largest_loss=losses["r_multiple"].min() if len(losses) > 0 else 0,
            consecutive_wins=self._max_consecutive(data["r_multiple"] > 0),
            consecutive_losses=self._max_consecutive(data["r_multiple"] <= 0),
            recovery_factor=data["r_multiple"].sum() / abs(self._calculate_max_drawdown(data["r_multiple"]))
                           if self._calculate_max_drawdown(data["r_multiple"]) != 0 else 0,
            expectancy=(len(wins) / len(data) * wins["r_multiple"].mean() + 
                       len(losses) / len(data) * losses["r_multiple"].mean())
                      if len(data) > 0 else 0
        )

        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _max_consecutive(self, condition: pd.Series) -> int:
        """Calculate maximum consecutive True values in series"""
        groups = (condition != condition.shift()).cumsum()
        return condition.groupby(groups).sum().max()

    def _apply_configuration_updates(self):
        """Apply recommended configuration updates to agent files"""
        # This would actually update the YAML files
        # For safety, we'll just log what would be updated
        for rec in self.recommendations[:3]:  # Limit to top 3
            if rec.confidence > 0.75:
                self.logger.info(
                    f"Would update {rec.component}.{rec.parameter} "
                    f"from {rec.current_value} to {rec.recommended_value}"
                )


# Usage Example
if __name__ == "__main__":
    # Initialize meta-agent
    meta_agent = MetaAgent()

    # Run weekly analysis
    report = meta_agent.run_weekly_analysis()

    # Print summary
    print("\n=== META-AGENT ANALYSIS COMPLETE ===")
    print(f"Report ID: {report['report_id']}")
    print(f"\nExecutive Summary:\n{report['executive_summary']}")
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"{i}. {rec['component']}: {rec['rationale']}")
        print(f"   Expected improvement: {rec['expected_improvement']}")
