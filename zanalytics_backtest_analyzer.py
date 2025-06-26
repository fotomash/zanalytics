import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BacktestAnalyzer:
    """Advanced backtesting analytics and reporting"""

    def __init__(self):
        self.metrics = {}
        self.trades = []
        self.equity_curve = []

    def analyze_results(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive backtest analysis"""

        analysis = {
            "performance_metrics": self._calculate_performance_metrics(backtest_results),
            "risk_metrics": self._calculate_risk_metrics(backtest_results),
            "trade_analysis": self._analyze_trades(backtest_results),
            "drawdown_analysis": self._analyze_drawdowns(backtest_results),
            "monthly_returns": self._calculate_monthly_returns(backtest_results),
            "win_loss_analysis": self._analyze_win_loss(backtest_results),
            "optimization_results": self._analyze_optimization(backtest_results)
        }

        return analysis

    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key performance metrics"""

        equity_curve = results.get('equity_curve', [])
        returns = pd.Series(equity_curve).pct_change().dropna()

        metrics = {
            "total_return": (equity_curve[-1] / equity_curve[0] - 1) * 100 if equity_curve else 0,
            "annual_return": self._calculate_annual_return(returns),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns),
            "sortino_ratio": self._calculate_sortino_ratio(returns),
            "calmar_ratio": self._calculate_calmar_ratio(returns, equity_curve),
            "win_rate": results.get('win_rate', 0),
            "profit_factor": results.get('profit_factor', 0),
            "expectancy": results.get('expectancy', 0),
            "max_consecutive_wins": results.get('max_consecutive_wins', 0),
            "max_consecutive_losses": results.get('max_consecutive_losses', 0)
        }

        return metrics

    def _calculate_risk_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk-related metrics"""

        equity_curve = results.get('equity_curve', [])
        returns = pd.Series(equity_curve).pct_change().dropna()

        metrics = {
            "max_drawdown": results.get('max_drawdown', 0),
            "average_drawdown": self._calculate_avg_drawdown(equity_curve),
            "drawdown_duration": results.get('max_drawdown_duration', 0),
            "volatility": returns.std() * np.sqrt(252),
            "downside_deviation": self._calculate_downside_deviation(returns),
            "var_95": np.percentile(returns, 5),
            "cvar_95": returns[returns <= np.percentile(returns, 5)].mean(),
            "omega_ratio": self._calculate_omega_ratio(returns),
            "tail_ratio": self._calculate_tail_ratio(returns)
        }

        return metrics

    def _analyze_trades(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual trades"""

        trades = results.get('trades', [])

        if not trades:
            return {}

        trade_returns = [t.get('return', 0) for t in trades]
        trade_durations = [t.get('duration', 0) for t in trades]

        analysis = {
            "total_trades": len(trades),
            "avg_trade_return": np.mean(trade_returns),
            "median_trade_return": np.median(trade_returns),
            "best_trade": max(trade_returns),
            "worst_trade": min(trade_returns),
            "avg_trade_duration": np.mean(trade_durations),
            "avg_winning_trade": np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0,
            "avg_losing_trade": np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0,
            "trade_frequency": len(trades) / (results.get('trading_days', 252) / 252)  # trades per year
        }

        return analysis

    def _analyze_drawdowns(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze drawdown periods"""

        equity_curve = pd.Series(results.get('equity_curve', []))

        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0

        for i in range(len(drawdown)):
            if drawdown[i] < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif drawdown[i] == 0 and in_drawdown:
                in_drawdown = False
                period = {
                    "start": start_idx,
                    "end": i,
                    "duration": i - start_idx,
                    "max_drawdown": drawdown[start_idx:i].min(),
                    "recovery_time": i - start_idx
                }
                drawdown_periods.append(period)

        # Sort by drawdown magnitude
        drawdown_periods.sort(key=lambda x: x['max_drawdown'])

        return drawdown_periods[:10]  # Top 10 drawdowns

    def _calculate_monthly_returns(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate monthly returns"""

        dates = pd.to_datetime(results.get('dates', []))
        returns = pd.Series(results.get('returns', []), index=dates)

        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        return {
            str(date.strftime('%Y-%m')): ret 
            for date, ret in monthly_returns.items()
        }

    def _analyze_win_loss(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze win/loss patterns"""

        trades = results.get('trades', [])

        winning_trades = [t for t in trades if t.get('return', 0) > 0]
        losing_trades = [t for t in trades if t.get('return', 0) <= 0]

        analysis = {
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades) if trades else 0,
            "avg_win": np.mean([t['return'] for t in winning_trades]) if winning_trades else 0,
            "avg_loss": np.mean([t['return'] for t in losing_trades]) if losing_trades else 0,
            "win_loss_ratio": abs(np.mean([t['return'] for t in winning_trades]) / np.mean([t['return'] for t in losing_trades])) 
                             if winning_trades and losing_trades else 0,
            "largest_win": max([t['return'] for t in winning_trades]) if winning_trades else 0,
            "largest_loss": min([t['return'] for t in losing_trades]) if losing_trades else 0
        }

        return analysis

    def _analyze_optimization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results if available"""

        optimization_results = results.get('optimization_results', {})

        if not optimization_results:
            return {}

        return {
            "best_parameters": optimization_results.get('best_params', {}),
            "best_sharpe": optimization_results.get('best_sharpe', 0),
            "parameter_sensitivity": optimization_results.get('parameter_sensitivity', {}),
            "robust_parameters": self._find_robust_parameters(optimization_results)
        }

    def _find_robust_parameters(self, opt_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find robust parameter sets"""

        all_results = opt_results.get('all_results', [])

        if not all_results:
            return {}

        # Find parameters that consistently perform well
        sorted_results = sorted(all_results, key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        top_20_percent = sorted_results[:int(len(sorted_results) * 0.2)]

        # Find most common parameter values in top performers
        param_counts = {}
        for result in top_20_percent:
            for param, value in result.get('parameters', {}).items():
                if param not in param_counts:
                    param_counts[param] = {}
                param_counts[param][value] = param_counts[param].get(value, 0) + 1

        robust_params = {}
        for param, values in param_counts.items():
            robust_params[param] = max(values, key=values.get)

        return robust_params

    def _calculate_annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        return (1 + total_return) ** (1/years) - 1 if years > 0 else 0

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0

    def _calculate_calmar_ratio(self, returns: pd.Series, equity_curve: List[float]) -> float:
        """Calculate Calmar ratio"""
        annual_return = self._calculate_annual_return(returns)
        max_dd = self._calculate_max_drawdown(equity_curve)
        return annual_return / abs(max_dd) if max_dd != 0 else 0

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0

        running_max = pd.Series(equity_curve).expanding().max()
        drawdown = (pd.Series(equity_curve) - running_max) / running_max
        return drawdown.min()

    def _calculate_avg_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate average drawdown"""
        if not equity_curve:
            return 0

        running_max = pd.Series(equity_curve).expanding().max()
        drawdown = (pd.Series(equity_curve) - running_max) / running_max
        return drawdown[drawdown < 0].mean()

    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation"""
        negative_returns = returns[returns < 0]
        return negative_returns.std() * np.sqrt(252)

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        positive_returns = returns[returns > threshold] - threshold
        negative_returns = threshold - returns[returns <= threshold]

        if negative_returns.sum() == 0:
            return np.inf

        return positive_returns.sum() / negative_returns.sum()

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio"""
        return abs(np.percentile(returns, 95) / np.percentile(returns, 5))

    def generate_report(self, analysis: Dict[str, Any], output_file: str = "backtest_report.json"):
        """Generate comprehensive backtest report"""

        report = {
            "generated_at": datetime.now().isoformat(),
            "performance_summary": analysis.get("performance_metrics", {}),
            "risk_summary": analysis.get("risk_metrics", {}),
            "trade_summary": analysis.get("trade_analysis", {}),
            "drawdown_analysis": analysis.get("drawdown_analysis", []),
            "monthly_returns": analysis.get("monthly_returns", {}),
            "win_loss_analysis": analysis.get("win_loss_analysis", {}),
            "optimization_summary": analysis.get("optimization_results", {}),
            "recommendations": self._generate_recommendations(analysis)
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on backtest results"""

        recommendations = []

        perf = analysis.get("performance_metrics", {})
        risk = analysis.get("risk_metrics", {})
        trades = analysis.get("trade_analysis", {})

        # Performance recommendations
        if perf.get("sharpe_ratio", 0) < 1:
            recommendations.append("Low Sharpe ratio. Consider improving risk-adjusted returns.")

        if perf.get("win_rate", 0) < 0.4:
            recommendations.append("Low win rate. Review entry criteria and timing.")

        # Risk recommendations
        if risk.get("max_drawdown", 0) < -0.2:
            recommendations.append("High maximum drawdown. Implement stricter risk management.")

        if risk.get("volatility", 0) > 0.3:
            recommendations.append("High volatility. Consider position sizing adjustments.")

        # Trade recommendations
        if trades.get("avg_losing_trade", 0) < trades.get("avg_winning_trade", 0) * -2:
            recommendations.append("Losses too large relative to wins. Tighten stop losses.")

        if trades.get("trade_frequency", 0) > 500:
            recommendations.append("Very high trade frequency. May be overtrading.")

        return recommendations

# Save the analyzer
analyzer = BacktestAnalyzer()
print("Created: zanalytics_backtest_analyzer.py")
