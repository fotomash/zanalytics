"""
ZAnalytics Advanced Analytics Module
Implements advanced trading analytics including:
- Market regime detection
- Volatility analysis
- Correlation analysis
- Risk metrics
- Performance analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """Detects market regimes using various methods"""

    def __init__(self):
        self.regimes = {
            'trending_up': {'volatility': 'normal', 'direction': 'bullish'},
            'trending_down': {'volatility': 'normal', 'direction': 'bearish'},
            'ranging': {'volatility': 'low', 'direction': 'neutral'},
            'volatile': {'volatility': 'high', 'direction': 'uncertain'}
        }

    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime"""
        try:
            # Calculate indicators
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['trend'] = df['close'].rolling(50).mean()

            # Current values
            current_price = df['close'].iloc[-1]
            current_vol = df['volatility'].iloc[-1]
            avg_vol = df['volatility'].mean()
            trend_slope = np.polyfit(range(20), df['close'].tail(20).values, 1)[0]

            # Determine regime
            if current_vol > avg_vol * 1.5:
                regime = 'volatile'
            elif abs(trend_slope) < 0.001:
                regime = 'ranging'
            elif trend_slope > 0:
                regime = 'trending_up'
            else:
                regime = 'trending_down'

            return {
                'regime': regime,
                'characteristics': self.regimes[regime],
                'volatility_ratio': current_vol / avg_vol,
                'trend_strength': abs(trend_slope),
                'confidence': self._calculate_confidence(df)
            }

        except Exception as e:
            return {'error': str(e), 'regime': 'unknown'}

    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        """Calculate confidence in regime detection"""
        try:
            # Multiple confirmations increase confidence
            confirmations = 0

            # Check ADX for trend strength
            if 'adx' in df.columns:
                if df['adx'].iloc[-1] > 25:
                    confirmations += 1

            # Check volume consistency
            if 'volume' in df.columns:
                vol_trend = df['volume'].rolling(10).mean().iloc[-1]
                if vol_trend > df['volume'].mean():
                    confirmations += 1

            # Check price action consistency
            recent_closes = df['close'].tail(10)
            if recent_closes.is_monotonic_increasing or recent_closes.is_monotonic_decreasing:
                confirmations += 1

            return min(0.3 + (confirmations * 0.2), 0.95)

        except:
            return 0.5

class VolatilityAnalyzer:
    """Advanced volatility analysis"""

    def __init__(self):
        self.windows = [10, 20, 50, 100]

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive volatility analysis"""
        try:
            df['returns'] = df['close'].pct_change()

            analysis = {
                'current_volatility': {},
                'historical_volatility': {},
                'volatility_regime': '',
                'risk_metrics': {}
            }

            # Calculate volatility for different windows
            for window in self.windows:
                vol = df['returns'].rolling(window).std().iloc[-1]
                analysis['current_volatility'][f'{window}d'] = vol
                analysis['historical_volatility'][f'{window}d_avg'] = df['returns'].rolling(window).std().mean()

            # Volatility regime
            current_vol = analysis['current_volatility']['20d']
            hist_vol = analysis['historical_volatility']['20d_avg']

            if current_vol < hist_vol * 0.8:
                analysis['volatility_regime'] = 'low'
            elif current_vol > hist_vol * 1.2:
                analysis['volatility_regime'] = 'high'
            else:
                analysis['volatility_regime'] = 'normal'

            # Risk metrics
            analysis['risk_metrics'] = {
                'value_at_risk_95': self._calculate_var(df['returns'], 0.95),
                'conditional_var_95': self._calculate_cvar(df['returns'], 0.95),
                'max_drawdown': self._calculate_max_drawdown(df['close']),
                'sharpe_ratio': self._calculate_sharpe_ratio(df['returns'])
            }

            return analysis

        except Exception as e:
            return {'error': str(e)}

    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns.dropna(), (1 - confidence) * 100)

    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

class CorrelationAnalyzer:
    """Analyze correlations between different assets and indicators"""

    def analyze_correlations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze correlations between multiple assets"""
        try:
            correlations = {}

            # Prepare returns data
            returns_data = {}
            for symbol, df in data.items():
                if 'close' in df.columns:
                    returns_data[symbol] = df['close'].pct_change()

            # Calculate correlation matrix
            returns_df = pd.DataFrame(returns_data)
            corr_matrix = returns_df.corr()

            # Find significant correlations
            significant_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        significant_corrs.append({
                            'pair': f"{corr_matrix.columns[i]}-{corr_matrix.columns[j]}",
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })

            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'significant_correlations': significant_corrs,
                'diversification_score': self._calculate_diversification_score(corr_matrix)
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_diversification_score(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate portfolio diversification score"""
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        return 1 - abs(avg_correlation)

class PerformanceAnalyzer:
    """Analyze trading performance metrics"""

    def analyze_performance(self, trades: List[Dict], initial_capital: float = 10000) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not trades:
                return {'error': 'No trades to analyze'}

            # Convert to DataFrame for easier analysis
            trades_df = pd.DataFrame(trades)

            # Calculate returns
            trades_df['return'] = trades_df['exit_price'] / trades_df['entry_price'] - 1
            trades_df['profit'] = trades_df['size'] * (trades_df['exit_price'] - trades_df['entry_price'])

            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            losing_trades = len(trades_df[trades_df['profit'] < 0])

            # Performance metrics
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_profit': trades_df['profit'].sum(),
                'average_profit': trades_df['profit'].mean(),
                'average_win': trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0,
                'average_loss': trades_df[trades_df['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0,
                'profit_factor': self._calculate_profit_factor(trades_df),
                'max_consecutive_wins': self._max_consecutive(trades_df['profit'] > 0),
                'max_consecutive_losses': self._max_consecutive(trades_df['profit'] < 0),
                'risk_reward_ratio': self._calculate_risk_reward_ratio(trades_df),
                'expectancy': self._calculate_expectancy(trades_df),
                'kelly_criterion': self._calculate_kelly_criterion(trades_df)
            }

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor"""
        gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values"""
        groups = (series != series.shift()).cumsum()
        return series.groupby(groups).sum().max()

    def _calculate_risk_reward_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate average risk/reward ratio"""
        avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean()
        avg_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].mean())
        return avg_win / avg_loss if avg_loss > 0 else float('inf')

    def _calculate_expectancy(self, trades_df: pd.DataFrame) -> float:
        """Calculate trading expectancy"""
        win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df)
        avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if any(trades_df['profit'] > 0) else 0
        avg_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].mean()) if any(trades_df['profit'] < 0) else 0
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def _calculate_kelly_criterion(self, trades_df: pd.DataFrame) -> float:
        """Calculate Kelly Criterion for position sizing"""
        win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df)
        rr_ratio = self._calculate_risk_reward_ratio(trades_df)
        if rr_ratio == float('inf') or rr_ratio == 0:
            return 0
        return (win_rate * rr_ratio - (1 - win_rate)) / rr_ratio

class AdvancedAnalytics:
    """Main class for advanced analytics"""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()

    def analyze(self, market_data: Dict[str, pd.DataFrame], trades: List[Dict] = None) -> Dict[str, Any]:
        """Perform comprehensive advanced analysis"""

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'market_regimes': {},
            'volatility_analysis': {},
            'correlation_analysis': {},
            'performance_metrics': {},
            'risk_assessment': {},
            'recommendations': []
        }

        # Analyze each symbol
        for symbol, df in market_data.items():
            # Market regime
            analysis['market_regimes'][symbol] = self.regime_detector.detect_regime(df)

            # Volatility analysis
            analysis['volatility_analysis'][symbol] = self.volatility_analyzer.analyze(df)

        # Correlation analysis
        if len(market_data) > 1:
            analysis['correlation_analysis'] = self.correlation_analyzer.analyze_correlations(market_data)

        # Performance analysis
        if trades:
            analysis['performance_metrics'] = self.performance_analyzer.analyze_performance(trades)

        # Risk assessment
        analysis['risk_assessment'] = self._assess_overall_risk(analysis)

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _assess_overall_risk(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        risk_score = 0
        risk_factors = []

        # Check volatility
        for symbol, vol_data in analysis['volatility_analysis'].items():
            if vol_data.get('volatility_regime') == 'high':
                risk_score += 2
                risk_factors.append(f"High volatility in {symbol}")

        # Check correlations
        if 'correlation_analysis' in analysis:
            high_corrs = [c for c in analysis['correlation_analysis'].get('significant_correlations', []) 
                         if c['correlation'] > 0.8]
            if high_corrs:
                risk_score += len(high_corrs)
                risk_factors.append(f"High correlations detected: {len(high_corrs)} pairs")

        # Check performance
        if 'performance_metrics' in analysis:
            metrics = analysis['performance_metrics']
            if metrics.get('max_consecutive_losses', 0) > 5:
                risk_score += 2
                risk_factors.append("High consecutive losses")

        return {
            'risk_score': risk_score,
            'risk_level': 'high' if risk_score > 5 else 'medium' if risk_score > 2 else 'low',
            'risk_factors': risk_factors
        }

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []

        # Market regime recommendations
        for symbol, regime_data in analysis['market_regimes'].items():
            regime = regime_data.get('regime')
            if regime == 'volatile':
                recommendations.append(f"Consider reducing position size in {symbol} due to high volatility")
            elif regime == 'trending_up':
                recommendations.append(f"Consider trend-following strategies for {symbol}")
            elif regime == 'ranging':
                recommendations.append(f"Consider mean-reversion strategies for {symbol}")

        # Risk recommendations
        risk_level = analysis['risk_assessment'].get('risk_level')
        if risk_level == 'high':
            recommendations.append("High risk detected - consider reducing overall exposure")

        # Performance recommendations
        if 'performance_metrics' in analysis:
            metrics = analysis['performance_metrics']
            if metrics.get('win_rate', 0) < 0.4:
                recommendations.append("Low win rate - review entry criteria")
            if metrics.get('profit_factor', 0) < 1.5:
                recommendations.append("Low profit factor - improve risk/reward ratio")

        return recommendations

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')

    # Sample market data
    market_data = {
        'BTC-USD': pd.DataFrame({
            'close': np.random.randn(len(dates)).cumsum() + 50000,
            'volume': np.random.randint(1000000, 5000000, len(dates)),
            'high': np.random.randn(len(dates)).cumsum() + 50100,
            'low': np.random.randn(len(dates)).cumsum() + 49900,
        }, index=dates),
        'ETH-USD': pd.DataFrame({
            'close': np.random.randn(len(dates)).cumsum() + 3000,
            'volume': np.random.randint(500000, 2000000, len(dates)),
            'high': np.random.randn(len(dates)).cumsum() + 3010,
            'low': np.random.randn(len(dates)).cumsum() + 2990,
        }, index=dates)
    }

    # Sample trades
    sample_trades = [
        {'entry_price': 50000, 'exit_price': 51000, 'size': 0.1, 'symbol': 'BTC-USD'},
        {'entry_price': 51000, 'exit_price': 50500, 'size': 0.1, 'symbol': 'BTC-USD'},
        {'entry_price': 3000, 'exit_price': 3100, 'size': 1, 'symbol': 'ETH-USD'},
    ]

    # Run analysis
    analyzer = AdvancedAnalytics()
    results = analyzer.analyze(market_data, sample_trades)

    print(json.dumps(results, indent=2, default=str))
