"""
Strategy Component Utilities
Helper functions for strategy configuration management
"""

import yaml
import json
from typing import Dict, Any, List
from datetime import datetime
import os

class StrategyComponentManager:
    """Manages strategy components and templates"""

    def __init__(self, strategies_dir: str = "knowledge/strategies"):
        self.strategies_dir = strategies_dir
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Any]:
        """Load strategy templates"""
        return {
            'default': {
                'strategy_name': 'New Strategy',
                'description': 'A new trading strategy',
                'status': 'testing',
                'timeframes': ['H1'],
                'entry_conditions': {
                    'primary': ['trend_direction'],
                    'confirmations': []
                },
                'exit_conditions': {
                    'take_profit': {'type': 'fixed', 'value': 50},
                    'stop_loss': {'type': 'fixed', 'value': 25}
                },
                'risk_management': {
                    'position_size': 0.01,
                    'max_positions': 1,
                    'max_daily_loss': 5.0,
                    'max_drawdown': 10.0
                },
                'parameters': {}
            },
            'scalping': {
                'strategy_name': 'Scalping Strategy',
                'description': 'Fast-paced scalping strategy for quick profits',
                'status': 'testing',
                'timeframes': ['M1', 'M5'],
                'entry_conditions': {
                    'primary': ['price_action', 'momentum'],
                    'confirmations': ['volume_spike']
                },
                'exit_conditions': {
                    'take_profit': {'type': 'fixed', 'value': 10},
                    'stop_loss': {'type': 'fixed', 'value': 5}
                },
                'risk_management': {
                    'position_size': 0.02,
                    'max_positions': 3,
                    'max_daily_loss': 3.0,
                    'max_drawdown': 5.0
                },
                'parameters': {
                    'fast_ma': 5,
                    'slow_ma': 20,
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30
                }
            },
            'swing': {
                'strategy_name': 'Swing Trading Strategy',
                'description': 'Medium-term strategy for capturing market swings',
                'status': 'testing',
                'timeframes': ['H4', 'D1'],
                'entry_conditions': {
                    'primary': ['trend_reversal', 'support_resistance'],
                    'confirmations': ['divergence', 'volume_confirmation']
                },
                'exit_conditions': {
                    'take_profit': {'type': 'atr', 'value': 3},
                    'stop_loss': {'type': 'atr', 'value': 1.5}
                },
                'risk_management': {
                    'position_size': 0.03,
                    'max_positions': 2,
                    'max_daily_loss': 6.0,
                    'max_drawdown': 15.0
                },
                'parameters': {
                    'atr_period': 14,
                    'swing_high_lookback': 20,
                    'swing_low_lookback': 20,
                    'trend_ma': 200
                }
            },
            'breakout': {
                'strategy_name': 'Breakout Strategy',
                'description': 'Captures strong moves from consolidation breakouts',
                'status': 'testing',
                'timeframes': ['M30', 'H1'],
                'entry_conditions': {
                    'primary': ['breakout', 'volatility_expansion'],
                    'confirmations': ['volume_breakout', 'momentum_confirmation']
                },
                'exit_conditions': {
                    'take_profit': {'type': 'percentage', 'value': 2.0},
                    'stop_loss': {'type': 'percentage', 'value': 1.0}
                },
                'risk_management': {
                    'position_size': 0.025,
                    'max_positions': 2,
                    'max_daily_loss': 4.0,
                    'max_drawdown': 8.0
                },
                'parameters': {
                    'consolidation_periods': 20,
                    'breakout_threshold': 1.5,
                    'volume_multiplier': 2.0,
                    'atr_period': 14
                }
            }
        }

    def create_strategy_from_template(self, template_name: str, custom_name: str = None) -> Dict[str, Any]:
        """Create a new strategy from a template"""
        if template_name not in self.templates:
            template_name = 'default'

        strategy = self.templates[template_name].copy()

        if custom_name:
            strategy['strategy_name'] = custom_name

        strategy['created_at'] = datetime.now().isoformat()
        strategy['version'] = '1.0'

        return strategy

    def merge_configurations(self, base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two strategy configurations"""
        merged = base_config.copy()

        for key, value in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                merged[key] = self.merge_configurations(merged[key], value)
            else:
                merged[key] = value

        return merged

    def extract_indicators(self, config: Dict[str, Any]) -> List[str]:
        """Extract all indicators used in a strategy"""
        indicators = set()

        # Check entry conditions
        entry = config.get('entry_conditions', {})
        indicators.update(entry.get('primary', []))
        indicators.update(entry.get('confirmations', []))

        # Check parameters for indicator settings
        params = config.get('parameters', {})
        for param_name in params:
            # Common indicator parameter patterns
            if any(ind in param_name.lower() for ind in ['ma', 'ema', 'sma', 'rsi', 'macd', 'atr', 'bb']):
                if 'ma' in param_name.lower():
                    indicators.add('moving_average')
                elif 'rsi' in param_name.lower():
                    indicators.add('rsi')
                elif 'macd' in param_name.lower():
                    indicators.add('macd')
                elif 'atr' in param_name.lower():
                    indicators.add('atr')
                elif 'bb' in param_name.lower():
                    indicators.add('bollinger_bands')

        return sorted(list(indicators))

    def generate_strategy_summary(self, config: Dict[str, Any]) -> str:
        """Generate a human-readable summary of a strategy"""
        summary = []

        summary.append(f"# {config.get('strategy_name', 'Unnamed Strategy')}")
        summary.append(f"Status: {config.get('status', 'unknown').upper()}")
        summary.append("")

        if 'description' in config:
            summary.append(f"## Description")
            summary.append(config['description'])
            summary.append("")

        summary.append(f"## Trading Configuration")
        summary.append(f"- Timeframes: {', '.join(config.get('timeframes', []))}")

        # Entry conditions
        entry = config.get('entry_conditions', {})
        if entry:
            summary.append(f"- Entry Signals: {', '.join(entry.get('primary', []))}")
            if entry.get('confirmations'):
                summary.append(f"- Confirmations: {', '.join(entry['confirmations'])}")

        # Exit conditions
        exit_cond = config.get('exit_conditions', {})
        if exit_cond:
            tp = exit_cond.get('take_profit', {})
            sl = exit_cond.get('stop_loss', {})
            summary.append(f"- Take Profit: {tp.get('value', 'N/A')} ({tp.get('type', 'N/A')})")
            summary.append(f"- Stop Loss: {sl.get('value', 'N/A')} ({sl.get('type', 'N/A')})")

        # Risk management
        risk = config.get('risk_management', {})
        if risk:
            summary.append("")
            summary.append(f"## Risk Management")
            summary.append(f"- Position Size: {risk.get('position_size', 0) * 100:.1f}%")
            summary.append(f"- Max Positions: {risk.get('max_positions', 1)}")
            if 'max_daily_loss' in risk:
                summary.append(f"- Max Daily Loss: {risk['max_daily_loss']}%")
            if 'max_drawdown' in risk:
                summary.append(f"- Max Drawdown: {risk['max_drawdown']}%")

        # Parameters
        params = config.get('parameters', {})
        if params:
            summary.append("")
            summary.append(f"## Parameters")
            for param, value in params.items():
                summary.append(f"- {param}: {value}")

        return "\n".join(summary)

    def export_to_json(self, config: Dict[str, Any], filepath: str):
        """Export strategy configuration to JSON"""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def import_from_json(self, filepath: str) -> Dict[str, Any]:
        """Import strategy configuration from JSON"""
        with open(filepath, 'r') as f:
            return json.load(f)
