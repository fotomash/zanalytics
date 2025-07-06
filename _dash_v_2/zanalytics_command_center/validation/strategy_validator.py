"""
Strategy Configuration Validation Module
Ensures strategy configurations are valid before deployment
"""

from typing import Dict, Any, List, Tuple
import yaml
import re
from datetime import datetime

class StrategyValidator:
    """Comprehensive validator for strategy configurations"""

    def __init__(self):
        self.valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']
        self.valid_statuses = ['active', 'inactive', 'testing', 'deprecated']
        self.valid_order_types = ['market', 'limit', 'stop', 'stop_limit']
        self.valid_indicators = [
            'sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stochastic',
            'ichimoku', 'volume', 'pivot', 'fibonacci', 'support_resistance'
        ]

    def validate_strategy(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validates a complete strategy configuration
        Returns: (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Validate required fields
        errors.extend(self._validate_required_fields(config))

        # Validate data types
        type_errors = self._validate_data_types(config)
        errors.extend(type_errors)

        # Skip further validation if type errors exist
        if not type_errors:
            # Validate specific fields
            errors.extend(self._validate_strategy_name(config.get('strategy_name', '')))
            errors.extend(self._validate_timeframes(config.get('timeframes', [])))
            errors.extend(self._validate_entry_conditions(config.get('entry_conditions', {})))
            errors.extend(self._validate_exit_conditions(config.get('exit_conditions', {})))
            errors.extend(self._validate_risk_management(config.get('risk_management', {})))
            errors.extend(self._validate_parameters(config.get('parameters', {})))

            # Generate warnings
            warnings.extend(self._generate_warnings(config))

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def _validate_required_fields(self, config: Dict[str, Any]) -> List[str]:
        """Check for required fields"""
        errors = []
        required_fields = [
            'strategy_name',
            'timeframes',
            'entry_conditions',
            'exit_conditions',
            'risk_management'
        ]

        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: '{field}'")

        return errors

    def _validate_data_types(self, config: Dict[str, Any]) -> List[str]:
        """Validate data types of fields"""
        errors = []
        type_map = {
            'strategy_name': str,
            'description': str,
            'status': str,
            'timeframes': list,
            'entry_conditions': dict,
            'exit_conditions': dict,
            'risk_management': dict,
            'parameters': dict
        }

        for field, expected_type in type_map.items():
            if field in config and not isinstance(config[field], expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}")

        return errors

    def _validate_strategy_name(self, name: str) -> List[str]:
        """Validate strategy name"""
        errors = []

        if not name:
            errors.append("Strategy name cannot be empty")
        elif len(name) < 3:
            errors.append("Strategy name must be at least 3 characters long")
        elif len(name) > 50:
            errors.append("Strategy name must not exceed 50 characters")
        elif not re.match(r'^[a-zA-Z0-9\s\-_]+$', name):
            errors.append("Strategy name contains invalid characters")

        return errors

    def _validate_timeframes(self, timeframes: List[str]) -> List[str]:
        """Validate timeframes"""
        errors = []

        if not timeframes:
            errors.append("At least one timeframe must be specified")
        else:
            for tf in timeframes:
                if tf not in self.valid_timeframes:
                    errors.append(f"Invalid timeframe: '{tf}'. Valid options: {', '.join(self.valid_timeframes)}")

        return errors

    def _validate_entry_conditions(self, conditions: Dict[str, Any]) -> List[str]:
        """Validate entry conditions"""
        errors = []

        if not conditions:
            errors.append("Entry conditions cannot be empty")
            return errors

        # Validate primary conditions
        if 'primary' not in conditions:
            errors.append("Entry conditions must include 'primary' conditions")
        elif not isinstance(conditions['primary'], list):
            errors.append("Primary conditions must be a list")
        elif not conditions['primary']:
            errors.append("At least one primary condition must be specified")

        # Validate confirmations (optional)
        if 'confirmations' in conditions:
            if not isinstance(conditions['confirmations'], list):
                errors.append("Confirmations must be a list")

        return errors

    def _validate_exit_conditions(self, conditions: Dict[str, Any]) -> List[str]:
        """Validate exit conditions"""
        errors = []

        if not conditions:
            errors.append("Exit conditions cannot be empty")
            return errors

        # Validate take profit
        if 'take_profit' not in conditions:
            errors.append("Exit conditions must include 'take_profit'")
        else:
            tp = conditions['take_profit']
            if not isinstance(tp, dict):
                errors.append("Take profit must be a dictionary")
            else:
                if 'type' not in tp:
                    errors.append("Take profit must specify a 'type'")
                elif tp['type'] not in ['fixed', 'atr', 'percentage', 'trailing']:
                    errors.append(f"Invalid take profit type: '{tp['type']}'")

                if 'value' not in tp:
                    errors.append("Take profit must specify a 'value'")
                elif not isinstance(tp['value'], (int, float)) or tp['value'] <= 0:
                    errors.append("Take profit value must be a positive number")

        # Validate stop loss
        if 'stop_loss' not in conditions:
            errors.append("Exit conditions must include 'stop_loss'")
        else:
            sl = conditions['stop_loss']
            if not isinstance(sl, dict):
                errors.append("Stop loss must be a dictionary")
            else:
                if 'type' not in sl:
                    errors.append("Stop loss must specify a 'type'")
                elif sl['type'] not in ['fixed', 'atr', 'percentage', 'trailing']:
                    errors.append(f"Invalid stop loss type: '{sl['type']}'")

                if 'value' not in sl:
                    errors.append("Stop loss must specify a 'value'")
                elif not isinstance(sl['value'], (int, float)) or sl['value'] <= 0:
                    errors.append("Stop loss value must be a positive number")

        return errors

    def _validate_risk_management(self, risk_mgmt: Dict[str, Any]) -> List[str]:
        """Validate risk management settings"""
        errors = []

        if not risk_mgmt:
            errors.append("Risk management settings cannot be empty")
            return errors

        # Validate position size
        if 'position_size' not in risk_mgmt:
            errors.append("Risk management must include 'position_size'")
        else:
            pos_size = risk_mgmt['position_size']
            if not isinstance(pos_size, (int, float)):
                errors.append("Position size must be a number")
            elif pos_size <= 0 or pos_size > 1:
                errors.append("Position size must be between 0 and 1")

        # Validate max positions
        if 'max_positions' not in risk_mgmt:
            errors.append("Risk management must include 'max_positions'")
        else:
            max_pos = risk_mgmt['max_positions']
            if not isinstance(max_pos, int):
                errors.append("Max positions must be an integer")
            elif max_pos < 1 or max_pos > 100:
                errors.append("Max positions must be between 1 and 100")

        # Validate optional fields
        if 'max_daily_loss' in risk_mgmt:
            mdl = risk_mgmt['max_daily_loss']
            if not isinstance(mdl, (int, float)):
                errors.append("Max daily loss must be a number")
            elif mdl <= 0 or mdl > 100:
                errors.append("Max daily loss must be between 0 and 100 (percentage)")

        if 'max_drawdown' in risk_mgmt:
            mdd = risk_mgmt['max_drawdown']
            if not isinstance(mdd, (int, float)):
                errors.append("Max drawdown must be a number")
            elif mdd <= 0 or mdd > 100:
                errors.append("Max drawdown must be between 0 and 100 (percentage)")

        return errors

    def _validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate strategy parameters"""
        errors = []

        for param_name, param_value in parameters.items():
            # Validate parameter name
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', param_name):
                errors.append(f"Invalid parameter name: '{param_name}'. Must start with letter and contain only letters, numbers, and underscores")

            # Validate parameter value
            if isinstance(param_value, dict):
                # Range parameter
                if 'min' in param_value and 'max' in param_value:
                    if not isinstance(param_value['min'], (int, float)) or not isinstance(param_value['max'], (int, float)):
                        errors.append(f"Parameter '{param_name}': min and max must be numbers")
                    elif param_value['min'] > param_value['max']:
                        errors.append(f"Parameter '{param_name}': min value cannot be greater than max value")

                if 'default' in param_value:
                    if not isinstance(param_value['default'], (int, float, str, bool)):
                        errors.append(f"Parameter '{param_name}': default value must be a simple type")

            elif not isinstance(param_value, (int, float, str, bool, list)):
                errors.append(f"Parameter '{param_name}': value must be a simple type or dictionary")

        return errors

    def _generate_warnings(self, config: Dict[str, Any]) -> List[str]:
        """Generate warnings for potential issues"""
        warnings = []

        # Check for missing optional but recommended fields
        if 'description' not in config or not config['description']:
            warnings.append("No description provided for the strategy")

        if 'status' in config and config['status'] not in self.valid_statuses:
            warnings.append(f"Unknown status: '{config['status']}'. Valid options: {', '.join(self.valid_statuses)}")

        # Check for potentially risky settings
        risk_mgmt = config.get('risk_management', {})
        if risk_mgmt.get('position_size', 0) > 0.1:
            warnings.append("Position size > 10% may be risky")

        if risk_mgmt.get('max_positions', 1) > 5:
            warnings.append("Having more than 5 concurrent positions may increase risk")

        # Check exit conditions
        exit_cond = config.get('exit_conditions', {})
        if exit_cond.get('take_profit', {}).get('value', 0) < exit_cond.get('stop_loss', {}).get('value', 1):
            warnings.append("Take profit is smaller than stop loss - unfavorable risk/reward ratio")

        # Check for testing status with aggressive settings
        if config.get('status') == 'testing' and risk_mgmt.get('position_size', 0) > 0.05:
            warnings.append("Testing strategy has position size > 5% - consider reducing for testing")

        return warnings

    def validate_field(self, field_name: str, field_value: Any) -> Tuple[bool, List[str]]:
        """Validate a single field"""
        errors = []

        if field_name == 'strategy_name':
            errors = self._validate_strategy_name(field_value)
        elif field_name == 'timeframes':
            errors = self._validate_timeframes(field_value)
        elif field_name == 'entry_conditions':
            errors = self._validate_entry_conditions(field_value)
        elif field_name == 'exit_conditions':
            errors = self._validate_exit_conditions(field_value)
        elif field_name == 'risk_management':
            errors = self._validate_risk_management(field_value)
        elif field_name == 'parameters':
            errors = self._validate_parameters(field_value)

        return len(errors) == 0, errors


# Additional validation utilities
def validate_yaml_syntax(yaml_content: str) -> Tuple[bool, str]:
    """Validate YAML syntax"""
    try:
        yaml.safe_load(yaml_content)
        return True, "Valid YAML syntax"
    except yaml.YAMLError as e:
        return False, str(e)


def suggest_fixes(errors: List[str]) -> List[str]:
    """Suggest fixes for common errors"""
    suggestions = []

    for error in errors:
        if "Missing required field" in error:
            field = error.split("'")[1]
            suggestions.append(f"Add '{field}' to your configuration")

        elif "Invalid timeframe" in error:
            suggestions.append("Use one of: M1, M5, M15, M30, H1, H4, D1, W1")

        elif "must be a positive number" in error:
            suggestions.append("Ensure all numeric values are greater than 0")

        elif "Position size must be between" in error:
            suggestions.append("Set position size as a decimal between 0.01 and 1.0")

    return suggestions
