"""
Enhanced Zanalytics API Service with Interactive Strategy Management
This extends the existing API with secure endpoints for strategy configuration
"""

import os
import yaml
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for dashboard access

# Configuration
STRATEGIES_DIR = "knowledge/strategies"
BACKUP_DIR = "knowledge/strategies/backups"
COMMAND_QUEUE_DIR = "commands/queue"

# Ensure directories exist
os.makedirs(STRATEGIES_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(COMMAND_QUEUE_DIR, exist_ok=True)

# ============= Strategy Management Endpoints =============

@app.route('/strategies', methods=['GET'])
def list_strategies():
    """Scans the strategies directory and returns a list of all available strategies."""
    try:
        strategies = []
        for filename in os.listdir(STRATEGIES_DIR):
            if filename.endswith((".yml", ".yaml")):
                filepath = os.path.join(STRATEGIES_DIR, filename)
                with open(filepath, 'r') as f:
                    config = yaml.safe_load(f)
                    strategies.append({
                        "id": filename,
                        "name": config.get("strategy_name", "Unnamed Strategy"),
                        "description": config.get("description", "No description"),
                        "status": config.get("status", "active"),
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(filepath)
                        ).isoformat()
                    })

        # Sort by name
        strategies.sort(key=lambda x: x['name'])
        return jsonify(strategies)

    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/strategies/<strategy_id>', methods=['GET'])
def get_strategy_config(strategy_id):
    """Fetches the full configuration for a single strategy."""
    try:
        filepath = os.path.join(STRATEGIES_DIR, strategy_id)
        if not os.path.exists(filepath):
            return jsonify({"error": "Strategy not found"}), 404

        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        # Add metadata
        config['_metadata'] = {
            'id': strategy_id,
            'last_modified': datetime.fromtimestamp(
                os.path.getmtime(filepath)
            ).isoformat(),
            'file_size': os.path.getsize(filepath)
        }

        return jsonify(config)

    except Exception as e:
        logger.error(f"Error getting strategy {strategy_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/strategies/<strategy_id>', methods=['POST'])
def update_strategy_config(strategy_id):
    """
    Receives a JSON payload and safely writes it back to the strategy's YAML file.
    This is the secure write-back mechanism with validation and backup.
    """
    try:
        filepath = os.path.join(STRATEGIES_DIR, strategy_id)
        if not os.path.exists(filepath):
            return jsonify({"error": "Strategy not found"}), 404

        new_config = request.json

        # Remove metadata before saving
        if '_metadata' in new_config:
            del new_config['_metadata']

        # Create backup before modifying
        backup_filename = f"{strategy_id}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        with open(filepath, 'r') as f:
            backup_content = f.read()
        with open(backup_path, 'w') as f:
            f.write(backup_content)
        logger.info(f"Created backup: {backup_filename}")

        # Validate the configuration
        validation_result = validate_strategy_config(new_config)
        if not validation_result['valid']:
            return jsonify({
                "error": "Invalid configuration",
                "details": validation_result['errors']
            }), 400

        # Write the new configuration
        with open(filepath, 'w') as f:
            yaml.dump(new_config, f, indent=2, sort_keys=False)

        logger.info(f"Strategy '{strategy_id}' has been updated.")

        # Create a reload command for the Action Dispatcher
        reload_command = {
            "action": "reload_strategy",
            "payload": {
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat()
            }
        }

        command_filename = f"reload_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        command_path = os.path.join(COMMAND_QUEUE_DIR, command_filename)
        with open(command_path, 'w') as f:
            json.dump(reload_command, f, indent=2)

        return jsonify({
            "status": "success",
            "message": f"{strategy_id} updated successfully",
            "backup": backup_filename,
            "reload_command": command_filename
        })

    except Exception as e:
        logger.error(f"Error updating strategy {strategy_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/strategies/<strategy_id>/validate', methods=['POST'])
def validate_strategy(strategy_id):
    """Validates a strategy configuration without saving it."""
    try:
        config = request.json
        validation_result = validate_strategy_config(config)

        return jsonify({
            "valid": validation_result['valid'],
            "errors": validation_result.get('errors', []),
            "warnings": validation_result.get('warnings', [])
        })

    except Exception as e:
        logger.error(f"Error validating strategy: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/strategies/<strategy_id>/backups', methods=['GET'])
def list_strategy_backups(strategy_id):
    """Lists all available backups for a strategy."""
    try:
        backups = []
        for filename in os.listdir(BACKUP_DIR):
            if filename.startswith(f"{strategy_id}.backup."):
                filepath = os.path.join(BACKUP_DIR, filename)
                timestamp = filename.split('.')[-1]
                backups.append({
                    "filename": filename,
                    "timestamp": timestamp,
                    "size": os.path.getsize(filepath),
                    "created": datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    ).isoformat()
                })

        # Sort by timestamp descending
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(backups)

    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/strategies/<strategy_id>/restore/<backup_filename>', methods=['POST'])
def restore_strategy_backup(strategy_id, backup_filename):
    """Restores a strategy from a backup."""
    try:
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        if not os.path.exists(backup_path):
            return jsonify({"error": "Backup not found"}), 404

        strategy_path = os.path.join(STRATEGIES_DIR, strategy_id)

        # Create a backup of current version before restoring
        current_backup = f"{strategy_id}.before_restore.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        current_backup_path = os.path.join(BACKUP_DIR, current_backup)
        with open(strategy_path, 'r') as f:
            current_content = f.read()
        with open(current_backup_path, 'w') as f:
            f.write(current_content)

        # Restore from backup
        with open(backup_path, 'r') as f:
            backup_content = f.read()
        with open(strategy_path, 'w') as f:
            f.write(backup_content)

        logger.info(f"Restored strategy '{strategy_id}' from backup '{backup_filename}'")

        return jsonify({
            "status": "success",
            "message": f"Strategy restored from {backup_filename}",
            "pre_restore_backup": current_backup
        })

    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        return jsonify({"error": str(e)}), 500

# ============= Validation Functions =============

def validate_strategy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a strategy configuration.
    Returns a dict with 'valid' boolean and 'errors'/'warnings' lists.
    """
    errors = []
    warnings = []

    # Required fields
    required_fields = ['strategy_name', 'timeframes', 'entry_conditions']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate timeframes
    if 'timeframes' in config:
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']
        for tf in config['timeframes']:
            if tf not in valid_timeframes:
                errors.append(f"Invalid timeframe: {tf}")

    # Validate numeric parameters
    if 'parameters' in config:
        for param_name, param_value in config['parameters'].items():
            if isinstance(param_value, dict):
                if 'min' in param_value and 'max' in param_value:
                    if param_value['min'] > param_value['max']:
                        errors.append(f"Parameter {param_name}: min > max")

    # Validate status
    if 'status' in config:
        valid_statuses = ['active', 'inactive', 'testing', 'deprecated']
        if config['status'] not in valid_statuses:
            warnings.append(f"Unknown status: {config['status']}")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

# ============= Additional Utility Endpoints =============

@app.route('/strategies/create', methods=['POST'])
def create_new_strategy():
    """Creates a new strategy from a template."""
    try:
        data = request.json
        strategy_name = data.get('name', 'new_strategy')
        template_id = data.get('template', 'default')

        # Generate filename
        filename = f"{strategy_name.lower().replace(' ', '_')}.yml"
        filepath = os.path.join(STRATEGIES_DIR, filename)

        # Check if already exists
        if os.path.exists(filepath):
            return jsonify({"error": "Strategy already exists"}), 409

        # Create from template
        template = get_strategy_template(template_id)
        template['strategy_name'] = strategy_name
        template['created_at'] = datetime.now().isoformat()

        # Save the new strategy
        with open(filepath, 'w') as f:
            yaml.dump(template, f, indent=2, sort_keys=False)

        logger.info(f"Created new strategy: {filename}")

        return jsonify({
            "status": "success",
            "strategy_id": filename,
            "message": f"Strategy '{strategy_name}' created successfully"
        })

    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        return jsonify({"error": str(e)}), 500

def get_strategy_template(template_id: str) -> Dict[str, Any]:
    """Returns a strategy template."""
    templates = {
        'default': {
            'strategy_name': 'New Strategy',
            'description': 'A new trading strategy',
            'status': 'testing',
            'timeframes': ['H1'],
            'entry_conditions': {
                'primary': [],
                'confirmations': []
            },
            'exit_conditions': {
                'take_profit': {'type': 'fixed', 'value': 50},
                'stop_loss': {'type': 'fixed', 'value': 25}
            },
            'risk_management': {
                'position_size': 0.01,
                'max_positions': 1
            },
            'parameters': {}
        },
        'scalping': {
            'strategy_name': 'Scalping Strategy',
            'description': 'Fast-paced scalping strategy',
            'status': 'testing',
            'timeframes': ['M1', 'M5'],
            'entry_conditions': {
                'primary': ['price_action'],
                'confirmations': ['volume']
            },
            'exit_conditions': {
                'take_profit': {'type': 'fixed', 'value': 10},
                'stop_loss': {'type': 'fixed', 'value': 5}
            },
            'risk_management': {
                'position_size': 0.02,
                'max_positions': 3
            },
            'parameters': {
                'fast_ma': 5,
                'slow_ma': 20
            }
        }
    }

    return templates.get(template_id, templates['default']).copy()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
