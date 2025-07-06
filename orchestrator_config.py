# ZAnalytics Orchestrator Configuration
import yaml
import os

def load_orchestrator_config():
    config_path = os.path.join(os.path.dirname(__file__), 'orchestrator_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

orchestrator_config = load_orchestrator_config()
