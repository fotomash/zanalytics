#!/usr/bin/env python3
"""
Trading System Setup Script
Organizes all files into proper architecture and prepares the system
"""

import os
import shutil
from pathlib import Path
import json

def create_folder_structure():
    """Create the recommended folder structure"""
    folders = [
        'data/raw',
        'data/processed',
        'data/processed/analysis_reports',
        'data/cache',
        'data_pipeline',
        'zanalytics/agents',
        'api_server', 
        'dashboard/components',
        'trading_engine',
        'config',
        'logs'
    ]

    print("Creating folder structure...")
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")

def organize_existing_files():
    """Organize existing files into proper folders"""
    print("\nOrganizing existing files...")

    # File organization map
    file_moves = {
        'data_pipeline/': [
            'data_enricher.py',
            'market_structure_analyzer_smc.py',
            'poi_manager_smc.py',
            'entry_executor_smc.py',
            'confirmation_engine_smc.py',
            'liquidity_engine_smc.py',
            'wyckoff_phase_engine.py',
            'fibonacci_filter.py',
            'volatility_engine.py',
            'indicators.py'
        ],
        'zanalytics/': [
            'zanalytics_adapter.py',
            'agent_registry.py'
        ],
        'zanalytics/agents/': [
            'agent_htfanalyst.py',
            'agent_microstrategist.py',
            'agent_riskmanager.py'
        ],
        'api_server/': [
            'websocket_server.py',
            'rest_endpoints.py'
        ],
        'dashboard/': [
            'dashboard.py'
        ],
        'trading_engine/': [
            'state_machine.py'
        ]
    }

    # Move files
    for target_dir, file_list in file_moves.items():
        for filename in file_list:
            if os.path.exists(filename):
                target_path = Path(target_dir) / filename
                try:
                    if not target_path.exists():
                        shutil.copy2(filename, target_path)
                        print(f"Copied: {filename} -> {target_path}")
                except Exception as e:
                    print(f"Could not copy {filename}: {e}")

def create_configuration_files():
    """Create default configuration files"""
    print("\nCreating configuration files...")

    settings = {
        "data_pipeline": {
            "refresh_interval": 60,
            "data_path": "./data",
            "timeframes": ["1M", "5M", "15M", "1H", "4H"]
        },
        "api_server": {
            "websocket_port": 8765,
            "rest_port": 8080,
            "host": "localhost"
        },
        "dashboard": {
            "port": 8050,
            "debug": True
        },
        "trading": {
            "paper_trading": True,
            "risk_per_trade": 0.02
        }
    }

    with open('config/settings.json', 'w') as f:
        json.dump(settings, f, indent=2)
    print("Created: config/settings.json")

def create_requirements_file():
    """Create requirements.txt"""
    print("\nCreating requirements.txt...")

    requirements = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.5.0",
        "plotly>=5.11.0",
        "dash>=2.7.0",
        "websockets>=10.4",
        "aiohttp>=3.8.0"
    ]

    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    print("Created: requirements.txt")

def create_startup_scripts():
    """Create startup scripts"""
    print("\nCreating startup scripts...")

    quick_analysis = """#!/usr/bin/env python3
import sys
sys.path.append('data_pipeline')

try:
    from data_enricher import DataEnricher

    config = {
        'data_path': './data',
        'refresh_interval': 60
    }

    print("Running analysis...")
    enricher = DataEnricher(config)
    enricher.process_all_data()
    print("Complete!")

except Exception as e:
    print(f"Error: {e}")
    print("Run setup_system.py first")
"""

    with open('quick_analysis.py', 'w') as f:
        f.write(quick_analysis)
    print("Created: quick_analysis.py")

def main():
    """Main setup function"""
    print("Ultimate Trading System Setup")
    print("="*50)

    create_folder_structure()
    organize_existing_files()
    create_configuration_files() 
    create_requirements_file()
    create_startup_scripts()

    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    print("Next Steps:")
    print("1. pip install -r requirements.txt")
    print("2. python main.py")
    print("3. Open http://localhost:8050")
    print("="*50)

if __name__ == "__main__":
    main()
