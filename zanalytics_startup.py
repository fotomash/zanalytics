#!/usr/bin/env python3
"""
ZAnalytics Startup Script
Ensures all components are properly configured before starting
"""

import os
import sys
import json
import yaml
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'zanalytics_data_pipeline.py',
        'zanalytics_integration.py',
        'zanalytics_signal_generator.py',
        'zanalytics_llm_formatter.py',
        '🏠 Home.py',
        'zanalytics_backtester.py',
        'zanalytics_advanced_analytics.py',
        'zanalytics_market_monitor.py',
        'zanalytics_llm_framework.py'
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    return True

def create_default_configs():
    """Create default configuration files"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Orchestrator config
    orchestrator_config = {
        "update_interval": 300,
        "max_workers": 4,
        "symbols": ["BTC/USD", "ETH/USD"],
        "timeframes": ["1h", "4h", "1d"],
        "data_source": "yahoo",
        "enable_backtesting": True,
        "enable_live_monitoring": True,
        "enable_llm_analysis": True
    }

    with open(config_dir / "orchestrator_config.yaml", 'w') as f:
        yaml.safe_dump(orchestrator_config, f)

    print("Created default configuration files")

def main():
    """Main startup routine"""
    print("ZAnalytics Startup Check")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        print("\nPlease ensure all required components are present.")
        sys.exit(1)

    # Create configs if needed
    if not Path("config/orchestrator_config.yaml").exists():
        create_default_configs()

    print("\nAll checks passed! You can now run:")
    print("  python -m core.orchestrator")
    print("\nFor the dashboard, run in a separate terminal:")
    print("  streamlit run 🏠 Home.py")
    print("\nTo run specific components:")
    print("  python zanalytics_market_monitor.py  # For real-time monitoring")
    print("  python zanalytics_backtester.py      # For backtesting")


if __name__ == "__main__":
    main()
