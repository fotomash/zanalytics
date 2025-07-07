#!/usr/bin/env python3
"""
Interactive Command Center Integration Script
Integrates the Strategy Editor into your existing Zanalytics platform
"""

import os
import shutil
import sys
from datetime import datetime

def integrate_command_center():
    """Integrate the Interactive Command Center into Zanalytics"""

    print("=" * 60)
    print("ZANFLOW Interactive Command Center Integration")
    print("=" * 60)
    print()

    # Check if we're in the right directory
    if not os.path.exists("zanalytics_api_service.py"):
        print("❌ Error: Could not find zanalytics_api_service.py")
        print("Please run this script from your Zanalytics project root directory")
        return False

    print("✅ Found existing Zanalytics installation")

    # Backup existing API service
    print("\n📁 Creating backup of existing API service...")
    backup_name = f"zanalytics_api_service.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    shutil.copy("zanalytics_api_service.py", backup_name)
    print(f"✅ Backup created: {backup_name}")

    # Merge API extensions
    print("\n🔧 Merging API extensions...")
    with open("api_extensions/strategy_management_api.py", "r") as f:
        new_api_code = f.read()

    # Find the insertion point in existing API
    with open("zanalytics_api_service.py", "r") as f:
        existing_code = f.read()

    # Insert strategy management endpoints before the main block
    if "if __name__ == '__main__':" in existing_code:
        parts = existing_code.split("if __name__ == '__main__':")

        # Extract just the strategy management endpoints
        endpoint_marker = "# ============= Strategy Management Endpoints ============="
        endpoint_end = "# ============= Additional Utility Endpoints ============="

        start_idx = new_api_code.find(endpoint_marker)
        end_idx = new_api_code.find("if __name__ == '__main__':")

        if start_idx != -1 and end_idx != -1:
            strategy_endpoints = new_api_code[start_idx:end_idx]

            # Merge the code
            merged_code = parts[0] + "\n" + strategy_endpoints + "\nif __name__ == '__main__':" + parts[1]

            with open("zanalytics_api_service.py", "w") as f:
                f.write(merged_code)

            print("✅ API extensions merged successfully")
        else:
            print("⚠️  Warning: Could not find endpoint markers. Manual merge required.")

    # Copy dashboard page
    print("\n📊 Installing Strategy Editor dashboard page...")
    dashboard_dir = "dashboards/pages"
    os.makedirs(dashboard_dir, exist_ok=True)

    shutil.copy(
        "dashboard_pages/strategy_editor.py",
        os.path.join(dashboard_dir, "🔧_Strategy_Editor.py")
    )
    print("✅ Dashboard page installed")

    # Copy validation module
    print("\n✔️  Installing validation module...")
    validation_dir = "core/validation"
    os.makedirs(validation_dir, exist_ok=True)

    shutil.copy(
        "validation/strategy_validator.py",
        os.path.join(validation_dir, "strategy_validator.py")
    )

    # Create __init__.py
    with open(os.path.join(validation_dir, "__init__.py"), "w") as f:
        f.write('from .strategy_validator import StrategyValidator, validate_yaml_syntax, suggest_fixes\n')

    print("✅ Validation module installed")

    # Copy component manager
    print("\n🧩 Installing component manager...")
    component_dir = "core/components"
    os.makedirs(component_dir, exist_ok=True)

    shutil.copy(
        "strategy_components/component_manager.py",
        os.path.join(component_dir, "strategy_component_manager.py")
    )

    # Create __init__.py
    with open(os.path.join(component_dir, "__init__.py"), "w") as f:
        f.write('from .strategy_component_manager import StrategyComponentManager\n')

    print("✅ Component manager installed")

    # Create necessary directories
    print("\n📁 Creating required directories...")
    dirs_to_create = [
        "knowledge/strategies/backups",
        "commands/queue"
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

    # Update requirements.txt
    print("\n📦 Updating requirements...")
    new_requirements = [
        "streamlit>=1.28.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0"
    ]

    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            existing_reqs = f.read()

        for req in new_requirements:
            if req.split('>=')[0] not in existing_reqs:
                existing_reqs += f"\n{req}"

        with open("requirements.txt", "w") as f:
            f.write(existing_reqs.strip() + "\n")
    else:
        with open("requirements.txt", "w") as f:
            f.write("\n".join(new_requirements) + "\n")

    print("✅ Requirements updated")

    # Create example strategy if none exist
    print("\n📝 Creating example strategy...")
    example_strategy = """strategy_name: Example London Breakout
description: Example strategy using the Interactive Command Center
status: testing
timeframes:
  - H1
  - M30
entry_conditions:
  primary:
    - london_session_breakout
    - trend_confirmation
  confirmations:
    - volume_spike
    - momentum_positive
exit_conditions:
  take_profit:
    type: atr
    value: 2.5
  stop_loss:
    type: atr
    value: 1.0
risk_management:
  position_size: 0.02
  max_positions: 2
  max_daily_loss: 5.0
  max_drawdown: 10.0
parameters:
  london_start_hour: 8
  london_end_hour: 9
  breakout_threshold: 10
  atr_period: 14
  volume_multiplier: 1.5
"""

    example_path = "knowledge/strategies/example_london_breakout.yml"
    if not os.path.exists(example_path):
        with open(example_path, "w") as f:
            f.write(example_strategy)
        print("✅ Example strategy created")

    print("\n" + "=" * 60)
    print("✅ Interactive Command Center Integration Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print("1. Restart your API service: python zanalytics_api_service.py")
    print("2. Launch the dashboard: streamlit run dashboards/Home.py")
    print("3. Navigate to the '🔧 Strategy Editor' page")
    print("4. Start editing your strategies interactively!")
    print()
    print("📚 Features Added:")
    print("- Interactive strategy configuration editor")
    print("- Real-time validation with helpful error messages")
    print("- Automatic backup system for all changes")
    print("- Visual and YAML editing modes")
    print("- Strategy templates for quick setup")
    print("- Live reload capabilities")
    print()

    return True

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = integrate_command_center()
    sys.exit(0 if success else 1)
