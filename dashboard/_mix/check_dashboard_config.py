# Dashboard Configuration Checker
import json
import os
from pathlib import Path

def check_dashboard_config():
    """Check dashboard configuration and data compatibility"""

    print("🔍 Checking Dashboard Configuration...")
    print("=" * 50)

    # Check for required modules
    required_modules = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scipy', 'openpyxl'
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} installed")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} NOT installed")

    if missing_modules:
        print(f"\n⚠️  Install missing modules with:")
        print(f"pip install {' '.join(missing_modules)}")

    # Check data directory
    data_dir = Path("./data")
    if data_dir.exists():
        print(f"\n✅ Data directory exists: {data_dir}")

        # Count files
        csv_files = list(data_dir.rglob("*.csv"))
        json_files = list(data_dir.rglob("*.json"))
        excel_files = list(data_dir.rglob("*.xlsx")) + list(data_dir.rglob("*.xls"))

        print(f"\n📊 Data files found:")
        print(f"   - CSV files: {len(csv_files)}")
        print(f"   - JSON files: {len(json_files)}")
        print(f"   - Excel files: {len(excel_files)}")

        # Check for specific patterns
        tick_files = [f for f in csv_files if 'tick' in f.name.lower()]
        bar_files = [f for f in csv_files if 'bar' in f.name.lower()]
        processed_files = [f for f in csv_files if 'processed' in f.name.lower()]

        print(f"\n📈 Specialized files:")
        print(f"   - Tick data files: {len(tick_files)}")
        print(f"   - Bar data files: {len(bar_files)}")
        print(f"   - Processed files: {len(processed_files)}")

    else:
        print(f"\n❌ Data directory not found: {data_dir}")
        print("   Create it with: mkdir ./data")

    # Check for analyzer defaults
    if os.path.exists("ncOS_ultimate_microstructure_analyzer_DEFAULTS.py"):
        print("\n✅ Analyzer defaults module found")

        # Try to import and check configuration
        try:
            import ncOS_ultimate_microstructure_analyzer_DEFAULTS as defaults

            if hasattr(defaults, 'SMC_FEATURES'):
                print("   - SMC_FEATURES configuration found")
            if hasattr(defaults, 'WYCKOFF_FEATURES'):
                print("   - WYCKOFF_FEATURES configuration found")

        except Exception as e:
            print(f"   ⚠️ Error loading defaults: {e}")
    else:
        print("\n⚠️  Analyzer defaults module not found")

    print("\n✅ Configuration check complete!")

if __name__ == "__main__":
    check_dashboard_config()
