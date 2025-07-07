#!/usr/bin/env python3
"""
Setup script to configure paths for ZANFLOW Bridge
"""

import os
import sys

def find_zanalytics():
    """Find the zanalytics directory"""
    print("🔍 Looking for zanalytics directory...")

    # Common locations to check
    locations = [
        os.getcwd(),
        os.path.join(os.getcwd(), 'zanalytics'),
        os.path.join(os.getcwd(), 'zanalytics-main'),
        '/Users/tom/Documents/zanalytics',
        os.path.expanduser('~/Documents/zanalytics'),
        os.path.expanduser('~/zanalytics')
    ]

    found_paths = []

    for loc in locations:
        if os.path.exists(loc):
            # Check for core folder
            core_path = os.path.join(loc, 'core')
            if os.path.exists(core_path):
                found_paths.append(loc)
                print(f"✅ Found zanalytics at: {loc}")

            # Check in subdirectories
            if os.path.isdir(loc):
                for subdir in os.listdir(loc):
                    sub_path = os.path.join(loc, subdir)
                    core_sub = os.path.join(sub_path, 'core')
                    if os.path.exists(core_sub):
                        found_paths.append(sub_path)
                        print(f"✅ Found zanalytics at: {sub_path}")

    if found_paths:
        print(f"\n📂 Found {len(found_paths)} possible zanalytics locations")
        return found_paths[0]  # Return the first valid path
    else:
        print("❌ Could not find zanalytics directory")
        print("\nPlease specify the path to your zanalytics folder:")
        print("It should contain a 'core' subfolder")
        return None

def test_imports(zanalytics_path):
    """Test if imports work"""
    print(f"\n🧪 Testing imports from: {zanalytics_path}")

    sys.path.insert(0, zanalytics_path)

    imports_status = {}

    # Test each import
    test_imports = [
        ('SMC Engine', 'core.analysis.smc_enrichment_engine', 'tag_smc_zones'),
        ('Wyckoff Engine', 'core.wyckoff_phase_engine', 'tag_wyckoff_phases'),
        ('Main Orchestrator', 'core.orchestrators.main_orchestrator', 'MainOrchestrator'),
        ('Advanced SMC', 'core.strategies.advanced_smc', 'run_advanced_smc_strategy')
    ]

    for name, module, func in test_imports:
        try:
            exec(f"from {module} import {func}")
            imports_status[name] = "✅ Success"
        except ImportError as e:
            imports_status[name] = f"❌ Failed: {e}"

    print("\n📊 Import Test Results:")
    for name, status in imports_status.items():
        print(f"  {name}: {status}")

    return all("✅" in status for status in imports_status.values())

def main():
    """Main setup function"""
    print("🚀 ZANFLOW Bridge Setup")
    print("="*50)

    # Find zanalytics
    zanalytics_path = find_zanalytics()

    if not zanalytics_path:
        print("\n💡 Please extract your zanalytics.zip file and run this setup again")
        print("Or manually set the path in zanflow_bridge_fixed.py")
        return

    # Test imports
    success = test_imports(zanalytics_path)

    if success:
        print("\n✅ All imports successful!")
        print(f"\n📝 Add this to your environment or scripts:")
        print(f"export PYTHONPATH='{zanalytics_path}:$PYTHONPATH'")
        print(f"\nOr in Python:")
        print(f"sys.path.insert(0, '{zanalytics_path}')")
    else:
        print("\n⚠️ Some imports failed")
        print("You can still use the simplified analysis mode")

    print("\n🎯 Next steps:")
    print("1. Run: python api_server_smc_wyckoff.py")
    print("2. Run: python zanflow_bridge_fixed.py")
    print("3. Or use: python zanflow_bridge_simple.py (no imports needed)")

if __name__ == "__main__":
    main()
