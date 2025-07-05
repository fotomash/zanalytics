
#!/usr/bin/env python3
# start_zanalytics_awareness.py - One-click startup for ZANALYTICS Data Awareness

import asyncio
import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from zanalytics_adapter import ZAnalyticsDataBridge
    from data_flow_manager import DataFlowManager
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all required files are in the same directory:")
    print("  - zanalytics_adapter.py")
    print("  - data_flow_manager.py")
    print("  - zanalytics_config.json (project root)")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('zanalytics_awareness.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'pandas', 'watchdog'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════╗
║              ZANALYTICS DATA AWARENESS               ║
║                Real-Time Trading Intel               ║
╠══════════════════════════════════════════════════════╣
║  🚀 Starting Multi-Agent Data Flow System           ║
║  📊 Monitoring: CSV, JSON, TICK files               ║
║  🤖 Agents: Micro, Macro, Risk, Journal, SMC        ║
║  ⚡ Real-time: File watching & event processing     ║
╚══════════════════════════════════════════════════════╝
    """
    print(banner)

def print_status_dashboard():
    """Print current system status"""
    try:
        config_path = Path(__file__).resolve().parent / 'zanalytics_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            print("\n📋 CURRENT CONFIGURATION:")
            print(f"   📁 Watch Directories: {', '.join(config.get('watch_directories', []))}")
            print(f"   🤖 Active Agents: {len([a for a in config.get('agents', {}).values() if a.get('active')])}")
            print(f"   📈 Symbols: {len(set().union(*[a.get('symbols', []) for a in config.get('agents', {}).values()]))}")
            print(f"   ⏱️  Update Interval: {config.get('real_time', {}).get('update_interval', 1.0)}s")
    except Exception as e:
        print(f"   ⚠️  Could not load config: {e}")

async def main():
    """Main startup function"""
    print_banner()

    # Check dependencies
    if not check_dependencies():
        return

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Print current status
    print_status_dashboard()

    print("\n🔧 INITIALIZING SYSTEM...")

    try:
        # Initialize the bridge
        bridge = ZAnalyticsDataBridge()

        print("✅ ZANALYTICS Bridge initialized")
        print("✅ Agents loaded and ready")
        print("✅ Data flow manager configured")

        print("\n🚀 STARTING DATA AWARENESS...")
        print("   Press Ctrl+C to stop")
        print("   Check 'zanalytics_awareness.log' for detailed logs")
        print("\n" + "="*60)

        # Start the data awareness system
        await bridge.start_data_awareness()

    except KeyboardInterrupt:
        print("\n\n🛑 SHUTDOWN INITIATED...")
        print("✅ System stopped gracefully")
        logger.info("ZANALYTICS Data Awareness stopped by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
