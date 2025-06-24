#!/usr/bin/env python3
# start_zanalytics_awareness.py - v2 (Robust Startup)

import asyncio
import sys
import os
from pathlib import Path
import logging
import json

# --- ROBUST PATH FIX ---
# This ensures that no matter where you run it from, it finds the adapter
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path(os.getcwd())
sys.path.insert(0, str(script_dir))
# --- END PATH FIX ---

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
    required_packages = ['pandas', 'watchdog']
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
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

async def main():
    """Main startup function"""
    print_banner()
    
    if not check_dependencies():
        return
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("\n🔧 INITIALIZING SYSTEM...")
    
    try:
        # --- MODIFICATION HERE ---
        # We now import and initialize inside the try block for better error handling
        from zanalytics_adapter import ZAnalyticsDataBridge, ZANALYTICS_FOUND
        
        if not ZANALYTICS_FOUND:
             logger.error("ZANALYTICS core package not found. Please ensure 'zanalytics_main' exists.")
             return

        bridge = ZAnalyticsDataBridge()
        
        print("✅ ZANALYTICS Bridge initialized")
        print("✅ Agents loaded and ready")
        print("✅ Data flow manager configured")
        
        print("\n🚀 STARTING DATA AWARENESS...")
        print("   Press Ctrl+C to stop")
        print("   Check 'zanalytics_awareness.log' for detailed logs")
        print("\n" + "="*60)
        
        await bridge.start_data_awareness()
        
    except KeyboardInterrupt:
        print("\n\n🛑 SHUTDOWN INITIATED...")
        print("✅ System stopped gracefully")
        logger.info("ZANALYTICS Data Awareness stopped by user")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR DURING STARTUP: {e}")
        logger.critical(f"System could not start: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())