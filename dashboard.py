
# dashboard.py - Real-time monitoring dashboard for ZANALYTICS Data Awareness
import asyncio
import json
import os
from datetime import datetime, timedelta
import time
from pathlib import Path

class ZAnalyticsDashboard:
    """Real-time dashboard for monitoring data flow"""

    def __init__(self):
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.stats = {
            'files_processed': 0,
            'events_total': 0,
            'symbols_active': set(),
            'last_events': [],
            'errors': 0
        }

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def format_uptime(self) -> str:
        """Format system uptime"""
        uptime = datetime.now() - self.start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        seconds = int(uptime.total_seconds() % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def read_log_file(self) -> list:
        """Read recent log entries"""
        log_file = 'zanalytics_awareness.log'
        if not os.path.exists(log_file):
            return []

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return lines[-10:]  # Last 10 lines
        except:
            return []

    def parse_log_stats(self, log_lines: list):
        """Parse statistics from log lines"""
        for line in log_lines:
            if 'Data Event:' in line:
                self.stats['events_total'] += 1
                # Extract symbol from line
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'Data Event:' in part and i + 1 < len(parts):
                        symbol = parts[i + 1]
                        self.stats['symbols_active'].add(symbol)
                        break
            elif 'ERROR' in line:
                self.stats['errors'] += 1
            elif 'Processed' in line:
                self.stats['files_processed'] += 1

    def render_dashboard(self):
        """Render the dashboard"""
        self.clear_screen()

        # Read recent logs
        log_lines = self.read_log_file()
        self.parse_log_stats(log_lines)

        # Header
        print("╔" + "═" * 78 + "╗")
        print("║" + " ZANALYTICS DATA AWARENESS - LIVE DASHBOARD ".center(78) + "║")
        print("╠" + "═" * 78 + "╣")

        # System Status
        status_color = "🟢" if self.stats['errors'] == 0 else "🔴"
        print(f"║ {status_color} SYSTEM STATUS: {'RUNNING' if self.stats['errors'] == 0 else 'ERRORS DETECTED':<20} Uptime: {self.format_uptime():<20} ║")
        print(f"║ 📊 Events Processed: {self.stats['events_total']:<15} Files: {self.stats['files_processed']:<15} Errors: {self.stats['errors']:<10} ║")
        print(f"║ 💱 Active Symbols: {len(self.stats['symbols_active']):<17} Symbols: {', '.join(list(self.stats['symbols_active'])[:5]):<25} ║")

        print("╠" + "═" * 78 + "╣")

        # Recent Activity
        print("║ 📋 RECENT ACTIVITY:" + " " * 57 + "║")
        print("║" + " " * 78 + "║")

        for line in log_lines[-5:]:  # Show last 5 log lines
            timestamp = datetime.now().strftime("%H:%M:%S")
            if len(line.strip()) > 0:
                display_line = line.strip()[:70]  # Truncate long lines
                print(f"║ {timestamp} {display_line:<65} ║")

        # Fill empty lines if needed
        for _ in range(5 - len(log_lines[-5:])):
            print("║" + " " * 78 + "║")

        print("╠" + "═" * 78 + "╣")

        # Directory Monitoring
        print("║ 📁 MONITORED DIRECTORIES:" + " " * 51 + "║")
        try:
            if os.path.exists('zanalytics_config.json'):
                with open('zanalytics_config.json', 'r') as f:
                    config = json.load(f)
                    directories = config.get('watch_directories', [])
                    for i, directory in enumerate(directories[:3]):  # Show first 3
                        exists = "✅" if os.path.exists(directory) else "❌"
                        print(f"║ {exists} {directory:<72} ║")
        except:
            print("║ ❌ Could not load configuration" + " " * 44 + "║")

        print("╠" + "═" * 78 + "╣")

        # Instructions
        print("║ 🎮 CONTROLS: Press Ctrl+C to stop monitoring" + " " * 31 + "║")
        print("║ 📝 LOGS: Check 'zanalytics_awareness.log' for detailed information" + " " * 8 + "║")
        print("║ ⚡ REFRESH: Dashboard updates every 2 seconds" + " " * 32 + "║")

        print("╚" + "═" * 78 + "╝")

        # Last update timestamp
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    async def run(self):
        """Run the dashboard"""
        print("🚀 Starting ZANALYTICS Dashboard...")
        print("   Press Ctrl+C to stop")
        time.sleep(2)

        try:
            while True:
                self.render_dashboard()
                await asyncio.sleep(2)  # Update every 2 seconds
        except KeyboardInterrupt:
            self.clear_screen()
            print("\n🛑 Dashboard stopped")

if __name__ == "__main__":
    dashboard = ZAnalyticsDashboard()
    asyncio.run(dashboard.run())
