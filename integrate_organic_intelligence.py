#!/usr/bin/env python3
"""Integration script for Organic Intelligence phase."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

PHASE5_DIR = Path("zanalytics_phase5_organic_intelligence")

FILES_TO_COPY = [
    ("core/agents/scheduling_agent.py", "core/agents/scheduling_agent.py"),
    ("core/agents/london_killzone_agent.py", "core/agents/london_killzone_agent.py"),
    ("core/dispatcher/action_dispatcher.py", "core/dispatcher/action_dispatcher.py"),
    ("core/command_processor.py", "core/command_processor.py"),
    ("core/commands/command_schema.json", "core/commands/command_schema.json"),
    ("knowledge/strategies/London_Killzone_SMC.yml", "knowledge/strategies/London_Killzone_SMC.yml"),
    ("api/endpoints/organic_intelligence.py", "api/endpoints/organic_intelligence.py"),
]

REQUIREMENTS = ["apscheduler>=3.10.0", "pyyaml>=6.0"]


def copy_components() -> None:
    """Copy Organic Intelligence components into the project."""
    print("\n📁 Copying Organic Intelligence components...")
    for src, dst in FILES_TO_COPY:
        src_path = PHASE5_DIR / src
        dst_path = Path(dst)
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"  ✅ {src} -> {dst}")
        else:
            print(f"  ⚠️ Missing {src}")


def update_requirements() -> None:
    """Append required packages to requirements.txt if missing."""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        return

    existing = req_file.read_text().splitlines()
    with req_file.open("a") as fh:
        for req in REQUIREMENTS:
            pkg = req.split(">=")[0]
            if not any(line.startswith(pkg) for line in existing):
                fh.write(f"\n{req}")
                print(f"  ✅ Added requirement {req}")


def create_config() -> None:
    """Create example configuration file."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    config = {
        "organic_intelligence": {
            "enabled": True,
            "command_processor": {
                "max_concurrent_commands": 10,
                "command_timeout": 300,
            },
            "scheduling": {
                "timezone": "UTC",
                "max_concurrent_agents": 5,
            },
            "agents": {
                "london_killzone": {
                    "enabled": True,
                    "max_daily_trades": 3,
                }
            },
        }
    }
    cfg_path = config_dir / "organic_intelligence.json"
    with cfg_path.open("w") as fh:
        json.dump(config, fh, indent=2)
    print(f"  ✅ Created {cfg_path}")


def create_main_snippet() -> None:
    """Write integration instructions for main.py."""
    snippet = """
# Organic Intelligence imports
from api.endpoints import organic_intelligence
from core.command_processor import OrganicIntelligenceOrchestrator

# Add Organic Intelligence router
app.include_router(organic_intelligence.router)

# Initialize Organic Intelligence on startup
@app.on_event(\"startup\")
async def startup_organic_intelligence():
    organic_intelligence.init_organic_intelligence(redis_client, app_config)
    await organic_intelligence.orchestrator.start()

@app.on_event(\"shutdown\")
async def shutdown_organic_intelligence():
    if organic_intelligence.orchestrator:
        await organic_intelligence.orchestrator.stop()
"""
    Path("main_additions.py").write_text(snippet.strip() + "\n")
    print("  ⚠️  Review 'main_additions.py' and merge into your main application")


def create_startup_script() -> None:
    """Create helper script to launch the orchestrator."""
    script = """#!/usr/bin/env python3
'''
Zanalytics with Organic Intelligence - Startup Script
'''

import asyncio
import redis
from core.command_processor import OrganicIntelligenceOrchestrator


async def main():
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    config = {"organic_intelligence": {"enabled": True}}
    orchestrator = OrganicIntelligenceOrchestrator(redis_client, config)
    try:
        await orchestrator.start()
        print('✅ Organic Intelligence is running!')
        print('\nActive Strategies:')
        for strategy_id in orchestrator.scheduling_agent.active_strategies:
            print(f'  - {strategy_id}')
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print('\nShutting down...')
        await orchestrator.stop()

if __name__ == '__main__':
    asyncio.run(main())
"""
    path = Path("run_organic_intelligence.py")
    path.write_text(script)
    os.chmod(path, 0o755)
    print(f"  ✅ Created {path}")


def integrate() -> None:
    print("🧠 Starting Organic Intelligence Integration...")
    copy_components()
    create_main_snippet()
    update_requirements()
    create_config()
    create_startup_script()
    print("\n✨ Integration complete!")
    print("\nNext steps:")
    print("1. Merge 'main_additions.py' into your entrypoint (e.g. run_system.py)")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Ensure Redis is running")
    print("4. Run: python run_organic_intelligence.py")


if __name__ == "__main__":
    integrate()
