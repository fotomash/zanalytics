
# gpt_root_loader.py

import os
from agent_initializer import bootstrap_config
from zanflow_synergy_orchestrator import zanflow_synergy_orchestrator
from zanflow_pipeline import run_all_agents, simulate_setup
from yaml import safe_load

def load_all_yaml_profiles(directory="."):
    yaml_profiles = [f for f in os.listdir(directory) if f.endswith(".yaml")]
    for profile in yaml_profiles:
        print(f"🧩 Loading YAML Profile: {profile}")
        with open(profile, "r") as f:
            data = safe_load(f)
            print(f"📌 Keys: {list(data.keys()) if isinstance(data, dict) else 'Unknown Structure'}")

def gpt_initialize_bundle():
    print("🔧 [GPT BOOT] Initializing ZANFLOW Bundle Environment")
    bootstrap_config(profile="zanflow_multiagent_profile_full.yaml")
    load_all_yaml_profiles()

def gpt_test_dispatch():
    print("🚀 Dispatching sample signal through orchestrator...")
    result = zanflow_synergy_orchestrator(input_data={
        "asset": "GBPUSD",
        "timestamp": "2025-06-01T13:15:00Z",
        "tick_count": 124,
        "spread_status": "stable"
    })
    print("🎯 Sample result summary:", result.get("summary", "N/A"))

def gpt_launch_all_agents():
    print("📡 Launching all registered agents...")
    run_all_agents()

def auto_scan_h1_csvs():
    h1_path = "csv/h1"
    if not os.path.exists(h1_path):
        print("⚠️ No H1 folder found at csv/h1.")
        return

    h1_files = [f for f in os.listdir(h1_path) if f.endswith(".csv")]
    if not h1_files:
        print("⚠️ No H1 .csv files found.")
        return

    print("📊 Detected H1 CSV files:")
    for f in h1_files:
        print(" -", f)

    choice = input("🔍 Do you want to auto-scan all H1 files? [Y/n]: ").strip().lower()
    if choice != "n":
        for file in h1_files:
            symbol = file.replace(".csv", "")
            print(f"📈 Scanning {symbol} (H1)...")
            simulate_setup(
                asset=symbol,
                analysis_htf="H1",
                entry_ltf="M1",
                conviction=3,
                account_balance=100000
            )

if __name__ == "__main__":
    gpt_initialize_bundle()
    gpt_test_dispatch()
    gpt_launch_all_agents()
    auto_scan_h1_csvs()
