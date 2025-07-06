#!/usr/bin/env python3
"""System Launcher"""

import subprocess
import sys

def main():
    print("🚀 Trading System Launcher")
    print("="*30)

    while True:
        print("\n1. Setup system")
        print("2. Quick analysis") 
        print("3. Start system")
        print("4. Exit")

        choice = input("\nChoice (1-4): ").strip()

        if choice == "1":
            subprocess.run([sys.executable, "setup_system.py"])
        elif choice == "2":
            subprocess.run([sys.executable, "quick_analysis.py"])
        elif choice == "3":
            subprocess.run([sys.executable, "main.py"])
        elif choice == "4":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
