# scripts/cleanup_wyckoff_duplicates.py
#!/usr/bin/env python3
"""
Cleanup script to remove duplicate Wyckoff analyzers
"""

import os
from pathlib import Path

def cleanup_duplicates():
    """Remove duplicate Wyckoff files"""
    
    files_to_remove = [
        "wyckoff_analyzer.py",
        "wyckoff_analyzer copy.py", 
        "utils/wyckoff_analyzer.py",
        "components/wyckoff_analyzer.py"  # Keep this one for now, deprecate later
    ]
    
    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            print(f"Removing duplicate: {file_path}")
            path.unlink()
    
    print("✅ Cleanup completed")

if __name__ == "__main__":
    cleanup_duplicates()