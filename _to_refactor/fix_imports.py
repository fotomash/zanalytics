#!/usr/bin/env python3
"""
Fix script to update class names in the unified analyzer module.
"""

import re

# Read the original unified_microstructure_analyzer.py
with open('unified_microstructure_analyzer.py', 'r') as f:
    content = f.read()

# Fix the import statement
content = content.replace(
    'from advanced_analysis import SMCAnalyzer, WyckoffAnalyzer, OrderFlowAnalyzer, TickAnalyzer',
    '# Advanced analyzers are defined in this file'
)

# Remove the problematic import line
lines = content.split('\n')
fixed_lines = []
for line in lines:
    if 'from advanced_analysis import' not in line:
        fixed_lines.append(line)

content = '\n'.join(fixed_lines)

# Write the fixed version
with open('unified_microstructure_analyzer.py', 'w') as f:
    f.write(content)

print("✅ Fixed unified_microstructure_analyzer.py")

# Also update the CLI to handle missing imports gracefully
cli_fix = """
# Add this at the top of analyzer_cli.py after the imports
try:
    from unified_microstructure_analyzer import (
        AnalyzerConfig, UnifiedAnalyzer, load_tick_data, load_bar_data,
        save_to_parquet, load_config
    )
except ImportError as e:
    print(f"Error importing analyzer modules: {e}")
    print("Please ensure unified_microstructure_analyzer.py is in the same directory")
    sys.exit(1)
"""

print("\n📝 Add this import check to analyzer_cli.py after the initial imports")
print(cli_fix)
