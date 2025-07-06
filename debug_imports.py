#!/usr/bin/env python3
"""Debug script to check package imports and Python environment"""

import sys
import os
import subprocess

print("🔍 Python Environment Debug")
print("=" * 50)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Virtual env: {os.environ.get('VIRTUAL_ENV', 'Not detected')}")
print()

print("📦 Python Path:")
for i, path in enumerate(sys.path):
    print(f"  [{i}] {path}")
print()

print("🧪 Testing imports:")
packages = {
    'yaml': 'pyyaml',
    'sklearn': 'scikit-learn',
    'streamlit': 'streamlit',
    'flask': 'flask',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'plotly': 'plotly',
    'requests': 'requests',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn'
}

for import_name, package_name in packages.items():
    try:
        if import_name == 'yaml':
            import yaml
            print(f"✅ {package_name}: {yaml.__version__ if hasattr(yaml, '__version__') else 'installed'}")
        elif import_name == 'sklearn':
            import sklearn
            print(f"✅ {package_name}: {sklearn.__version__}")
        else:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'installed')
            print(f"✅ {package_name}: {version}")
    except ImportError as e:
        print(f"❌ {package_name}: {str(e)}")
print()

print("📍 Checking pip list:")
try:
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                          capture_output=True, text=True)
    if 'pyyaml' in result.stdout.lower():
        print("✅ pyyaml found in pip list")
    else:
        print("❌ pyyaml NOT found in pip list")

    if 'scikit-learn' in result.stdout.lower():
        print("✅ scikit-learn found in pip list")
    else:
        print("❌ scikit-learn NOT found in pip list")
except Exception as e:
    print(f"Error checking pip list: {e}")
