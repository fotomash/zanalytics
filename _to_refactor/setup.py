#!/usr/bin/env python3
"""
Setup script for the Unified Market Microstructure Analyzer.
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'pandas', 'numpy', 'scipy', 'scikit-learn', 
        'toml', 'pyarrow', 'matplotlib', 'seaborn', 'plotly'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return missing


def install_dependencies(packages):
    """Install missing packages."""
    if packages:
        print(f"Installing missing packages: {', '.join(packages)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        print("✓ Dependencies installed successfully")


def create_directories():
    """Create required directories."""
    dirs = ['data', 'output', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✓ Created directories: " + ", ".join(dirs))


def verify_files():
    """Verify all required files are present."""
    required_files = [
        'unified_microstructure_analyzer.py',
        'analyzer_cli.py',
        'config.toml'
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print("⚠️  Missing files: " + ", ".join(missing))
        return False

    print("✓ All required files present")
    return True


def main():
    """Run setup."""
    print("Setting up Unified Market Microstructure Analyzer...")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("⚠️  Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]}")

    # Check and install dependencies
    missing = check_dependencies()
    if missing:
        response = input(f"\nInstall missing packages {missing}? (y/n): ")
        if response.lower() == 'y':
            install_dependencies(missing)
        else:
            print("⚠️  Please install missing packages manually")
            sys.exit(1)
    else:
        print("✓ All dependencies installed")

    # Create directories
    create_directories()

    # Verify files
    if not verify_files():
        print("\n⚠️  Please ensure all required files are in the current directory")
        sys.exit(1)

    print("\n✅ Setup complete!")
    print("\nTo run the analyzer:")
    print("  python analyzer_cli.py --help")
    print("  python analyzer_cli.py --config config.toml")
    print("  python example_usage.py")


if __name__ == "__main__":
    main()
