# Quick Start Guide

## 1. Setup

First, run the setup script to check dependencies and create directories:

```bash
python setup.py
```

## 2. Prepare Your Data

Place your CSV files in the `data` directory:
- Tick data files should contain "TICK" in the filename
- Bar data files should contain "bars", "M1", "M5", etc. in the filename

Expected tick data columns:
- timestamp, timestamp_ms, bid, ask, volume, last

Expected bar data columns:
- timestamp, open, high, low, close, volume

## 3. Run Analysis

### Using the CLI with all features:
```bash
python analyzer_cli.py --input-dir ./data --enable-all --max-ticks 50000 --max-bars 10000
```

### Using a configuration file:
```bash
python analyzer_cli.py --config config.toml
```

### Using the example script:
```bash
python example_usage.py
```

## 4. View Results

Results are saved as Parquet files in the `output` directory.

To read the results:
```python
import pandas as pd

# Read analyzed tick data
tick_results = pd.read_parquet('output/XAUUSD_TICK_analyzed.parquet')

# Read analyzed bar data
bar_results = pd.read_parquet('output/XAUUSD_M1_bars_analyzed.parquet')
```

## 5. Key Analysis Features

### Smart Money Concepts (SMC)
- Break of Structure (BOS)
- Fair Value Gaps (FVG)
- Order Blocks
- Liquidity Grabs

### Wyckoff Analysis
- Phase detection
- Composite operator tracking
- Volume analysis

### Order Flow
- Volume delta
- CVD divergence
- Absorption/exhaustion patterns

### Advanced Tick Analysis
- HFT detection
- Microstructure patterns
- Information content

## Troubleshooting

If you get import errors:
1. Ensure all files are in the same directory
2. Run `python fix_imports.py` if available
3. Check that all class names match (AdvancedSMCAnalyzer, not SMCAnalyzer)

For other issues, check the logs in the console output.
