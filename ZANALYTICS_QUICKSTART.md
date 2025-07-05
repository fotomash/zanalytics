# Zanalytics Quick Start Guide

## 1. Installation (5 minutes)

```bash
# Clone or download the zanalytics files
mkdir zanalytics && cd zanalytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy ta scipy scikit-learn loguru
pip install backtrader vectorbt
pip install streamlit plotly
pip install aiohttp beautifulsoup4
pip install loguru
```

## 2. Basic Setup (2 minutes)

```bash
# Create directory structure
mkdir -p config data exports logs

# Copy all zanalytics_*.py files to current directory
# Copy configuration files to config/
```

## 3. First Run (1 minute)

```python
# test_run.py
import asyncio
from zanalytics_orchestrator import ZanalyticsOrchestrator

async def main():
    # Initialize
    orchestrator = ZanalyticsOrchestrator()

    # Run one cycle
    await orchestrator.execute_pipeline()

    print("✓ Analysis complete! Check exports/ folder")

# Run
asyncio.run(main())
```

## 4. View Results

```bash
# Start dashboard
streamlit run "🏠 Home.py"

# View exports
ls exports/
```

## 5. Customize

Edit `config/orchestrator_config.json`:
- Add your symbols
- Set your timeframes
- Configure risk limits

## That's it! 🚀

Full documentation: ZANALYTICS_DOCUMENTATION.md
