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

Run the orchestrator directly from the command line using the new CLI:

```bash
python -m core.orchestrator --strategy advanced_smc
```

This launches `core.orchestrator.AnalysisOrchestrator` with the
strategy defined in your `zsi_config.yaml` file and prints a summary
when the run completes.

## 4. View Results

```bash
# Start dashboard
streamlit run "🏠 Home.py"

# View exports
ls exports/
```

## 5. Customize

Edit `config/orchestrator_config.yaml`:
- Add your symbols
- Set your timeframes
- Configure risk limits

## That's it! 🚀

Full documentation: ZANALYTICS_DOCUMENTATION.md
