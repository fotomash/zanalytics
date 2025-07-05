# Zanalytics Trading System Documentation

## Overview

Zanalytics is a comprehensive trading system that integrates advanced market analysis, signal generation, backtesting, and LLM-ready data formatting. The system consists of multiple components working together to provide institutional-grade trading analytics.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ZANALYTICS ORCHESTRATOR                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │Data Pipeline│  │ Integration  │  │Signal Generator  │ │
│  │             │──│   Engine     │──│                  │ │
│  └─────────────┘  └──────────────┘  └──────────────────┘ │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Backtester  │  │ LLM Formatter│  │   Dashboard      │ │
│  │             │  │              │  │   (Streamlit)    │ │
│  └─────────────┘  └──────────────┘  └──────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Microstructure Analyzers                │  │
│  │  • Wyckoff Analyzer  • OrderFlow  • Smart Money     │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. **Zanalytics Orchestrator** (`zanalytics_orchestrator.py`)
The master control system that coordinates all components.

**Key Features:**
- Automated pipeline execution
- State management
- Risk management integration
- Scheduled updates

### 2. **Data Pipeline** (`zanalytics_data_pipeline.py`)
Handles data fetching, cleaning, and enrichment.

**Features:**
- Multi-timeframe data processing
- Technical indicator calculation
- Market microstructure metrics
- Volume profile analysis

### 3. **Integration Engine** (`zanalytics_integration.py`)
Combines multiple analysis methods into consensus signals.

**Analyzers:**
- Wyckoff Method
- Order Flow Analysis
- Smart Money Concepts (SMC)
- Custom indicators

### 4. **Signal Generator** (`zanalytics_signal_generator.py`)
Generates actionable trading signals with risk parameters.

**Signal Types:**
- Breakout signals
- Reversal signals
- Continuation patterns
- Divergence signals

### 5. **LLM Formatter** (`zanalytics_llm_formatter.py`)
Formats analysis data for optimal LLM consumption.

**Output Formats:**
- Structured prompts
- Context-aware summaries
- Trading recommendations
- Risk assessments

### 6. **Backtesting Framework** (`zanalytics_backtester.py`)
Comprehensive backtesting with multiple engines.

**Engines:**
- Backtrader integration
- VectorBT integration
- Custom strategy testing
- Walk-forward analysis

### 7. **Dashboard** (`dashboard/app.py`)
Interactive Streamlit dashboard for visualization.

**Features:**
- Real-time price charts
- Signal visualization
- Performance metrics
- Risk analytics
- Orchestrator results written to `dashboard/latest_update.json`

## Installation

### Prerequisites
```bash
# Python 3.8 or higher required
python --version

# Create virtual environment
python -m venv zanalytics_env
source zanalytics_env/bin/activate  # On Windows: zanalytics_env\Scripts\activate
```

### Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt  # now includes loguru

# Additional dependencies for specific components
pip install backtrader vectorbt streamlit plotly loguru
```

### Directory Structure
```
zanalytics/
├── config/
│   ├── orchestrator_config.json
│   ├── pipeline_config.json
│   ├── signal_config.json
│   └── backtest_config.json
├── data/
│   └── market_data/
├── exports/
│   ├── analysis/
│   ├── signals/
│   └── llm_ready/
├── logs/
├── microstructure_analyzers/
│   ├── zanflow_microstructure_analyzer.py
│   ├── ncOS_ultimate_microstructure_analyzer.py
│   └── convert_final_enhanced_smc_ULTIMATE.py
└── src/
    ├── zanalytics_orchestrator.py
    ├── zanalytics_data_pipeline.py
    ├── zanalytics_integration.py
    ├── zanalytics_signal_generator.py
    ├── zanalytics_llm_formatter.py
    ├── zanalytics_backtester.py
    ├── zanalytics_backtest_analyzer.py
    └── dashboard/app.py
```

## Quick Start

### 1. Basic Setup
```python
from zanalytics_orchestrator import ZanalyticsOrchestrator

# Initialize orchestrator
orchestrator = ZanalyticsOrchestrator("config/orchestrator_config.json")

# Run single analysis
import asyncio
asyncio.run(orchestrator.execute_pipeline())
```

### 2. Running the Dashboard
```bash
streamlit run dashboard/app.py
```

### 3. Generating LLM-Ready Data
```python
from zanalytics_llm_formatter import LLMDataFormatter

formatter = LLMDataFormatter()
llm_data = formatter.format_for_llm(
    data=analysis_results,
    query_type="market_analysis"
)
```

## Configuration

### Orchestrator Configuration
```json
{
  "symbols": ["BTC/USDT", "ETH/USDT"],
  "timeframes": ["1h", "4h", "1d"],
  "update_interval": 300,
  "risk_limits": {
    "max_positions": 3,
    "max_risk_per_trade": 0.02
  }
}
```

### Signal Configuration
```json
{
  "min_confidence": 0.6,
  "signal_types": ["breakout", "reversal"],
  "risk_reward_ratio": 2.0
}
```

## Usage Examples

### Example 1: Custom Analysis Pipeline
```python
from zanalytics_data_pipeline import DataProcessor
from zanalytics_integration import IntegrationEngine

# Process data
processor = DataProcessor()
data = processor.fetch_and_enrich("BTC/USDT", "1h")

# Run analysis
engine = IntegrationEngine()
analysis = engine.run_integrated_analysis(data)
```

### Example 2: Backtesting Strategy
```python
from zanalytics_backtester import BacktestingFramework

backtester = BacktestingFramework()
results = backtester.run_backtest(
    strategy_class="MyStrategy",
    data=historical_data,
    params={"stop_loss": 0.02}
)
```

### Example 3: Real-time Signal Generation
```python
from zanalytics_signal_generator import SignalGenerator

generator = SignalGenerator()
signals = generator.generate_signals(
    market_data=current_data,
    analysis=analysis_results
)
```

## API Integration

### Exchange Connections
```python
# Configure in orchestrator_config.json
{
  "exchange": {
    "name": "binance",
    "api_key": "your_api_key",
    "api_secret": "your_secret"
  }
}
```

### Webhook Support
```python
# Send signals to webhook
orchestrator.config["webhooks"] = {
  "discord": "https://discord.com/api/webhooks/...",
  "telegram": "https://api.telegram.org/bot..."
}
```

## Advanced Features

### 1. Multi-Strategy Backtesting
```python
strategies = [
    {"name": "Breakout", "params": {...}},
    {"name": "MeanReversion", "params": {...}}
]

for strategy in strategies:
    results = backtester.run_backtest(**strategy)
```

### 2. Custom Indicators
```python
def custom_indicator(data):
    # Your custom logic
    return indicator_values

processor.add_custom_indicator(custom_indicator)
```

### 3. Risk Management
```python
from zanalytics_risk import RiskManager

risk_manager = RiskManager({
    "max_drawdown": 0.10,
    "position_sizing": "kelly"
})
```

## Performance Optimization

### 1. Data Caching
```python
processor.enable_caching(cache_dir="data/cache")
```

### 2. Parallel Processing
```python
orchestrator.config["parallel_processing"] = True
orchestrator.config["num_workers"] = 4
```

### 3. Memory Management
```python
# Limit data retention
processor.config["max_data_points"] = 5000
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt  # includes loguru
   ```

2. **API Rate Limits**
   ```python
   # Configure rate limiting
   orchestrator.config["rate_limit"] = {
     "requests_per_minute": 60
   }
   ```

3. **Memory Issues**
   ```python
   # Reduce data size
   processor.config["downsample"] = True
   ```

## Best Practices

1. **Data Management**
   - Regular data validation
   - Implement data versioning
   - Use appropriate timeframes

2. **Risk Management**
   - Always set stop losses
   - Diversify across symbols
   - Monitor correlation

3. **Backtesting**
   - Use walk-forward analysis
   - Account for slippage/fees
   - Validate on out-of-sample data

## Support and Contributing

- GitHub: [your-repo-url]
- Documentation: [docs-url]
- Discord: [discord-invite]

## License

ZANALYTICS EULA - See [LICENSE_EULA.md](LICENSE_EULA.md) for details

## Changelog

### Version 1.0.0 (2024-06-25)
- Initial release
- Core components implemented
- Basic documentation

---

For more detailed information on each component, refer to the individual module documentation.
