# Tick Manipulation Dashboard

## Quick Start

1. Install requirements:
   ```bash
   pip install -r requirements_dashboard.txt
   ```

2. Run the dashboard:
   ```bash
   streamlit run tick_manipulation_dashboard.py
   ```

## Features

- **Dynamic Module Loading**: Automatically loads your analyzer modules
- **Multiple Data Sources**: Supports CSV and Parquet files
- **Manipulation Detection**: Identifies spoofing and wash trading
- **Advanced Analysis**: Integrates with your ncOS and Zanflow analyzers
- **LLM Ready**: Export data formatted for GPT analysis

## Module Integration

Place these files in the same directory:
- `ncOS_ultimate_microstructure_analyzer_DEFAULTS.py`
- `zanflow_microstructure_analyzer.py`
- Your data files (CSV/Parquet)

The dashboard will automatically detect and load available modules.

## Extending the Dashboard

Add new analysis functions in the "Advanced Analysis" tab:

```python
if 'your_module' in modules:
    analyzer = modules['your_module'].YourAnalyzer()
    results = analyzer.analyze(df)
```
