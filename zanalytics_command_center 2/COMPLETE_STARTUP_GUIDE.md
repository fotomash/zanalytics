# 🚀 ZANFLOW Complete Startup Guide

## Prerequisites Installation

First, install all required dependencies:

```bash
pip install -r requirements.txt
```

If requirements.txt is missing, install these core packages:

```bash
pip install streamlit flask flask-cors pandas numpy plotly pyyaml requests
pip install scikit-learn matplotlib seaborn
```

## 🔧 Module Mapping Fixes

### 1. Fix Python Path Issues

Create a file called `fix_imports.py` in your project root:

```python
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create __init__.py files in all directories
for root, dirs, files in os.walk(project_root):
    if any(f.endswith('.py') for f in files):
        init_file = os.path.join(root, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('')
            print(f"Created {init_file}")

print("✅ Python path fixed!")
```

Run it: `python fix_imports.py`

### 2. Common Import Fixes

Replace these imports in your files:

| Old Import | New Import |
|------------|------------|
| `from data_manager import DataManager` | `from core.data_manager import DataManager` |
| `from analysis_orchestrator import` | `from core.orchestration.analysis_orchestrator import` |
| `import config` | `from config import orchestrator_config` |
| `from wyckoff_analysis import` | `from analysis.wyckoff_analysis import` |

## 📊 Fix Wyckoff Dashboard

The Wyckoff dashboard needs to connect to the correct data sources. Here's the fixed structure:

### 1. Data Manager Connection

Ensure your `DataManager` class has these methods:

```python
class DataManager:
    def get_market_data(self, symbol, timeframe, start_date, end_date):
        """Fetch market data from your data source"""
        # Connect to your data manifest
        # Return DataFrame with columns: open, high, low, close, volume
        pass
```

### 2. Wyckoff Analyzer Structure

```python
class WyckoffAnalyzer:
    def analyze(self, data):
        """Analyze market data using Wyckoff methodology"""
        return {
            'current_phase': 'Accumulation|Markup|Distribution|Markdown',
            'support_levels': [1.0850, 1.0820],
            'resistance_levels': [1.0900, 1.0920],
            'signals': [
                {'type': 'buy', 'message': 'Spring detected at support'},
                {'type': 'info', 'message': 'Volume increasing on rallies'}
            ],
            'volume_insights': {
                'avg_volume': 50000,
                'trend': 'Increasing',
                'unusual_activity': 'None'
            }
        }
```

## 🚀 Complete Startup Sequence

### Step 1: Start API Service

```bash
# Terminal 1
cd /path/to/zanalytics
python zanalytics_api_service.py
```

Expected output:
```
Starting Zanalytics API Service...
 * Running on http://0.0.0.0:5010
```

### Step 2: Start Dashboard

```bash
# Terminal 2
cd /path/to/zanalytics
streamlit run dashboards/Home.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 3: Start Background Services (Optional)

```bash
# Terminal 3 - Analysis Orchestrator
python core/orchestration/analysis_orchestrator.py

# Terminal 4 - Scheduling Agent
python agents/scheduling_agent.py
```

## 🤖 Custom GPT Magic Integration

To add more "magic" commentary from your Custom GPT, create this integration:

### 1. Create `ai_commentary.py`:

```python
import openai
import json
from typing import Dict, Any

class AICommentaryGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key

    def generate_market_commentary(self, analysis_data: Dict[str, Any]) -> str:
        """Generate intelligent market commentary"""

        prompt = f"""
        You are a senior trading analyst. Based on this Wyckoff analysis:

        Phase: {analysis_data.get('current_phase')}
        Support: {analysis_data.get('support_levels')}
        Resistance: {analysis_data.get('resistance_levels')}
        Signals: {analysis_data.get('signals')}

        Provide:
        1. Market structure interpretation
        2. Key levels to watch
        3. Potential scenarios
        4. Risk management advice
        5. Actionable insights

        Be specific, insightful, and practical.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Wyckoff method expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"AI Commentary unavailable: {str(e)}"

    def generate_trade_ideas(self, market_context: Dict[str, Any]) -> list:
        """Generate specific trade ideas"""

        prompt = f"""
        Based on current market conditions:
        {json.dumps(market_context, indent=2)}

        Generate 3 specific trade ideas with:
        - Entry level
        - Stop loss
        - Take profit targets
        - Risk/reward ratio
        - Confidence level
        """

        # Process with GPT-4
        # Return structured trade ideas
        pass
```

### 2. Integrate into Wyckoff Dashboard:

```python
# In your Wyckoff dashboard
from ai_commentary import AICommentaryGenerator

# Initialize
ai_gen = AICommentaryGenerator()

# After analysis
commentary = ai_gen.generate_market_commentary(analysis_results)
trade_ideas = ai_gen.generate_trade_ideas(market_context)

# Display in dashboard
st.markdown("### 🤖 AI Market Intelligence")
st.markdown(commentary)

st.markdown("### 💡 Trade Ideas")
for idea in trade_ideas:
    st.info(idea)
```

## 🔍 Troubleshooting

### Issue: "Module not found"
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add to your scripts:
import sys
sys.path.insert(0, '/path/to/zanalytics')
```

### Issue: "Cannot connect to API"
```bash
# Check if API is running
curl http://localhost:5010/health

# Check firewall
sudo ufw allow 5010
```

### Issue: "No data displayed"
1. Check data_manifest.yml exists
2. Verify data directory structure
3. Check DataManager initialization
4. Look at API logs for errors

## 📁 Correct Project Structure

```
zanalytics/
├── __init__.py
├── requirements.txt
├── zanalytics_api_service.py
├── core/
│   ├── __init__.py
│   ├── data_manager.py
│   └── orchestration/
│       ├── __init__.py
│       └── analysis_orchestrator.py
├── analysis/
│   ├── __init__.py
│   ├── wyckoff_analysis.py
│   └── midas_analysis.py
├── dashboards/
│   ├── Home.py
│   └── pages/
│       ├── __init__.py
│       ├── 📊_Wyckoff_Analysis.py
│       └── 🔧_Strategy_Editor.py
├── data/
│   └── data_manifest.yml
├── knowledge/
│   └── strategies/
│       └── *.yml
└── config/
    └── orchestrator_config.yaml
```

## 🎯 Quick Test

After starting everything, test with:

```python
# test_connection.py
import requests

# Test API
api_response = requests.get('http://localhost:5010/health')
print(f"API Status: {api_response.json()}")

# Test data access
data_response = requests.get('http://localhost:5010/data/market_data')
print(f"Data Access: {data_response.status_code}")
```

## 💡 Pro Tips

1. **Use tmux or screen** for persistent terminal sessions
2. **Set up systemd services** for production
3. **Use environment variables** for configuration
4. **Monitor logs** in real-time: `tail -f logs/*.log`
5. **Create aliases** for common commands

## 🚨 Emergency Commands

```bash
# Kill all Python processes
pkill -f python

# Find process using port
lsof -i :5010
lsof -i :8501

# Force restart
./scripts/restart_all.sh
```

---

This guide should get your ZANFLOW system running smoothly with proper module mappings and enhanced AI commentary!
