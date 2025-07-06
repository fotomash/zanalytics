# ZANFLOW v12 Intelligence Evolution - Quick Start

## Installation

1. Extract the package to a temporary directory
2. Run the integration script:
   ```bash
   python integrate.py /path/to/your/zanflow
   ```

## Immediate Actions

1. **Update Your Agents**
   ```python
   from core.agents.enhanced_base import EnhancedAgentBase

   class YourAgent(EnhancedAgentBase):
       # Your agent now has intelligence features!
   ```

2. **Configure Risk Curves**
   Edit `configs/adaptive_risk_config.yaml` to match your risk tolerance

3. **Enable Weekly Analysis**
   The meta-agent will run automatically every Sunday at 6 PM

## First Week

- Let the system collect confluence path data
- Monitor path signatures in your logs
- Check risk calculations are appropriate

## Week 2+

- Review meta-agent recommendations
- Apply high-confidence optimizations
- Watch your system improve!

For full documentation, see README.md
