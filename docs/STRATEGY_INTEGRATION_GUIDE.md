# ZANFLOW Strategy Integration Guide

## Available Strategies for Dashboard Integration

### 1. Smart Money Concepts (SMC)
- **Liquidity Sweep Detector**: `core.liquidity_sweep_detector`
- **POI Manager**: `core.poi_manager_smc` (Order Blocks, FVGs, Breaker Blocks)
- **Entry Executor**: `core.entry_executor_smc`
- **Confirmation Engine**: `core.confirmation_engine_smc`
- **Wick Liquidity Monitor**: `core.wick_liquidity_monitor`

### 2. Inducement & Inversion Strategy
- **ZSI Agent Framework**: Inducement-Sweep-POI detection
- **Liquidity Engine**: `core.liquidity_engine_smc`
- **Analysis Orchestrator**: `core.orchestrator.AnalysisOrchestrator`

### 3. Wyckoff Analysis
- **Phase Engine**: `core.wyckoff_phase_engine`
- **Micro Wyckoff**: `core.micro_wyckoff_phase_engine`
- **Phase Detector**: `core.phase_detector_wyckoff_v1`

### 4. Advanced Strategies
- **MENTFX ICI**: Impulse-Correction-Impulse patterns
- **VSA Signals**: Volume Spread Analysis
- **Divergence Engine**: Multi-indicator divergences
- **Fibonacci Filter**: Advanced fib analysis

### 5. Risk Management
- **Risk Model**: `core.risk_model`
- **Advanced Stop-Loss Engine**: `core.advanced_stoploss_lots_engine`

### 6. Market Intelligence
- **Intermarket Sentiment**: Cross-market correlations
- **Macro Sentiment Enricher**: Fundamental analysis integration

## LLM Integration Points

1. **Pattern Interpretation**: Convert technical patterns to narratives
2. **Multi-Strategy Confluence**: Combine signals from all strategies
3. **Risk-Adjusted Recommendations**: Integrate risk metrics
4. **Alert Generation**: Create actionable trading alerts
5. **Educational Insights**: Explain why patterns are significant

## Usage Example

```python
# Initialize the enhanced dashboard
from enhanced_unified_dashboard import UnifiedAnalysisEngine

engine = UnifiedAnalysisEngine()

# Select strategies
strategies = {
    "SMC Advanced": ["liquidity_sweep_detector", "poi_manager_smc"],
    "Wyckoff Analysis": ["wyckoff_phase_engine"],
    "Risk Management": ["risk_model"]
}

# Run analysis
results = await engine.run_comprehensive_analysis(df, strategies)

# Get LLM insights
llm_analysis = results['llm_analysis']
print(f"Confluence Score: {llm_analysis['confluence_score']}%")
print(f"Recommendation: {llm_analysis['recommendation']}")
```
