# ZANFLOW v12 Intelligence Evolution

## Overview

The Intelligence Evolution (Phase 5) adds three powerful enhancements to the ZANFLOW v12 trading system:

1. **Confluence Path Tracking** - Formalizes event sequences as first-class citizens
2. **Adaptive Risk Management** - Dynamically adjusts position sizing based on maturity scores
3. **Meta-Agent Learning** - Continuously analyzes and optimizes system performance

## Key Components

### 1. Confluence Path Tracker

The Confluence Path Tracker makes the sequence of market events a core part of your system's intelligence.

**Features:**
- Tracks the exact sequence of events leading to each trade
- Generates unique signatures for each confluence path pattern
- Maintains performance statistics for each path type
- Identifies top-performing sequences for prioritization

**Usage Example:**
```python
# In your agent's validation logic
tracker = ConfluencePathTracker()

# Start tracking a new opportunity
path = tracker.start_path("TRADE_001", "EURUSD", "M5")

# Add events as they occur
tracker.add_event("TRADE_001", "HTF_BIAS_CONFIRMED", 0.85, 1500)
tracker.add_event("TRADE_001", "SWEEP_VALIDATED", 0.95, 3500)
tracker.add_event("TRADE_001", "BOS_CONFIRMED", 0.88, 2200)

# Complete the path with outcome
outcome = {"executed": True, "r_multiple": 2.5}
tracker.complete_path("TRADE_001", 0.91, outcome)
```

### 2. Adaptive Risk Manager

The Adaptive Risk Manager connects maturity scores directly to position sizing, making your system truly adaptive.

**Features:**
- Multiple risk curves (conservative, moderate, aggressive)
- Maturity score-based position sizing
- Market condition adjustments (killzone, news, volatility)
- Daily risk limits and correlation adjustments

**Risk Curves:**
- **Conservative**: Stepped levels (0.85+ = 1%, 0.70-0.85 = 0.5%, etc.)
- **Moderate**: Linear interpolation between min/max risk
- **Aggressive**: Exponential scaling for high-conviction trades

**Usage Example:**
```python
risk_manager = AdaptiveRiskManager()

# Calculate position size
conditions = {
    "killzone_active": True,
    "high_impact_news": False,
    "volatility": 0.8
}

risk_profile = risk_manager.calculate_position_size(
    symbol="EURUSD",
    maturity_score=0.88,
    stop_distance_pips=15,
    account_balance=100000,
    current_conditions=conditions
)

print(f"Risk: {risk_profile.risk_percent}%")
print(f"Position Size: {risk_profile.position_size} lots")
```

### 3. Meta-Agent for Learning

The Meta-Agent is your system's learning engine, analyzing performance and providing data-driven recommendations.

**Analysis Modules:**
- **Confluence Path Analyzer**: Identifies winning and losing path patterns
- **Rejection Analyzer**: Evaluates effectiveness of rejection rules
- **Maturity Correlation Analyzer**: Optimizes maturity score thresholds
- **Risk Optimizer**: Calculates optimal risk using Kelly Criterion

**Weekly Analysis Output:**
- Performance report for all confluence paths
- Rejection rule effectiveness metrics
- Optimal risk parameters based on recent performance
- Actionable recommendations with confidence scores

## Integration with ZANFLOW v12

The Intelligence Evolution integrates seamlessly with your existing system through the `IntelligenceIntegration` class.

### Enhanced Agent State

Your agents can now enhance their state with intelligence features:

```python
# In your agent's execute() method
intelligence = IntelligenceIntegration()

# Enhance the agent state
agent_state = intelligence.enhance_agent_state(agent_state)

# The state now includes:
# - Confluence path tracking
# - Adaptive risk calculations
# - Path signatures and recommendations
```

### API Endpoints

New API endpoints for intelligence features:

```
POST   /api/intelligence/enhance-state      # Enhance agent state
POST   /api/intelligence/complete-tracking  # Complete trade tracking
GET    /api/intelligence/recommendations    # Get path recommendations
PUT    /api/intelligence/risk-curve         # Update risk curve
POST   /api/intelligence/optimize           # Run optimization cycle
GET    /api/intelligence/status            # Get system status
```

## Configuration Files

### adaptive_risk_config.yaml
Configure risk curves, position sizing rules, and market condition adjustments.

### meta_agent_config.yaml
Set analysis schedules, enable/disable modules, and configure optimization targets.

### confluence_path_schema.json
Defines the structure for confluence path tracking and standard event types.

## Best Practices

1. **Path Tracking**: Always complete path tracking for both executed and rejected trades
2. **Risk Curves**: Start with conservative and move to moderate as confidence grows
3. **Meta-Agent**: Run weekly analysis on weekends when markets are closed
4. **Recommendations**: Review all recommendations before auto-applying

## Performance Impact

The Intelligence Evolution adds minimal overhead:
- Path tracking: ~1ms per event
- Risk calculations: ~5ms per calculation
- Meta-agent analysis: Runs asynchronously (weekly)

## Next Steps

1. **Integration**: Use the provided integration script to add these features
2. **Configuration**: Customize the YAML configs for your trading style
3. **Monitoring**: Use the status API to monitor intelligence components
4. **Optimization**: Let the system run for 2-4 weeks before applying major recommendations

The Intelligence Evolution transforms ZANFLOW v12 from a sophisticated trading system into a truly intelligent, self-optimizing platform that learns and adapts to market conditions.
