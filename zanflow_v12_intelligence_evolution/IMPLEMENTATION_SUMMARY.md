# Intelligence Evolution - Implementation Summary

This package extends ZANFLOW v12 with three tightly coupled components:

1. **Confluence Path Tracker** – Captures the exact sequence of market events that lead to each trade.
2. **Adaptive Risk Manager** – Calculates position sizing based on maturity scores and market conditions.
3. **Meta-Agent** – Performs weekly analysis and optimization using the accumulated path statistics.

## Interaction Flow

- Agents call `IntelligenceIntegration` to enhance their state. When a new opportunity appears, a `ConfluencePath` is started and events are appended as validation passes.
- When an entry is considered, the integration delegates to the Adaptive Risk Manager to determine an appropriate risk percentage and position size. These calculations are influenced by the active risk curve defined in `configs/adaptive_risk_config.yaml`.
- After a trade completes (or is abandoned) the integration calls `complete_trade_tracking` so the path data and outcome are logged to the journal defined in `orchestrator_config.yaml`.
- The Meta-Agent reads this journal weekly. Using settings from `configs/meta_agent_config.yaml`, it analyzes path performance, rejection effectiveness and maturity correlations. Recommendations may include switching risk curves or prioritising successful path signatures.

## Configuration Examples

```yaml
# configs/adaptive_risk_config.yaml
adaptive_risk_config:
  overrides:
    killzone_multiplier: 1.2
    max_daily_risk: 2.0
  risk_curves:
    conservative:
      curve_type: stepped
      levels:
        - maturity_min: 0.85
          maturity_max: 1.0
          risk_percent: 1.0
```

```yaml
# configs/meta_agent_config.yaml
meta_agent_config:
  schedule:
    day: sunday
    time: '18:00'
    frequency: weekly
  analysis_modules:
    confluence_path_analyzer:
      enabled: true
```

The schema `schemas/confluence_path_schema.json` defines the structure of each path, ensuring a consistent record of events, maturity scores and outcomes.

## Expected Workflow

1. **Opportunity Detected** – Agent begins a new confluence path.
2. **Validation Events** – Path Tracker records each confirmation step.
3. **Risk Calculation** – When ready for entry, the Adaptive Risk Manager provides risk and position size based on the current maturity score and market conditions.
4. **Trade Outcome** – On completion, the path is finalized and stored in the journal.
5. **Weekly Analysis** – The Meta-Agent processes the journal, generating reports and recommendations that can adjust risk curves or highlight profitable path signatures.

With these components working together, ZANFLOW evolves into a self-improving trading system where historical event sequences guide future risk decisions and ongoing optimization.
