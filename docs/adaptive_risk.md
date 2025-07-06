# Adaptive Risk Configuration

The `RiskManagerAgent` supports dynamic position sizing based on the maturity score produced by the `PredictiveScorer`.
Below is an example YAML block that maps score thresholds to risk percentages.

```yaml
risk_manager:
  base_risk_pct: 1.0
  score_risk_tiers:
    - threshold: 0.80
      risk_pct: 0.5
    - threshold: 0.60
      risk_pct: 1.0
    - threshold: 0.40
      risk_pct: 1.5
```

Scores are evaluated from highest to lowest. The first tier whose threshold is met determines the recommended risk percentage.
