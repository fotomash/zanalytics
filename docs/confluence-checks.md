# Entry Confluence Checks

The `execute_smc_entry` routine applies several optional confluence filters when executing a trade. These checks can be tuned via the strategy configuration.  
The table below lists the available keys.

| Key | Description | Default |
|-----|-------------|---------|
| `maz2.fvg_retest_required` | Require a candle body to re‑test the refined FVG before entry | `true` |
| `maz2.allow_body_engulf` | Permit the mitigation candle to engulf the previous body | `false` |
| `tmc_dss_slope_min` | Minimum DSS slope for a TMC entry | `0.05` |
| `mentfx_dss_slope_min` | Minimum DSS slope for Mentfx variant | `0.1` |
| `mentfx_rsi_overbought` | RSI threshold used for shorts | `70` |
| `mentfx_rsi_oversold` | RSI threshold used for longs | `30` |
| `mentfx_pinbar_body_ratio` | Maximum body size ratio for pinbar detection | `0.3` |

These values may be supplied inside the `risk_model_config` or passed directly to `execute_smc_entry`.
