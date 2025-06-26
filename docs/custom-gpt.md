# Custom GPT Integration

This page summarizes how to initialize a Custom GPT around the **Zanalytics** stack. It distills the boot procedure defined in `prompts/ZANZIBAR_AGENT_INSTRUCTIONS.md` and related configuration files.

## Master Directive
- Always load `prompts/ZANZIBAR_AGENT_INSTRUCTIONS.md` on startup.
- Treat documents listed there (`v5.md`, `v10.md`, `zan_flow_*.md`, `advanced_SL_TP.md`, etc.) as authoritative knowledge.

## Runtime Persona
- Inject persona files: `agent_identity.json`, `conversation_profile.json`, `user_identity.json`, and `trader_psyche.json`.
- These shape tone, decision journaling, and historical preferences.

## Default Settings
- Time zone: **Europe/London**.
- Default timeframes: HTF = M15, LTF = M1.
- Merge CLI flags, YAML profiles, and voice commands via `trait_engine.merge_config()`.

## Initialization Phases
1. **Load Core Config** – merge `copilot_config.json`, `chart_config.json`, `strategy_profiles.json`, and `scalp_config.json`.
2. **Initialize Data** – ingest tick/M1 CSVs from `/mnt/data/`, resample to M5–W, then run the v10 logic on the last 500 rows.
3. **Inject Intermarket Sentiment** – execute `intermarket_sentiment.py` and store `sentiment_snapshot.json`.
4. **Activate Scalping Filter** – load `microstructure_filter.py`, `scalp_filters.py`, and `micro_wyckoff_phase_engine.py`.

The workflow defined in `zanzibar_boot.yaml` runs full analysis: top‑down bias, POI validation, entry confirmation, risk checks, and chart export.

## Kill Zone Patch
- Detect Judas sweeps during London (07:00–10:00 UK) and New York (13:00–16:00 UK).
- Apply the 08:00–08:45 London continuation patch for XAUUSD entries as detailed in `fvg8am.md`.

## Feature Flags
Default flags from `ZANZIBAR_AGENT_INSTRUCTIONS.md`:

| Flag | Default | Purpose |
| --- | --- | --- |
| `autonomous_mode_enabled` | `true` | Auto-run full workflow |
| `auto_generate_charts` | `true` | Export annotated charts |
| `enable_scalping_module` | `true` | Enable tick-level logic |
| `telegram_alert_enabled` | `true` | Send Telegram alerts |

Predictive scoring is controlled by `yaml/predictive_profile.yaml`. Disable with `--predictive_scoring_enabled=false` if needed.

## Outputs
- `journal/zanalytics_log.json`
- `journal/sentiment_snapshot.json`
- `journal/trade_log.csv`
- `journal/accepted_entry_<symbol>_<timestamp>.md`
- `journal/rejected_entry_<symbol>_<timestamp>.md`
- Optional Telegram alerts

## Quick Example
```bash
simulate_setup asset=EURUSD analysis_htf=H1 entry_ltf=M1 conviction=4
```
This triggers data ingestion, sentiment checks, scalping filters, and predictive scoring.

Use ISO‑8601 timestamps with timezone offsets in all logs and messages.
