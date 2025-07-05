# Standalone Logic Bundle (YAML-Driven Multi-Agent Framework)

## 🔍 How to Use

This bundle is your **modular trading logic engine**. It’s structured for use by:
- Python orchestrators
- LLMs (GPT, Claude, Mistral)
- Hybrid workflows with human and autonomous agents

### 📦 Structure

- `profiles.yaml` – Manifest of all available agent configs.
- `/profiles/*.yaml` – Modular agents with composable logic.
- `/schemas/agent_profile.schema.yaml` – JSON schema for validation.
- `runner.py` – Minimal YAML loader to validate logic integration.
- `/exports/` – Sample analysis JSON files watched by `DataFlowManager`.

## 🧠 GPT & LLM Integration Instructions

- Load `profiles.yaml`, parse each `profiles/*.yaml`.
- Treat each block (e.g., `context_analyzer`, `liquidity_engine`) as a **deterministic logic unit**.
- If `tick_context.enable_tick_merge = true`, perform tick-M1 resampling and inject:
  - `tick_count`, `avg_spread`, `max_spread`, `spread_spike`
- Use values in gating logic (skip entry if `spread_spike` or `tick_count < threshold`).
- Respect all YAML thresholds and flags. **Do not invent logic**.
- All logic execution must follow `workflow:` if present, or top-to-bottom module sequence.

## ✅ Execution Flow

1. Load profile from `profiles.yaml`
2. Inject data (tick + OHLC)
3. Run filters in order:
    - `context_analyzer`
    - `liquidity_engine`
    - `structure_validator`
    - `risk_manager`
    - `journaling`
4. Skip any module where config is missing/empty
5. Log outcome using `journaling` block

- `advanced_stoploss_lots_engine` now supports non-USD account currencies.
  Point values are automatically converted using live FX rates via
  `get_live_fx_rate`.

## 🔧 Strategy Parameters

Key engines pull defaults from `profiles/base/default.yaml`. Adjust values per profile as needed:

```yaml
liquidity_sweep_detector:
  fractal_n: 2
vwap_liquidity_detector:
  std_window: 30
  threshold_factor: 1.5
micro_wyckoff_phase:
  micro_window: 5
  micro_buffer_pips: 5
  pip_size: 0.0001
risk_manager:
  atr_period: 14
  atr_multiplier: 1.5
```

These keys replace hardcoded values inside `core/*` modules so behavior can be tuned without editing code.

### Dynamic Lookback Windows

Enable adaptive window sizing by toggling `dynamic_lookback` in your config. The
helper `core/lookback_adapter.py` shortens lookbacks in explosive markets and
lengthens them during quiet regimes based on `volatility_engine` output.

```yaml
liquidity_engine:
  dynamic_lookback: true
  min_lookback: 3
  max_lookback: 8
phase_detection_config:
  dynamic_lookback: true
  min_lookback: 15
  max_lookback: 40
```

## 📈 Risk Calculation Example

```python
from risk_manager import calculate_risk

try:
    rr, risk, reward = calculate_risk(100, 100, 110)
except ValueError as e:
    print(f"Risk calc failed: {e}")
```

*Multi-agent. Schema-driven. LLM-interpretable. Fully portable.*

## Data Pipeline

`core/data` exposes unified helpers for fetching and resampling OHLCV data.
Use `get_market_data(symbol, timeframe)` to retrieve candles, and
`resample_all(df)` to convert M1 bars to higher timeframes.

Tick CSVs are parsed according to header profiles defined in `core/tick_header_profiles.json`.

### Logging

Logging is configured via `config/logging.json`. Running the pipeline writes
messages to `logs/zanalytics.log` and the console.

```python
from core.data_pipeline import DataPipeline

pipeline = DataPipeline()
pipeline.run_full()
# Inspect logs/zanalytics.log for detailed output
```

## Integrated Context–Catalyst Orchestrator

`core/orchestrator.py` exposes the `AnalysisOrchestrator` class which coordinates trading logic across three layers:

1. **Strategic** – runs Wyckoff regime detection and selects a playbook.
2. **Operational** – executes a Context → Catalyst → Confirmation → Execution sequence.
3. **Technical** – fetches data via `DataManager` and broadcasts events on a simple message bus.

Run it from the CLI using the strategy selector. Any extra flags are passed
directly to the chosen strategy:

```bash
python -m core.orchestrator --strategy advanced_smc --symbol OANDA:EUR_USD --json
```

Set `ZSI_CONFIG_PATH` or supply `--config` to load a different YAML file:

```bash
python -m core.orchestrator --strategy advanced_smc -c custom.yml --symbol OANDA:EUR_USD
```

The command prints a JSON summary describing each stage.

### zsi_config.yaml

Define available orchestrators and the default selection in `zsi_config.yaml` at the project root. The CLI reads this file automatically and you can provide a different path via `--config` or the `ZSI_CONFIG_PATH` environment variable:

```yaml
default_orchestrator: copilot
orchestrators:
  copilot:
    module: core.copilot_orchestrator
    callable: handle_prompt
  advanced_smc:
    module: core.advanced_smc_orchestrator
    callable: run_advanced_smc_strategy
```

The orchestrator falls back to `default_orchestrator` when `--strategy` is omitted.

## ISPTS Pipeline Example

`core/ispts_pipeline.py` implements a minimal, deterministic flow fusing Wyckoff phase logic, SMC structure analysis, inducement sweep detection and microstructure gating. Load the profile in `profiles/ispts_template.yaml` and pass your data frames to `ISPTSPipeline` to reproduce the standard pipeline.

## Local Quick Start

Run the local engine without external API keys:

```bash
./scripts/start_local_ncos.sh
```

The script installs minimal requirements, launches `ncos_local_engine.py` on port 8000, and verifies the `/status` endpoint.

### API Service

Run `zanalytics_api_service.py` to expose a REST API with optional WebSocket streaming:

```bash
python zanalytics_api_service.py
```

Endpoints like `/status` and `/analysis/summary/{symbol}` will be available on port `5010`.

### Redis Configuration

The API caches analysis results and agent decisions in Redis. Configure the host and port via environment variables before starting the service:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

These values default to `localhost:6379` if unset and are loaded by `config_helper.RedisConfig`.

### Regenerating cache files

Running the Streamlit dashboard will recreate any missing `.cache/*.pkl` files automatically.
Use the following command:

```bash
streamlit run "🏠 Home.py"
```

To spin up both the API and dashboard together you can run:

```bash
python run_full_stack.py --api-port 8000 --dash-port 8501
```

The application populates the cache as data is requested.

## License

This project is distributed under the terms of the [ZANALYTICS EULA](LICENSE_EULA.md).
