# Orchestrator CLI

The `AnalysisOrchestrator` can be invoked directly from the command line. Use `--strategy` to choose which orchestrator implementation to run. Any additional flags are forwarded to that strategy as keyword arguments.

```bash
python -m core.orchestrator --strategy copilot --prompt "help" --json
python -m core.orchestrator --strategy advanced_smc --symbol OANDA:EUR_USD
```

Specify `--config` to load a custom `zsi_config.yaml`:

```bash
python -m core.orchestrator -c path/to/config.yaml --strategy copilot
```
