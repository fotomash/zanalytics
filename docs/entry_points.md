# Entry Points

This project includes a few official scripts that act as supported entry points for starting or testing the system. All other experimental helpers were moved to the `scripts/` or `misc/` directories to keep the repository tidy.

## main.py
FastAPI application exposing the core HTTP API. Useful for running the service directly during development.

```bash
python main.py
```

## run_system.py
Convenience launcher that verifies Redis is running and then starts the API server with `uvicorn`.

```bash
python run_system.py
```

## run_full_stack.py
Starts the FastAPI service and Streamlit dashboard together. Use `--orchestrator` to also launch the
CLI orchestrator and `--ngrok` to expose the dashboard via a tunnel.

```bash
python run_full_stack.py --api-port 8000 --dash-port 8501 --orchestrator --ngrok
```

## runner.py
Minimal YAML loader that validates agent profiles.

```bash
python runner.py
```

## start_local_ncos.sh
Shell script that spins up the lightweight ncOS engine without external API keys.

```bash
./start_local_ncos.sh
```

## run_zanalytics.sh
Wrapper script that runs startup checks via `zanalytics_startup.py` and then launches the integrated orchestrator.

```bash
./run_zanalytics.sh
```

Experimental scripts, quick tests or utilities have been relocated to `scripts/` or `misc/` so the root directory only contains officially supported entry points and libraries.
