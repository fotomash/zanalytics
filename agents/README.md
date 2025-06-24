# Agents Overview

This document describes how the Copilot Framework discovers, registers, and executes modular agents. Agents encapsulate domain-specific logic and enable plug‑and‑play extensibility without core orchestration changes.

## Agent Module Contract

Each agent must implement either:
- A standalone `handle_intent(intent: Intent, memory: dict) -> dict` function, **or**
- A subclass of `BaseAgent` with a `handle_intent(self, intent: Intent, memory: dict) -> dict` method, and be decorated with `@register_agent("your_agent_key")`.

### Dynamic Discovery & Registration
At startup, the orchestrator scans the directory specified by the `AGENTS_MODULE_PATH` environment variable (default: `modules/`) for any submodules named `agent.py`. It imports each module and registers:
- Any standalone `handle_intent` function under the module directory name as the agent key.
- Any `BaseAgent` subclass decorated with `@register_agent("<agent_key>")`.
This enables drop‑in extensibility without changing core code.

Each agent must implement a `handle_intent(intent: Intent, memory: dict) -> dict` function, which:
- Parses the incoming structured intent.
- Reads from and updates the shared memory dictionary.
- Returns a response payload containing:
  - `message`: User-facing text.
  - `tags`: Optional list of context tags.
  - `actions`: Optional directives for follow-up agents.
- Agents are auto‑discovered at runtime from the directory defined by the `AGENTS_MODULE_PATH` environment variable (defaults to `modules/`).

## Adding a New Agent

1. Create a new folder `modules/<your_agent_name>/`.
2. Inside, add `agent.py` exporting either:
   - A `handle_intent(intent: Intent, memory: dict) -> dict` function, or
   - A `BaseAgent` subclass decorated with `@register_agent("<agent_key>")`.
3. Define any required schemas or helpers in the same module.
4. On the next application start, the orchestrator will auto-register your agent.

## Example

```python
# modules/example/agent.py
from schema.models import Intent

def handle_intent(intent: Intent, memory: dict) -> dict:
    # Parse intent, update memory
    memory['example_calls'] = memory.get('example_calls', 0) + 1
    return {
        "message": f"Handled by ExampleAgent, call count: {memory['example_calls']}",
        "tags": ["#ExampleAgent"],
        "actions": []
    }
```

## Security & Configuration

Security and configuration are driven entirely by environment variables and middleware. Key settings include:

- Protect `/log` and related endpoints with API Key, OAuth2, or JWT authentication configured via environment variables (`API_KEY_HEADER`, `OAUTH2_SCHEME`, etc.).
- Configure `AGENTS_MODULE_PATH` and other runtime settings via environment variables in your chosen hosting layer (FastAPI, AWS Lambda, etc.).
- Follow security best practices: no hard‑coded secrets, enable CORS policies per environment, and include observability/logging hooks.
- Enable environment‑specific CORS policies via `ALLOWED_ORIGINS` and logging/observability hooks via `LOG_LEVEL` and `TRACE_ENABLED`.

## Observability & Extension
- Use structured logging (e.g., JSON format) and integrate with tools like Prometheus or Datadog.
- Optional: Add metrics middleware to track agent invocation counts, latencies, and error rates.
- For advanced security, integrate OAuth2 scopes or JWT claims verification in `core/orchestrator.py`.
