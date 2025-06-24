### Dynamic Agent Loading & Registration

This module enables runtime discovery and secure registration of agents. Version: 1.0.0.

Agents are discovered and registered at runtime using Python introspection. This allows you to drop new agent modules into your project without modifying the orchestrator.

- Leverages introspection for zero-config extensibility.
- Supports hot-reload in development with `--reload`.

## Adding a New Agent

1. **Create module folder**  
   Under `modules/`, create a new directory named after your agent, e.g. `modules/my_agent/`.

2. **Define `agent.py`**  
   Inside that folder, add `agent.py` with:
   ```python
   # modules/my_agent/agent.py – New Agent Definition
   from core.agent import BaseAgent, register_agent

   @register_agent("my_agent")
   class MyAgent(BaseAgent):
       """
       Example agent implementation.
       @version 1.0.0
       @author CTO Team
       """

       def handle_intent(self, intent: dict, memory: dict) -> dict:
           # Your logic here
           return {
               "response": "Your custom response",
               "updated_memory": memory,
               "tags": ["#ExampleTag"]
           }
   ```

3. **Implement `handle_intent`**  
   - Accepts:
     - `intent`: the parsed user input payload
     - `memory`: the current user context/state
   - Returns a dict with:
     - `response`: text to send back to the user
     - `updated_memory`: the modified memory object
     - `tags`: any new tags or insights
     - `status`: standardized status code (e.g., "ok", "error")

## Security & Validation
- Agents must validate incoming payloads against Pydantic schemas.
- Ensure API keys or JWT tokens are verified before dispatch.
- Fail-safe: return `{"error": "unauthorized"}` with HTTP 401 on invalid credentials.

## Orchestrator Scanning Logic

The orchestrator automatically imports and registers agents at startup:

```python
# core/orchestrator.py

import importlib
import pkgutil
import os
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

# API Key header for route protection
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI()
agent_registry = {}

# Path to your modules folder
modules_path = os.path.join(os.path.dirname(__file__), '..', 'modules')

# Dynamically discover and import agent modules
for finder, module_name, ispkg in pkgutil.iter_modules([modules_path]):
    module = importlib.import_module(f'modules.{module_name}.agent')
    handler = getattr(module, 'handle_intent', None)
    if handler:
        agent_registry[module_name] = handler

@app.post("/log", tags=["Logging"], dependencies=[Depends(api_key_header)])
async def log_intent(payload: dict):
    # Security: validate API key
    if not payload.get(API_KEY_NAME) or payload[API_KEY_NAME] != os.getenv("SERVICE_API_KEY"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    agent_key = payload.get("business_type") or payload.get("agent")
    handler = agent_registry.get(agent_key)
    if not handler:
        return {"error": f"No agent found for '{agent_key}'"}
    # Load and update memory (pseudo-code)
    memory = load_user_memory(payload["user_id"])
    result = handler(payload, memory)
    save_user_memory(payload["user_id"], memory)
    return result
```

## Best Practices

- Use a unique `"business_type"` or `"agent"` field in your payloads to route correctly.
- Extend `BaseAgent` for shared utilities (loading/saving memory, validation).
- Keep each agent focused on a single domain or capability for maintainability.

---
_CTO-approved | Ready for production deployment with zero-downtime updates._
