# Endpoints

> **Security:** All endpoints require a valid API key in the `X-API-Key` header. See [Security](security.md) for details.

- `POST /log` — Log a user intent (meal, mood, steps, etc.)
- `GET /health` — Check service health

# API Endpoints

## Core Functionality
- `POST /log`  
  Log a user intent (e.g., meal, mood, steps, etc.).  
  **Payload:** Intent JSON per schema.  
  **Response:** Updated memory snapshot + agent response.

- `GET /summary`  
  Retrieve a summary of today's logs (macros, activity, mood).  
  **Response:** Summary JSON.

- `POST /reset`  
  Clear today's memory and start fresh.  
  **Response:** Confirmation message and empty memory state.

## Management & Inspection
- `GET /health`  
  Check service health and readiness.  
  **Response:** `{ "status": "ok", "uptime": "..." }`

- `GET /agents`  
  List all loaded agents and their status.  
  **Response:** `{ "agents": ["macronator", "suppbro", ...] }`

- `GET /memory`  
  Fetch the raw user memory state for today.  
  **Query Params:** `user_id` (optional)  
  **Response:** Memory JSON.

- `GET /config`  
  Retrieve current runtime configuration (e.g., active environment, feature flags).  
  **Response:** Config JSON.

## Proactive & Internal
- `POST /nudge`  
  Trigger a proactive nudge or scheduled check (e.g., morning reminder).  
  **Payload:** `{ "user_id": "...", "nudge_type": "daily_check" }`  
  **Response:** Nudge message.

- `GET /docs`  
  Serve API documentation or route to MkDocs UI.  
  **Response:** HTML or redirect to `/docs/index.html`.

> **Note:** Runtime configurations (CORS, feature flags, environment-specific settings) are managed via `mkdocs/env.yml`.