<file name=docs/schemas.md><!-- Version: 1.6.0 | Last Updated: 2024-08-01 -->
# Schemas

Defines Pydantic models and JSON schemas for the generic Copilot framework.

## Intent Model

```python
from pydantic import BaseModel, Field
from typing import Any, Dict
from datetime import datetime

class Intent(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    business_type: str = Field(..., description="Logical agent or domain to handle this intent (e.g., 'dietpilot', 'carematch')")
    intent_type: str = Field(..., description="Specific intent within the business domain (e.g., 'log_meal', 'record_step')")
    payload: Dict[str, Any] = Field(..., description="Arbitrary data required to fulfill the intent")
    timestamp: datetime = Field(..., description="UTC timestamp of when the intent was created")
```

## AgentResponse Model

```python
from pydantic import BaseModel, Field
from typing import Any, Dict, List

class AgentResponse(BaseModel):
    success: bool = Field(..., description="Whether the agent handled the intent successfully")
    message: str = Field(..., description="User-facing response text")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured data or suggestions returned by the agent")
    triggers: List[str] = Field(default_factory=list, description="Any follow-up intents or agents to trigger next")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata, e.g., diagnostics or timing info")
```

## UserMemory Model

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Any

class UserMemory(BaseModel):
    """User memory snapshot model."""
    schema_version: str = Field("1.0", description="Schema version of this memory record")
    user_id: str = Field(..., description="Unique identifier for the user")
    date: str = Field(..., description="Date string (YYYY-MM-DD) for this memory snapshot")
    context: Dict[str, Any] = Field(default_factory=dict, description="Persistent key/value state across requests")
    agent_calls: List[str] = Field(default_factory=list, description="Ordered list of agent keys invoked today")
    logs: List[Dict[str, Any]] = Field(default_factory=list, description="Raw intent payloads and responses for audit")
```

## Config Model

```python
from pydantic import BaseSettings, Field
from typing import List, Dict, Any

class AppConfig(BaseSettings):
    environment: str = Field(..., env="ENVIRONMENT", description="Deployment environment (e.g., 'dev', 'staging', 'prod')")
    allowed_business_types: List[str] = Field(..., env="ALLOWED_BUSINESS_TYPES", description="List of business_type keys that map to agent modules")
    default_memory_backend: str = Field(..., env="DEFAULT_MEMORY_BACKEND", description="Default storage backend (e.g., 'json', 'firestore', 'dynamo')")
    agent_registry: Dict[str, str] = Field(..., env="AGENT_REGISTRY", description="Mapping of business_type to Python module path")
    llm_settings: Dict[str, Any] = Field(..., env="LLM_SETTINGS", description="Settings for the LLM provider (model name, temperature, etc.)")
    webhook_urls: Dict[str, str] = Field(default_factory=dict, env="WEBHOOK_URLS", description="Optional external webhook endpoints")
    auth: Dict[str, Any] = Field(default_factory=dict, env="AUTH", description="Authentication configuration (e.g., JWT provider, OAuth settings)")
```

## Environment Configuration

- All sensitive configuration values (API keys, database URLs, secrets) **must** be supplied via environment variables or a secrets manager.
- For local development, provide a `.env.example` with placeholder values; do **not** commit `.env` to version control.
- In production, integrate with a dedicated secrets store (e.g., HashiCorp Vault, AWS Secrets Manager, Azure Key Vault) and inject at deploy time.
- Ensure environment variable names match Pydantic `env` fields in `AppConfig`.

> **Note**: `AppConfig` reads its values from environment variables or a .env file. To generate a template `.env.example`, run:
>
> ```bash
> python -c "from schema.models import AppConfig; print(AppConfig.schema_json(indent=2))" > .env.example
> ```

## JSON Schemas

Corresponding JSON schema files should live in `schema/`:
- `intent.json`
- `agent_response.json`
- `user_memory.json`
- `app_config.json`

## JSON Schemas and Generation
Each Pydantic model can emit its JSON schema automatically. 
Example:
```python
from schema.models import Intent
print(Intent.schema_json(indent=2))
```
Store the output under `schema/intent.json`, etc.

## Version History
| Version | Date       | Description                         |
| ------- | ---------- | ----------------------------------- |
| 1.3     | 2024-06-01 | Initial schema definitions          |
| 1.4     | 2024-07-xx | Added BaseSettings env-var support  |

## Security Best Practices

1. **Least Privilege**: Grant minimal permissions to your memory backend and external services.
2. **Rotation**: Regularly rotate secrets and enforce automated rotation policies.
3. **Encryption In Transit & At Rest**: Use TLS for all network communication and encrypt stored secrets.
4. **Audit Logging**: Enable audit logging on your configuration stores and memory backends.
5. **Rotate Dev Credentials**: Never share long-lived credentials in development; use short-lived tokens.

## Observability & Logging

- Instrument all FastAPI endpoints with structured logging (e.g., JSON logs) for request tracing.
- Include request IDs and user IDs in every log entry for auditability.
- Integrate with a centralized logging system (e.g., Loki, CloudWatch, Datadog) via environment-configurable endpoints.
- Expose metrics (request count, latency, error rates) via Prometheus middleware.

## Monitoring & Alerting

- Configure health-check endpoints (`/health`, `/metrics`) and expose via load balancer.
- Define SLOs for request latency and error budget; alert on breaches.
- Use automated alerting (PagerDuty, Slack) for critical failures in memory backend or agent discovery.

## Future Expansion Hooks

- **Plugin Registration**: When new Pydantic models are added, document their JSON schema output path under `schema/`.
- **Agent Extensions**: Update `AGENT_REGISTRY` in `AppConfig` to include new `business_type` keys.
- **Custom Middleware**: Provide templates for JWT/OAuth2 authentication in `core/middleware.py`.
- **Schema Versioning**: Adopt URL-based versioning (e.g., `/schema/v1/intent.json`) for backward compatibility.

> **Tip**: After adding new models or config fields, regenerate `.env.example` and JSON schemas with:
> ```bash
> python -c "from schema.models import AppConfig, Intent, AgentResponse, UserMemory; \
> print(AppConfig.schema_json(indent=2)) > .env.example; \
> print(Intent.schema_json(indent=2)) > schema/intent.json; \
> print(AgentResponse.schema_json(indent=2)) > schema/agent_response.json; \
> print(UserMemory.schema_json(indent=2)) > schema/user_memory.json"
> ```

## Changelog

- **v1.6.0** (2024-08-01): 
  - Added Observability & Logging section  
  - Added Monitoring & Alerting section  
  - Added Future Expansion Hooks and regeneration tip  