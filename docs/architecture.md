# Architecture

## Overview
This application employs a modular orchestrator and agent architecture built on FastAPI and designed for scalability and maintainability. Agents are dynamically discovered and loaded at runtime, enabling seamless integration of new business logic without modifying core code.

**Configuration-Driven & Environment-Agnostic**  
All runtime settings—including API endpoints, database connections, feature flags, and deployment targets—are managed through a layered configuration system:

1. zsi_config.yaml with environment overlays (e.g. `zsi_config.development.yaml`, `zsi_config.production.yaml`)
2. Environment variables (loaded via Pydantic BaseSettings or python-decouple)

Example Pydantic settings loader:
```python
from pydantic import BaseSettings

class AppConfig(BaseSettings):
    ENV: str
    DATABASE_URL: str
    FEATURE_FLAGS: dict = {}
    LOG_LEVEL: str = "INFO"
    DEPLOY_TARGET: str = "railway"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

config = AppConfig()
```

- No credentials, URLs, or platform specifics are hard-coded.
- Switch deployment targets (Railway, AWS Lambda, Azure Functions, GCP, Docker) by changing `DEPLOY_TARGET` or YAML profile.
- Configuration is validated at startup, with sensible defaults and required overrides.

## Components
- **Orchestrator** (`main.py` / `core/orchestrator.py`): Central entry point handling all user requests, configuration loading, and routing to agents.
- **Agents** (`modules/`): Each agent resides in its own directory and exposes either a `BaseAgent` subclass or a `handle_intent` function. The orchestrator dynamically imports and registers them by `business_type`.
- **Configuration** (`zsi_config.yaml` + env): Centralizes all environment-specific settings—deployment profiles (Railway, AWS, Azure, GCP), API keys, database URIs, logging levels, and feature toggles.
- **Flows** (`flows/`): Markdown files defining user journeys and dialogue flows.
- **System** (`system/`): Core prompts, routing logic, and memory schema definitions.
- **Documentation** (`docs/`): Automatically generated MkDocs-based documentation covering architecture, usage, and development guidelines.

## Data Flow
```
User Input
  → Load Configuration (zsi_config.yaml + env)
  → FastAPI Endpoint
  → Plugin Registry (plugins/)
  → Dynamic Agent Discovery (modules/)
  → Orchestrator Routes Request
  → Agent Processes Intent
  → Memory Persistence (Firestore/Supabase/JSON)
  → GPT-4 API Call
  → Observability & Metrics
  → Hosting Abstraction Layer
  → Response to User
```


## Extensibility and Deployment
- Add new agents by simply placing their folders under `modules/` with the required interface.
- Extend deployment targets via `zsi_config.yaml` without code changes (supports Railway, AWS Lambda, Azure Functions, Google Cloud Run, Docker, etc.).
- Modular design supports easy updates and scaling in production environments.

### Future-Proofing Hooks
- **Plugin Registry**: Extend `core/orchestrator.py` to load custom plugins or middleware via `plugins/` directory.
- **Plugins Directory** (`plugins/`): drop in custom middleware, route interceptors, or extension points.
- **Observability** (`observability/`): include OpenTelemetry initializers, Prometheus exporters, and structured logging setups.
- **Security Layers**: Add OAuth2/JWT flows or mTLS configurations under `security/` for enterprise-grade authentication.

### Configuration Examples

```yaml
# zsi_config.development.yaml
ENV: development
DATABASE_URL: postgres://localhost:5432/dev_db
FEATURE_FLAGS:
  enableBetaFeature: true
LOG_LEVEL: DEBUG
DEPLOY_TARGET: railway
```
```yaml
# zsi_config.production.yaml
ENV: production
DATABASE_URL: postgres://prod_host:5432/prod_db
FEATURE_FLAGS:
  enableBetaFeature: false
LOG_LEVEL: INFO
DEPLOY_TARGET: aws_lambda
```
