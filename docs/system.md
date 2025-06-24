## memory_layer.md
---
version: "1.3"
last_updated: "2025-05-30"
---
Describes the persistent user context storage and retrieval API used by all agents.

- **User State Model:** Holds structured data (e.g., metrics, tags, preferences) in a single JSON or DB record.
- **Trend Tracker:** Aggregates and analyzes historical data for pattern detection.
- **Agent Contracts:** Clearly defined read/write schemas for each agent to prevent schema drift.
- **Persistence Backends:** Pluggable adapters for:
  - **JSON Files** (local, MVP)
  - **Firestore** (GCP)
  - **DynamoDB** (AWS)
  - **PostgreSQL** (self-hosted or managed)
- **Configuration Overrides:** All memory-layer paths and connection settings are driven by:
  - ENV var `MEMORY_CONFIG_PATH` for config file
  - `MEMORY_LAYER_PATH` key in `zsi_config.yaml`
- **Automatic Migrations:** Skips or applies schema changes based on `ZSI_ENV` (dev, staging, prod).

See [docs/persistence.md](../docs/persistence.md) for full details on schema, migrations, and backend adapters.

---

## 4. Environment & Deployment Overview  
This section outlines how to configure, containerize, and deploy your ZSI framework across environments.

**4.1 Config Overrides**  
All paths and settings in this document can be overridden via **environment variables** or the central **`zsi_config.yaml`**. The orchestrator reads these at startup.

```yaml
# zsi_config.yaml
env: production             # ZSI_ENV (development, staging, production)
db_provider: dynamodb       # DB_PROVIDER (json, firestore, dynamodb, postgres)
llm_provider: openai        # LLM_PROVIDER (openai, azure, local)
orchestrator_host: ${ORCHESTRATOR_HOST}  # Base API URL
log_level: info             # LOG_LEVEL (debug, info, warn, error)
system_prompt_path: ${SYSTEM_PROMPT_PATH}
agent_routing_path: ${AGENT_ROUTING_CONFIG}
memory_layer_path: ${MEMORY_CONFIG_PATH}
dockerfile_path: ${DOCKERFILE_PATH}
```
# Note: You can also specify `JWT_PROVIDER` and `AUTH_CONFIG_PATH` here for token validation settings.

**4.2 Containerization**  
- Include a `Dockerfile` at the project root to build a container image for the orchestrator.  
- The Dockerfile path can be overridden via the `DOCKERFILE_PATH` environment variable for custom container setups.
- Support for building with FastAPI base images (Railway) and packaging for AWS Lambda via AWS SAM/Serverless Framework is available out of the box.
- Use multi-stage builds to isolate dependencies and minimize image size.  
- Tag images with git commit SHA and semantic version.

**4.3 CI/CD Workflow**  
- Configure a pipeline (e.g., GitHub Actions, GitLab CI) to:
  1. Install dependencies, run linters and tests.
  2. Build and push Docker images on merges to `main`.
  3. Deploy to your chosen platform (Railway, Render, AWS ECS, or AWS Lambda via Serverless Framework).

**4.4 Observability & Security**  

### 4.4.1 Security & Observability Hooks
- FastAPI middleware for request logging, metrics (Prometheus), and API key enforcement is preconfigured.

- Integrate logging and metrics (CloudWatch, Datadog, Prometheus).  
- Enforce API key or OAuth2/JWT security on all endpoints.  
- Rotate secrets and use a vault (e.g., AWS Secrets Manager, HashiCorp Vault).

### 4.4.2 OAuth2 & JWT Support
- **OAuth2 Flows:** Prebuilt FastAPI dependencies for Authorization Code, Client Credentials, and Password Grant flows.
- **JWT Validation:** Middleware to verify JWT signatures, claims, and expiration, pluggable via `JWT_PROVIDER` env var.
- **Token Rotation & Revocation:** Hooks for rotating refresh tokens and revoking tokens using Redis or your chosen store.
- **Config Override:** Control OAuth2 and JWT settings via `AUTH_CONFIG_PATH` in `zsi_config.yaml`.
- **JWT Middleware Configuration:** Selectable via `JWT_PROVIDER` and `AUTH_CONFIG_PATH`, with support for JWK discovery and claim validation.

# System Prompt & Routing

This document defines the core orchestration logic for your ZSI-driven Copilot framework: the system prompt fed to the LLM, how intents are routed to agents, and how the memory layer integrates.

---

## 1. system_prompt.md  
**Location:** `/system/system_prompt.md` (override via `SYSTEM_PROMPT_PATH` or `zsi_config.yaml`)  
Defines the master LLM prompt used on every request. It includes:  
- **Role and Scope:** “You are DietPilot, a voice-first macro assistant…”  
- **Agent Triggers:** Maps user patterns to agent modules  
- **Behavior Rules:** Tone, style, and one-insight-per-turn guideline  
- **Examples:** Sample user–assistant interactions  
- **Source Override:** Can supply a custom prompt file path via the `SYSTEM_PROMPT_PATH` env var or `zsi_config.yaml` key `system_prompt_path`.

---

## 2. agent_routing_logic.md  
**Location:** `/system/agent_routing_logic.md` (override via `AGENT_ROUTING_CONFIG` or `zsi_config.yaml`)  
Describes how parsed intents map to specific agents via the FastAPI orchestrator (`/log` endpoint). Example routing table:

| Intent Pattern              | Agent(s)             |
|-----------------------------|----------------------|
| Meal description (“I ate…”) | Macronator           |
| Steps/Workout (“I ran…”)    | Captain Stepsalot    |
| Supplements (“I took…”)     | SuppBro              |
| Mood (“I feel…”)            | Moody Judy           |
| Repeat Meal (“Same as…”)    | Chef Déjà Vu         |
| Shopping (“Add to list…”)   | ShopBot McListy      |
| Trend Summary (“Today’s summary”) | Brainy Susan  |

Includes flow notes for chaining, fallback, and escalation logic.

- **Config Override:** Custom routing rules can be loaded via the `AGENT_ROUTING_CONFIG` env var or `zsi_config.yaml` key `agent_routing_path`.

---

## 3. memory_layer.md  
**Location:** `/system/memory_layer.md` (override via `MEMORY_CONFIG_PATH` or `zsi_config.yaml`)  
Explains persistent user context storage and access patterns:  
- **User State Object:** Macro totals, steps, supplements, mood, tags  
- **Trend Tracker:** Weekly patterns and anomaly detection by Brainy Susan  
- **Agent Contracts:** What each agent reads and writes  
- **Persistence:** JSON file for MVP or pluggable DB (Firestore, DynamoDB)  
- **Persistence Override:** Custom memory layer config via `MEMORY_CONFIG_PATH` env var or `zsi_config.yaml` key `memory_layer_path`.

---

## 4. Future Hosting & Extension

### 4.1 Environment-Driven Configuration & Overrides

Configuration is driven by `zsi_config.yaml` or environment variables. Override defaults via ENV.

Your orchestrator reads settings from environment variables or `zsi_config.yaml` at startup:

| Variable            | Description                                          | Example               |
|---------------------|------------------------------------------------------|-----------------------|
| ZSI_ENV             | Deployment environment (`development`, `staging`, `production`) | production    |
| DB_PROVIDER         | Memory backend (`json`, `firestore`, `dynamodb`, `postgres`)    | dynamodb      |
| LLM_PROVIDER        | LLM service (`openai`, `azure`, `local`)             | openai               |
| ORCHESTRATOR_HOST   | Base URL for the FastAPI service                     | https://api.myapp.com |
| LOG_LEVEL           | Logging verbosity (`debug`, `info`, `warn`, `error`) | info                 |

```yaml
# zsi_config.yaml
env: production
db_provider: dynamodb
llm_provider: openai
orchestrator_host: https://api.myapp.com
log_level: info
```

### 4.2 Containerization & CI/CD
- Include a `Dockerfile` to build and package the orchestrator.
- Use a CI pipeline (GitHub Actions, GitLab CI) to:
  1. Lint and test on each PR.
  2. Build and publish Docker image on merge to `main`.
  3. Deploy automatically via CI to ECS, Lambda, Railway, or Render as configured.

### 4.3 Monitoring & Alerting
- Integrate with your observability stack (CloudWatch, Datadog, Prometheus, or Railway metrics).
- Track key metrics: uptime, request latency, error rates, cost monitoring, and LLM call cost.
- Set alerts on error spikes or SLA breaches.

This environment-driven, containerized approach ensures you can switch or add platforms (Railway, Render, AWS, GCP) without code changes.

This document serves as your guide to a fully modular, environment-driven ZSI deployment.

> _Note: This document is maintained in sync with your `zsi_config.yaml` schema. Any new config keys should be reflected here for visibility._

## Appendix: Key Environment Variables
| Variable            | Description                                                   |
|---------------------|---------------------------------------------------------------|
| ZSI_ENV             | Deployment environment (`development`, `staging`, `production`) |
| DB_PROVIDER         | Memory backend (`json`, `firestore`, `dynamodb`, `postgres`)    |
| LLM_PROVIDER        | LLM service (`openai`, `azure`, `local`)                        |
| ORCHESTRATOR_HOST   | Base URL for the FastAPI service                               |
| LOG_LEVEL           | Logging verbosity (`debug`, `info`, `warn`, `error`)           |
| AUTH_CONFIG_PATH    | Path to OAuth2/JWT configuration overrides                    |
| JWT_PROVIDER        | JWT validation backend (`jwks`, `oauth2_server`, `custom`)    |