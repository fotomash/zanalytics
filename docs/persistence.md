# Configuration Reference

We centralize all environment variable defaults and overrides in `configuration.md`. The orchestrator and persistence layer load settings via a Pydantic `Settings` class.

```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    persistence_backend: str = "json"
    persistence_json_path: str = "./data/{user_id}/{date}/user_context.json"
    firestore_project_id: str = None
    firestore_credentials: str = None
    supabase_url: str = None
    supabase_key: str = None
    dynamodb_table_name: str = None
    aws_region: str = None
    deployment_env: str = "local"  # e.g., local, railway, lambda, docker
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000

    class Config:
        env_file = ".env"
```

# Persistence

This document describes memory storage options and best practices for managing user context data.

## Local JSON
Ideal for MVP and local testing.

- **Storage**: Simple file-based JSON storage with the **default path: `./data/{user_id}/{date}/user_context.json`**.
- **Configuration**: Override the file path by setting the `PERSISTENCE_JSON_PATH` environment variable.
- **Pros**: Zero external dependencies; easy to inspect, version-control, and debug.
- **Cons**: Single-user; no concurrency or real-time updates.

## Environment Configuration

All environment variable defaults and overrides are managed in your deployment configuration (e.g. `.env`, Kubernetes Secrets, or CI/CD environment settings). See [Configuration Reference](configuration.md) for details.

*Pro Tip:* You can template `user_id` and `date` in your `.env` or runtime configuration to automatically partition memory by user and day.

### .env Example (Local Development)

```dotenv
# .env
# Note: Never commit this file to version control as it contains sensitive credentials.
PERSISTENCE_BACKEND=firestore
FIRESTORE_PROJECT_ID=my-gcp-project
FIRESTORE_CREDENTIALS=path/to/key.json
```

In production, manage these variables via your platform's secrets management (e.g., AWS Secrets Manager, Railway Variables, Kubernetes Secrets).

## FastAPI Orchestrator & Hosting Configuration

| Variable          | Purpose                                         | Default / Example   |
|-------------------|------------------------------------------------|---------------------|
| `FASTAPI_HOST`    | Host address for the FastAPI server             | `0.0.0.0`           |
| `FASTAPI_PORT`    | Port number for the FastAPI server              | `8000`              |
| `RAILWAY_ENV`     | Set to `true` when deployed on Railway to enable environment-specific adjustments |                     |
| `AWS_LAMBDA`      | Set to `true` when running under AWS Lambda to optimize runtime behavior |                     |
| `DEPLOYMENT_ENV`  | Explicit deployment target identifier (`railway`, `lambda`, `docker`, etc.) | Enables environment-specific config hooks  |

These flags allow the orchestrator to detect the runtime environment and adjust logging, CORS policies, and scheduling behavior accordingly.

---

## Managed JSON Stores (Firestore / Supabase)
Plug-and-play managed JSON storage.

- **Firestore**: Set `PERSISTENCE_BACKEND=firestore`, `FIRESTORE_PROJECT_ID`, and `FIRESTORE_CREDENTIALS`.
- **Supabase**: Set `PERSISTENCE_BACKEND=supabase`, `SUPABASE_URL`, and `SUPABASE_KEY`.
- **Pros**: Scalable, multi-user, real-time updates.
- **Cons**: Requires managed service; incur usage costs.

## AWS DynamoDB
High availability, autoscaling key-value store.

- **Configuration**: Set `PERSISTENCE_BACKEND=dynamodb`, then provide: `DYNAMODB_TABLE_NAME`, `AWS_REGION`, and AWS credentials via `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (or IAM role).
- **Pros**: Enterprise-grade scalability, fine-grained control.
- **Cons**: More setup complexity; potential cold-start latency in serverless.

---

**Note:** Always ensure secrets and sensitive credentials are managed securely. Refer to the [security.md](security.md) document for best practices on secrets management and environment variable handling.

## Future Extensions

As your Copilot framework evolves, consider:
- **Custom Backends**: Implement additional persistence drivers (e.g., SQL, Redis).
- **Dynamic Routing**: Use `DEPLOYMENT_ENV` to switch CORS, logging, and observability settings.
- **Feature Flags**: Integrate a feature-flag service (e.g., LaunchDarkly) to toggle new capabilities at runtime.
- **Distributed Memory**: Coordinate user context across microservices via a shared store or event bus.
- **Observability Hooks**: Provide native support for logging frameworks (e.g., Structlog, Loguru) and metrics (e.g., Prometheus).
