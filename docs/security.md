# Security

---

This section covers authentication, authorization, data protection, and security best practices for a production-grade, future-proof Copilot framework.

---

## Configuration & Environment Management
All configuration—endpoint URLs, credentials, and platform-specific settings—is injected via environment variables for consistency and security.
We support multiple hosting tiers (Railway, AWS Lambda/API Gateway, Docker Swarm, Kubernetes, on‑prem), controlled by:
- `ENV` (e.g., production, staging, local)
- `HOST_PROVIDER` (e.g., railway, aws_lambda, docker, k8s, on_prem)
- Platform-specific variables:
  - `RAILWAY_STATIC_URL`, `RAILWAY_API_TOKEN`
  - `AWS_LAMBDA_ARN`, `AWS_API_GATEWAY_URL`
  - `DOCKER_REGISTRY_URL`, `K8S_CLUSTER_NAME`

A unified `core/config.py` reads these variables and exposes strongly‑typed settings for the entire application.

### Example: core/config.py

```python
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    ENV: str                           # e.g., "production", "staging", "local"
    HOST_PROVIDER: str                 # e.g., "railway", "aws_lambda", "docker", "k8s", "on_prem"
    JWT_SECRET_KEY: str                # for JWT signing
    X_API_KEY: str                     # API key for header auth
    API_KEY_HEADER_NAME: str           # Header name for API key authentication
    OAUTH2_TOKEN_URL: str = None       # URL for OAuth2 token endpoint
    LOG_LEVEL: str = "INFO"            # Application log level
    CORS_ORIGINS: List[str] = []       # Allowed CORS origins

    # Platform-specific
    RAILWAY_STATIC_URL: str = None
    RAILWAY_API_TOKEN: str = None
    AWS_LAMBDA_ARN: str = None
    AWS_API_GATEWAY_URL: str = None
    DOCKER_REGISTRY_URL: str = None
    K8S_CLUSTER_NAME: str = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate and use throughout the app:
settings = Settings()
```

---

## Authentication
### Authentication & API Security
All routes are protected via header-based API keys or OAuth2. Configuration is driven by environment settings.

```python
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from starlette.status import HTTP_401_UNAUTHORIZED
from core.config import settings

app = FastAPI(
    title="Copilot Framework API",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "agents", "description": "Agent operations"},
    ],
)

# API Key header
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != settings.X_API_KEY:
        raise HTTPException(HTTP_401_UNAUTHORIZED, "Invalid or missing API Key")
    return api_key

# OAuth2 (JWT) setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=settings.OAUTH2_TOKEN_URL)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # decode and verify JWT here, raise HTTPException if invalid
    return decode_jwt(token, settings.JWT_SECRET_KEY)

# Apply global dependencies
app.dependencies = [Depends(get_api_key)]
```

---

## Data Encryption
- At rest: use encrypted storage (Firestore encryption, AWS KMS).
- In transit: HTTPS/TLS (TLS v1.2+); optionally mutual TLS for service-to-service encryption.
- **Database Encryption**: Use transparent data encryption (TDE) where supported.
- **HTTP Security Headers**: Enforce HSTS (`Strict-Transport-Security`), Content Security Policy, X-Frame-Options, etc.

---

## Access Control

Access control is enforced via RBAC and principle of least privilege.

- **Role-Based Access Control (RBAC):**  
  Define and enforce roles (e.g., `admin`, `user`, `system`) via environment-configured settings or a centralized permission service. In `core/config.py`, load allowed roles from environment:
  ```python
  from pydantic import BaseSettings
  class Settings(BaseSettings):
      allowed_roles: dict = {"admin": ["*"], "user": ["read", "write"], "system": ["manage"]}
      class Config:
          env_file = ".env"
  settings = Settings()
  ```
  Then, in `core/security.py`, implement a FastAPI dependency:
  ```python
  from fastapi import HTTPException, Security
  from fastapi.security.api_key import APIKeyHeader
  from starlette.status import HTTP_403_FORBIDDEN

  api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

  async def get_current_role(api_key: str = Security(api_key_header)):
      # lookup role by API key (e.g., from DB or cache)
      role = lookup_role_from_key(api_key)
      if role not in settings.allowed_roles:
          raise HTTPException(HTTP_403_FORBIDDEN, "Invalid role")
      return role

  def require_permissions(action: str):
      async def permission_dependency(role: str = Security(get_current_role)):
          perms = settings.allowed_roles.get(role, [])
          if "*" not in perms and action not in perms:
              raise HTTPException(HTTP_403_FORBIDDEN, "Permission denied")
      return permission_dependency
  ```
  Finally, enforce per-route:
  ```python
  from fastapi import Depends
  @app.post("/admin/task", dependencies=[Depends(require_permissions("manage"))])
  async def admin_task():
      ...
  ```
- **Environment Variables & Secrets Management:**  
  Store all sensitive configuration (API keys, database URLs, OAuth client secrets, AWS credentials) in environment variables. Never commit secrets to source control. Example `.env` entries:  
  ```env
  # General
  ENV=production
  HOST_PROVIDER=aws_lambda

  # Database
  DATABASE_URL=postgresql://user:pass@host:5432/db

  # AWS
  AWS_REGION=us-east-1
  AWS_LAMBDA_ARN=arn:aws:lambda:us-east-1:123456789012:function:MyFunction
  AWS_API_GATEWAY_URL=https://api.myapp.com

  # Railway
  RAILWAY_STATIC_URL=https://trainy.app
  RAILWAY_API_TOKEN=railway-xyz

  # Security
  X_API_KEY=your-api-key
  OAUTH_CLIENT_ID=...
  OAUTH_CLIENT_SECRET=...
  JWT_SECRET_KEY=...
  ```
- **Input Validation & Sanitization:**  
  Validate and sanitize all incoming data using Pydantic models. Reject or sanitize any unexpected fields.
- **Principle of Least Privilege:**  
  Grant only the minimal permissions required for each service or user role. For AWS, attach least-privilege IAM policies. For databases, use role-specific accounts with limited access.
- **Audit & Logging:**  
  Implement audit logging for sensitive operations (e.g., role changes, user data export). Log access attempts, successes, and failures to a secure, append-only log store.
- **Future-Proofing for Multiple Hosts:**  
  Ensure that access control and secret-loading logic works identically across all target deployments (Railway, AWS Lambda/API Gateway, Docker, on-prem). Abstract environment-specific details into a common configuration layer in `core/config.py`.

## Observability & Incident Response
- Integrate structured logging (JSON) and distributed tracing (OpenTelemetry).
- Configure alerts for anomalous patterns or repeated failures.
- Maintain runbooks for incident triage and response.

### Structured Logging Example
```python
import logging, sys
logger = logging.getLogger("copilot")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","message":%(message)s}')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

- **Future Extensions**
  - Plugin in CI/CD secret scanning (e.g., GitHub Actions secret review).
  - Integrate WAF and API Gateway threat detection per provider.
  - Adopt fine‑grained IAM via OPA or external policy engine.

## Next Steps
1. Integrate OAuth2 password flow for user authentication.
2. Hook up OpenTelemetry tracing in `main.py`.
3. Add CI/CD secret scanning to pipeline.

---

### CORS Configuration
Configure allowed origins via `settings.CORS_ORIGINS`. Example integration:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
<!-- End of Security Guide -->