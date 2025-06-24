# Deploying Your Copilot Framework

A comprehensive, production-grade deployment and operations guide for your modular Copilot framework, with security, observability, and flexible environment configurations baked in from day one.

---

## 1. Deployment Environments & Tiers (Dev → Prod)

Define clear environment tiers for development, staging, and production to enforce consistent practices, security posture, and observability requirements across the software lifecycle.

### 1.1 Development
- **Platform:** Railway (or local Docker Compose)
- **Hosting:** FastAPI dev server
- **Runtime:** FastAPI dev server
- **Persistence:** Local JSON / SQLite
- **Configuration:** `.env` file (Railway Envs)
- **CORS Policy:** Open for local testing
- **Logging Level:** DEBUG
- **Observability:** None by default
- **CI/CD Trigger:** Manual or local

### 1.2 Staging
- **Platform:** Docker on VPS or Kubernetes namespace
- **Hosting:** Kubernetes or Docker host
- **Runtime:** Uvicorn + Gunicorn
- **Persistence:** Supabase / Firebase
- **Configuration:** Managed via Vault / Secrets Manager
- **CORS Policy:** Restricted to staging domains
- **Logging Level:** INFO
- **Observability:** Prometheus + Grafana
- **CI/CD Trigger:** Automated via CI pipeline

### 1.3 Production
- **Platform:** AWS Lambda + API Gateway (or alternative serverless)
- **Hosting:** Serverless environment
- **Runtime:** `Mangum` adapter for FastAPI
- **Persistence:** DynamoDB / RDS
- **Configuration:** AWS Systems Manager Parameter Store / Secrets Manager
- **CORS Policy:** Strictly scoped
- **Logging Level:** WARN / ERROR
- **Observability:** CloudWatch + X-Ray tracing
- **CI/CD Trigger:** GitHub Actions / GitLab CI → Docker image → ECR → Lambda

### 1.4 Modular API Definitions
- **OpenAPI Specs:** Maintain `[openapi.yaml](./openapi.yaml)` under `/api` for machine-readable API contracts.
- **Router Modules:** Place FastAPI routers in `routers/` with naming convention `v{n}/[service]_router.py`.
- **Security Tags:** Annotate each router with tags and dependencies for API key or JWT enforcement.
- **Definition Files:** Store route and security definitions in `[openapi.yaml](./openapi.yaml)` or `[api/*.yaml](./api/*.yaml)`
- **Import:** Use FastAPI’s `app.include_router()` to mount modular routers.
- **Versioning:** Namespace routes by version (e.g., `/v1`, `/v2`) for backward compatibility.

---

## 2. Environment Configuration

All environment-specific settings are managed via environment variables or external secret stores. Use the following conventions:

### 2.1 Local Development (.env)
```env
FASTAPI_HOST=${FASTAPI_HOST:-0.0.0.0}
FASTAPI_PORT=${FASTAPI_PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-debug}
MEMORY_STORE=${MEMORY_STORE:-local}
RAILWAY_STATIC_URL=${RAILWAY_STATIC_URL}
OPENAI_API_KEY=${OPENAI_API_KEY}
JWT_SECRET=${JWT_SECRET:-your-local-jwt-secret}
OAUTH2_CLIENT_ID=${OAUTH2_CLIENT_ID:-}
OAUTH2_CLIENT_SECRET=${OAUTH2_CLIENT_SECRET:-}
```

### 2.2 Secrets Management
| Variable            | Purpose                              | Secret Store               |
|---------------------|------------------------------------|----------------------------|
| FASTAPI_HOST        | Host binding for FastAPI            | SSM Parameter Store / Vault|
| FASTAPI_PORT        | Port binding                      | SSM / Vault                |
| LOG_LEVEL           | Logging verbosity (DEBUG/INFO/WARN/ERROR) | SSM / Vault          |
| MEMORY_STORE        | `local`, `sqlite`, `supabase`, `dynamo` | SSM / Vault              |
| OPENAI_API_KEY      | OpenAI API key                    | Secrets Manager / Vault    |
| JWT_SECRET          | HS256 signing key for JWT tokens  | Secrets Manager / Vault    |
| OAUTH2_CLIENT_ID/SECRET | OAuth2 client credentials for external flows | Secrets Manager / Vault |
| CORS_ORIGINS        | Comma-separated allowed origins   | SSM / Vault                |

### 2.3 Platform Overrides
```env
PLATFORM=${PLATFORM:-dev}             # dev, staging, prod
HOST=${FASTAPI_HOST:-0.0.0.0}
PORT=${FASTAPI_PORT:-8000}
MEMORY_STORE=${MEMORY_STORE:-local}
```

Reference these values in your deployment scripts or CI/CD workflows for conditional branching.

---

## 3. CI/CD Pipeline

1. **Security Scans**
   - Snyk / Trivy container and dependency scans
2. **Build & Test**
   - Lint: `flake8`
   - Format: `black`
   - Unit tests: `pytest`
3. **Infrastructure as Code**
   - Terraform fmt and validate
4. **Release**
   - Build Docker image
   - Push to registry (DockerHub/ECR)
   - Deploy to target environment:
     ```bash
     if [ "$PLATFORM" = "aws_lambda" ]; then
       # Deploy via Terraform to AWS
       terraform apply -var="env=$PLATFORM"
     else
       # Deploy via Kubernetes/Terraform
       kubectl apply -f k8s/
     fi
     ```
5. **Post-Deploy Smoke Test**
   ```bash
   curl -H "x-api-key: $API_KEY" https://api.example.com/health
   ```

---

## 4. Security & Secrets

- **API Key Enforcement** at gateway level
- **OAuth2/JWT** integration hooks (see `[security.md](./security.md)`)
- **Rate limiting** via API Gateway or Nginx
- **TLS** enforced end-to-end

> **Best Practice:** Rotate all secrets quarterly and record versions in audit logs.

---

## 5. Observability & Monitoring

- **Logging:** Structured JSON logs integrated into ELK or CloudWatch Logs pipelines.
- **Metrics:** Prometheus exporters via FastAPI middleware for real-time monitoring.
- **Tracing:** OpenTelemetry with Jaeger or AWS X-Ray for distributed tracing.
- **Alerts:** PagerDuty or SNS configured for critical error notifications.

---

## 6. Scalability & Reliability Enhancements

- Implement multi-region failover via Route 53
- Automate canary and blue/green deployments with feature flags
- Configure autoscaling rules based on metrics
- Integrate advanced caching strategies using Redis and CDNs

---

## 7. OAuth2 & JWT Integration

This framework supports pluggable authentication schemes:

1. **API Key**  
   - Enforced via `x-api-key` header.
   - Configure valid keys in secret store and validate in FastAPI middleware.

2. **OAuth2 (Authorization Code Flow)**  
   - Plug in your provider via `OAUTH2_CLIENT_ID` and `OAUTH2_CLIENT_SECRET`.
   - Use FastAPI’s `OAuth2AuthorizationCodeBearer` for token retrieval.

3. **JWT Bearer**  
   - Issue tokens signed with `JWT_SECRET`.
   - Validate via FastAPI’s `BearerToken` dependency.
   - Extend claims schema in `schema/auth.py`.

For examples, see `[security.md](./security.md)` under “OAuth2 Examples” and “JWT Middleware”.

---

## 9. Future-Proofing Checklist
- [ ] Centralized secret management already in place
- [ ] API versioning enforced via routing namespace
- [ ] Observability pipeline automated in CI/CD
- [ ] Modular service definitions (routers, handlers) documented
- [ ] Platform overrides tested in each environment

---

## 8. Appendix
- **Directory Layout**: Reference to repository structure (`/api`, `/routers`, `/core`, `/modules`, `/docs`).
- **Key Dependencies**:
  - FastAPI >=0.85,<1.0
  - Pydantic >=1.9,<2.0
  - Mangum >=0.13,<1.0
  - boto3 >=1.21,<2.0
  - OpenTelemetry >=1.14,<2.0
