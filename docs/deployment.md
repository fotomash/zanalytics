# Deployment Guide (CTO-Approved)

This document provides a production-ready, scalable, and secure deployment strategy for the Copilot Framework, from local development through enterprise-grade architectures.

---

## 1. Environment & Configuration

### 1.1. Environment Variables
Define all sensitive configuration in environment variables, not in code. Example `.env`:

```dotenv
# Core
APP_ENV=development               # development | staging | production
API_KEY=your_global_api_key
SECRET_KEY=your_jwt_signing_key

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Storage
STORAGE_PROVIDER=s3               # s3 | gcs | azure
STORAGE_BUCKET=your-bucket-name

# Observability
LOG_LEVEL=INFO
SENTRY_DSN=your_sentry_dsn       # optional
```

Load settings with Pydantic in `core/config.py`:

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_env: str
    api_key: str
    secret_key: str
    database_url: str
    storage_provider: str
    storage_bucket: str
    log_level: str
    sentry_dsn: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()
```

### 1.2. Directory Structure
```
project/
├── core/
│   ├── config.py
│   ├── orchestrator.py
│   └── security.py
├── modules/
│   └── <agent_name>/
│       └── agent.py
├── schema/
│   └── models.py
├── docs/
│   ├── index.md
│   └── deployment.md
├── mkdocs.yml
├── .env
└── requirements.txt
```

---

## 2. Hosting Tiers

| Tier         | Stack                                         | Deployment Commands                                           |
| ------------ | --------------------------------------------- | ------------------------------------------------------------- |
| **Dev**      | FastAPI + SQLite (local) + Railway            | `uvicorn core.orchestrator:app --reload`<br>`railway up`      |
| **Staging**  | FastAPI + PostgreSQL + Docker + GitHub Actions| `docker build -t copilot-staging .`<br>`docker run ...`       |
| **Prod**     | AWS Lambda + API Gateway + DynamoDB + S3      | `sam build`<br>`sam deploy --stack-name copilot-prod ...`     |
| **Enterprise**| Kubernetes (EKS) + RDS + Redis + Prometheus | Helm charts or Argo CD for multi-region resilience            |

Each tier reuses the same codebase with environment-specific configuration.

---

## 3. CI/CD Pipeline

Use GitHub Actions or equivalent:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: flake8 . && mypy .
      - run: pytest --cov

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Environment
        run: |
          if [[ "${GITHUB_REF}" == "refs/heads/main" ]]; then
            railway up --production
          fi
          if [[ "${GITHUB_REF}" =~ ^refs/tags\/v ]]; then
            sam deploy --stack-name copilot-prod --capabilities CAPABILITY_IAM
          fi
```

---

## 4. Security & Compliance

- **Authentication**: Enforce `X-API-KEY` header via FastAPI dependency in `core/security.py`.  
- **Authorization**: Integrate OAuth2 / JWT flows for fine-grained access control.  
- **Encryption**: SSL/TLS everywhere; encrypt data at rest (DynamoDB, S3).  
- **Secrets Management**: Use AWS Secrets Manager or GitHub Secrets for production.  
- **Dependency Scanning**: Integrate Dependabot and Snyk for regular vulnerability checks.

---

## 5. Observability & Monitoring

- **Logging**: Structured JSON logs using `python-json-logger`, shipped to CloudWatch or ELK.  
- **Metrics**: Use Prometheus client library; expose `/metrics` endpoint.  
- **Tracing**: Instrument with OpenTelemetry and export to a tracing backend (Jaeger, X-Ray).  
- **Alerting**: Set up CloudWatch alarms or PagerDuty integration for error rate and latency spikes.

---

## 6. Future-Proofing & Expansion

- **Plugin Architecture**: Drop new agents into `modules/<agent_name>/agent.py`; they auto-register.  
- **Feature Flags**: Integrate LaunchDarkly or a DIY flag system in `core/flags.py`.  
- **Event-Driven**: Add Kafka/RabbitMQ adapters in `core/events.py` for asynchronous processing.  
- **Multi-Tenancy**: Namespace data by `tenant_id` in both memory and storage layers.  
- **GraphQL API**: Optionally expose a GraphQL layer via `Ariadne` or `Strawberry` for client flexibility.

---

*End of Deployment Guide*