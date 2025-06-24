# Quickstart Guide

Get up and running with the ZSI Copilot Framework in minutes.

## Prerequisites

- Python 3.9+ installed  
- Git installed  
- [Poetry](https://python-poetry.org/) or `pipenv` for dependency management  
- Docker (optional, for containerized dev)

## Clone the Repository

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

## Create and Activate Virtual Environment

```bash
poetry install   # or pipenv install
poetry shell     # or pipenv shell
```

## Configure Environment

Copy the example env file and update values in `.env`:

```bash
cp .env.example .env
```

Edit `.env` and set:

```dotenv
# FastAPI
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# Database or Memory
DATABASE_URL=sqlite:///./data.db  # or postgres://... for scale dev
AWS_REGION=us-east-1              # for AWS Lambda / DynamoDB

# Security
API_KEY=<your_api_key>            # required for all /invoke requests

# Deployment
RAILWAY_PROJECT=<railway_project_id>
AWS_LAMBDA_FUNCTION=<lambda_name>
```

## Run the Application Locally

```bash
# Start FastAPI server with hot reload
uvicorn core.orchestrator:app \
  --host "${FASTAPI_HOST}" \
  --port "${FASTAPI_PORT}" \
  --reload
```

Test the generic `/invoke` endpoint:

```bash
curl -X POST http://localhost:${FASTAPI_PORT}/invoke \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "user_id": "demo_user",
    "business_type": "example",
    "intent": "test_intent",
    "data": {"foo": "bar"}
}'
```

## Serve Documentation

```bash
# Live MkDocs site
mkdocs serve
```

View interactive docs at `http://127.0.0.1:8000/docs` (FastAPI) and `http://127.0.0.1:8000/docs/` (MkDocs).

## Deployment Tiers

### 1. MVP (Railway)

- Zero-config JSON memory backend.  
- Push repo, set ENV vars in Railway dashboard.

### 2. Scale Dev (Docker + Cloud DB)

```bash
docker build -t zsi-copilot-framework .
docker run --env-file .env -p ${FASTAPI_PORT}:${FASTAPI_PORT} zsi-copilot-framework
```

Use managed PostgreSQL, Supabase, or DynamoDB.

### 3. Production (AWS Lambda + API Gateway)

- Package with AWS SAM or Serverless Framework.  
- Set `AWS_LAMBDA_FUNCTION` and `AWS_REGION` in `.env`.  
- Enable DynamoDB memory backend via `CORE_MEMORY_BACKEND=dynamo`.

## Next Steps

1. Review [Architecture](architecture.md).  
2. Add new agents under `modules/<agent_name>/agent.py`.  
3. Check [Security](security.md) for best practices.  
4. Explore [Persistence & Memory](persistence.md) for customization.

## Further Reading

- [Getting Started](getting-started.md)  
- [Agents](agents.md)  
- [Flows](flows.md)  
- [Schemas](schemas.md)  
