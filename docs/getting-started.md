[![Build Status](https://img.shields.io/github/actions/workflow/status/your-org/your-repo/ci.yml)](https://github.com/your-org/your-repo/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/your-org/your-repo)](https://codecov.io/gh/your-org/your-repo)

# Getting Started

## Prerequisites
- Python 3.8+.
- Poetry or pipenv.
- Docker (optional).
- Access to OpenAI GPT-4 API.
- MkDocs (mkdocs-material) & mkdocs-techdocs-core plugin.
- Node.js (for MkDocs).
- Environment variables (e.g., `.env` file).

## Installation
```bash
git clone <repo-url>
cd your-project
# If using Poetry:
poetry install
# Or with pip:
pip install -r requirements.txt

# Copy example environment variables
cp .env.example .env
# Edit .env to include your OPENAI_API_KEY and other settings

# (Optional) Start the FastAPI server for local development
uvicorn main:app --reload --env-file .env

# (Optional) Serve documentation with MkDocs
mkdocs serve

# Deploy to Railway
railway up

# Deploy to Vercel
vercel --prod
```

## Configure Environment Variables
Rename and edit your `.env` file to include the following keys:

- `OPENAI_API_KEY` — Your OpenAI API key.
- `FASTAPI_HOST` — Hostname or IP for FastAPI (default: `0.0.0.0`).
- `FASTAPI_PORT` — Port number for FastAPI (default: `8000`).
- `RAILWAY_API_URL` — (Optional) Railway deployment URL.
- `AWS_LAMBDA_STAGE` — (Optional) AWS Lambda stage name for production.
- `DYNAMO_TABLE_NAME` — (Optional) DynamoDB table for memory persistence.
- `LOG_LEVEL` — (Optional) Logging verbosity (e.g. `DEBUG`, `INFO`).

## Project Structure
The project is organized into the following directories and files:
- `main.py` — FastAPI entrypoint.
- `core/` — orchestrator, memory, agent router.
- `modules/` — ZSI agent modules (zbot, zse, zbar).
- `schema/` — Pydantic models and JSON schemas.
- `data/` — persistent user context storage.
- `docs/` — MkDocs site sources.
- `.env.example` — sample environment variables, customize for each environment (RAILWAY_API_URL, AWS_LAMBDA_STAGE, etc.).
- `mkdocs.yml` — documentation site configuration.

_For security recommendations, see the [Security](security.md) guide._
