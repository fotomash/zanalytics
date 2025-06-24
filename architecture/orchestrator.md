# Orchestrator

```python
# Supported Deployment Environments
#   - FastAPI on Railway: set ENV=development or production, RAILWAY_URL, DATABASE_URL
#   - AWS Lambda: AWS_LAMBDA_HANDLER, AWS_REGION, DYNAMODB_TABLE
#   - Docker/Kubernetes: CONTAINER_ENV, SERVICE_NAME, K8S_NAMESPACE

import importlib
import pkgutil
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_403_FORBIDDEN
from user_memory import load_user_memory, save_user_memory
from datetime import datetime

# Load environment
load_dotenv()

# Deployment environment
ENV = os.getenv("ENV", "development")
RAILWAY_URL = os.getenv("RAILWAY_URL")
AWS_LAMBDA_HANDLER = os.getenv("AWS_LAMBDA_HANDLER")
AWS_REGION = os.getenv("AWS_REGION")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE")

API_KEY = os.getenv("API_KEY")
MODULES_PATH = os.getenv("MODULES_PATH", os.path.join(os.path.dirname(__file__), '..', 'modules'))

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Security dependency
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        logging.warning("Unauthorized access attempt")
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API Key")
    return api_key

# FastAPI app with metadata
app = FastAPI(
    title=os.getenv("APP_NAME", "ZSI Copilot Framework"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="Generic, pluggable Copilot framework for dynamic agent-based apps",
    openapi_tags=[
        {"name": "health", "description": "Health and wellness agents"},
        {"name": "productivity", "description": "Productivity and coaching agents"},
        {"name": "analytics", "description": "Data and signal analysis agents"},
    ]
)

# Configure CORS per environment
if ENV == "production":
    origins = os.getenv("CORS_ORIGINS", "").split(",")
else:
    origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——————————————
# DYNAMIC AGENT LOADING
# ——————————————
agent_registry = {}
for finder, name, ispkg in pkgutil.iter_modules([MODULES_PATH]):
    try:
        module = importlib.import_module(f'modules.{name}.agent')
        handler = getattr(module, 'handle_intent', None)
        if handler:
            agent_registry[name] = handler
    except ImportError as ie:
        logging.error(f"Failed to load agent '{name}': {ie}")

logging.info(f"Registered agents: {list(agent_registry.keys())}")

@app.post("/log", dependencies=[Depends(verify_api_key)], tags=["analytics"])
async def log_intent(payload: dict):
    """
    Dispatch an intent payload to the appropriate agent.
    """
    user_id = payload.get("user_id", "default_user")
    agent_key = payload.get("business_type") or payload.get("agent")
    if not agent_key or agent_key not in agent_registry:
        raise HTTPException(status_code=400, detail="Agent not found")
    date = datetime.utcnow().strftime("%Y-%m-%d")
    memory = load_user_memory(user_id, date)
    try:
        result = await agent_registry[agent_key](payload, memory)
    except Exception as e:
        logging.error(f"Error in agent '{agent_key}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent '{agent_key}' failed: {str(e)}")
    save_user_memory(user_id, date, memory)
    return result

@app.get("/ping", dependencies=[Depends(verify_api_key)], tags=["analytics"])
async def ping():
    """
    Health check and registered agents listing.
    """
    return {"status": "ok", "agents": list(agent_registry.keys())}

# Future hooks for:
#   - OAuth2 / JWT authentication
#   - Observability / metrics (e.g., Prometheus, AWS CloudWatch)
#   - Railway integrations (e.g., secrets, plugins)
#   - AWS Lambda wrapper (could export `handler = Mangum(app)` for API Gateway)
```