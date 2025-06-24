 # Entry point for FastAPI app
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != "YOUR_SECURE_API_KEY":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key

app = FastAPI(
    title="ZSI Copilot Framework",
    version="1.0.0",
    description="Universal, pluggable agent-based Copilot framework with dynamic discovery, memory, and security built-in.",
    openapi_tags=[
        {"name": "health-check", "description": "Ping endpoint"},
        {"name": "intents", "description": "Log user intents and dispatch to agents"}
    ]
)

# CORS configuration
origins = ["*"]  # TODO: tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import HTTPException, Request
from core.orchestrator import handle_user_input

@app.post("/log", tags=["intents"], dependencies=[Depends(verify_api_key)])
async def log_intent(request: Request):
    payload = await request.json()
    try:
        result = await handle_user_input(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping", tags=["health-check"])
async def ping():
    return {"message": "pong"}

# — Expansion Hooks — 
# e.g., integrate structured logging, tracing, metrics here
