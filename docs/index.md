# Copilot Framework Documentation
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-org/your-repo/ci.yml)](https://github.com/your-org/your-repo/actions)
[![Version](https://img.shields.io/github/v/release/your-org/your-repo)](https://github.com/your-org/your-repo/releases)

Welcome to the **Copilot Framework**—your turnkey guide to building modular, voice‑aware AI copilots powered by the Zanzibar Structured Intelligence (ZSI) architecture. From DietPilot to CareMatch, KynderWay to Zanalytics.app, this framework adapts to any domain and scales with your vision.

## What You’ll Find Here

### Getting Started
- Step-by-step setup
- Your first “Hello, Copilot” interaction

### Architecture
- Core entry points (`main.py`, `core/orchestrator.py`)
- Dynamic agent discovery in `modules/`
- Agent contracts & routing logic
- Memory layer design & state persistence
- Flow engine for user journeys (`/flows/`)
- Hosting strategies: FastAPI (Railway), AWS Lambda/API Gateway, and beyond

### Agents
- **ZBOT**: Action execution
- **ZSE**: Signal & event analysis
- **ZBAR**: Behavioral journaling

### Flows
- Prebuilt interaction scenarios: daily logging, macro adjustments, cheat days, and more

### Schemas
- Pydantic models & JSON schemas for intents, user context, and agent outputs

### System Prompt & Routing
- LLM prompt best practices
- Persona definitions
- Intent dispatch patterns

### Persistence & Memory
- Local JSON vs. cloud storage
- Designing a robust user context DB

### Security
- Deployment checklist
- Data protection & API security best practices

Use the sidebar to navigate, or jump directly to the [GitHub repo](https://github.com/your-org/your-repo) for code, issues, and contributions.

## Configuration
This framework uses [Pydantic Settings](https://pydantic-docs.helpmanual.io/usage/settings/) and environment variables to centralize runtime configuration. Copy `.env.example` to `.env` and define:
- `API_KEY`: Your global service authentication key
- `ENVIRONMENT`: `development` or `production`
- `DATABASE_URL`: Connection string for your persistence layer
- `LOG_LEVEL`: `DEBUG`, `INFO`, `WARN`, or `ERROR`

## Observability
The framework includes hooks for OpenTelemetry and structured logging. To enable:
1. Configure exporters in your `.env`.
2. Review the [Observability](security.md#observability) guide for setup details.
