# ZAnalytics System Data Flow

## Overview
This document illustrates how data flows through the ZAnalytics system from user request to final response.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Dashboard     │───▶│   API Server    │───▶│  Orchestrator   │
│   (Streamlit)   │◀───│   (FastAPI)     │◀───│   (Asyncio)     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                     │                     │
          │                     ▼                     │
          │              ┌─────────────────┐          │
          │              │                 │          │
          │              │   DataManager   │          │
          │              │    (Phase 1)    │          │
          │              │                 │          │
          │              └─────────────────┘          │
          │                     │                     │
          ▼                     ▼                     ▼
     ┌─────────────────┐   ┌───────────┐   ┌─────────────────┐
     │     Parquet     │   │   CSV     │   │       S3        │
     └─────────────────┘   └───────────┘   └─────────────────┘
```

## Detailed Flow

### 1. User Interaction (Dashboard Layer)
The user selects analysis parameters in the Streamlit dashboard. The dashboard sends a request to the API server using the configured API client.

### 2. API Processing (API Layer)
The FastAPI server validates the request and constructs an `AnalysisRequest`. The request is submitted to the orchestrator and the API waits for a result or a timeout.

### 3. Orchestration (Core Intelligence Layer)
The orchestrator retrieves the request from its queue, fetches data via the `DataManager`, runs each analysis engine, aggregates the results and places them in a result queue.

### 4. Data Access (Data Layer)
The `DataManager` resolves file paths using `data_manifest.yml`, reads data from local storage or S3, standardises column names and validates the dataset before returning a `pandas.DataFrame`.

### 5. Response Flow
Results flow back through the layers:

- **Orchestrator** places the result in its output queue.
- **API** retrieves the result and sends a JSON response.
- **Dashboard** receives the response and updates the UI.

## Key Components

```python
@dataclass
class AnalysisRequest:
    request_id: str
    symbol: str
    timeframe: str
    analysis_type: str
    parameters: Dict[str, Any]
    priority: int = 5
    timestamp: datetime
    callback: Optional[Callable] = None

@dataclass
class AnalysisResult:
    request_id: str
    symbol: str
    timeframe: str
    analysis_type: str
    result_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    success: bool = True
    error: Optional[str] = None
```

## Error Handling
- **Dashboard:** displays user-friendly messages.
- **API:** returns HTTP error codes with explanatory JSON.
- **Orchestrator:** logs errors and forwards error details.
- **DataManager:** validates data and handles missing files.

## Performance Optimisations
- Caching of API responses and data files.
- Asynchronous operations and thread pools for I/O.
- Priority queues and configurable timeouts for request handling.

## Configuration Points
- `orchestrator_config.yaml` for orchestrator options.
- `data_manifest.yml` for data source configuration.
- Environment variables for runtime overrides.

## Monitoring
Key metrics include processing time, queue depth and error rates.

## Security Considerations
- Input validation in every layer.
- OAuth2 authentication (Phase 4.3).
- Rate limiting in the API layer.
- Centralised data access via the `DataManager`.

## Extending the System
Add a new engine, data source or API endpoint by updating the relevant configuration files and modules. The orchestration layer automatically integrates these components.

## Troubleshooting
- **No data returned:** verify `data_manifest.yml` paths and file existence. Check `DataManager` logs.
- **Timeout errors:** increase the API timeout, check orchestrator queue depth and verify engine performance.
- **Connection errors:** verify the API URL in the dashboard configuration, ensure all services are running and review firewall settings.
