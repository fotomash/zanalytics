# Memory Layer

The following schema is a generic, extensible memory model suitable for any Copilot-based application.

```json
{
  "version": "1.0",
  "user_id": "demo",
  "date": "2025-05-24",
  "context": {
    "global": {},
    "session": {},
    "preferences": {}
  },
  "interactions": [
    {
      "timestamp": "2025-05-24T12:34:56Z",
      "agent": "example",
      "input": {},
      "output": {},
      "metadata": {}
    }
  ],
  "agent_history": [
    {
      "timestamp": "2025-05-24T12:34:56Z",
      "agent": "example",
      "result": {},
      "tags": [],
      "insights": []
    }
  ],
  "dynamic_state": {},
  "custom_metrics": {},
  "extensions": {}
}
```