# CLI Simulator

The CLI Simulator provides a local, interactive entry point to your Copilot Framework agents. It emulates the same input/output flow as the FastAPI endpoints, but directly in your terminal for rapid testing, development, and debugging.

---

## 🚀 Prerequisites

- **Python** 3.9+
- Project dependencies installed:  
  ```bash
  pip install -r requirements.txt
  ```
- A valid **API key** (or JWT) set via environment variable or flag.

---

## 🔧 Installation

1. **Clone your repository**  
   ```bash
   git clone https://github.com/your-org/your-repo.git
   cd your-repo
   ```
2. **Create & activate** a virtual environment:  
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\Scripts\activate       # Windows
   ```
3. **Install** dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

The CLI reads configuration from environment variables or a `.env` file in the project root:

```bash
# .env example
API_KEY=your_api_key_here
MEMORY_BACKEND=json         # choices: json, redis, dynamodb
AGENT_MODULES_PATH=modules  # path to dynamic agent folders
LOG_LEVEL=info              # debug, info, warn, error
HOSTING_TIER=mvp            # mvp, scale, production
```

Override via flags:
```
--api-key
--memory-backend
--agent-path
--log-level
--hosting-tier
```

---

## ▶️ Usage

Launch the interactive REPL:

```bash
python -m cli_simulator \
  --api-key "$API_KEY" \
  --memory-backend json \
  --agent-path modules \
  --hosting-tier mvp
```

At the prompt, enter JSON payloads that conform to the agent contract:

```json
{
  "user_id": "demo_user",
  "business_type": "example",
  "intent": "greet",
  "payload": { "message": "Hello CLI!" }
}
```

- **Enter** to submit.
- **exit** or **Ctrl+D** to terminate.

---

## 🛠️ Commands & Flags

| Flag                | Description                                             | Default    |
|---------------------|---------------------------------------------------------|------------|
| `--api-key`         | API key or JWT for authentication                       | `$API_KEY` |
| `--memory-backend`  | Memory store (`json`, `redis`, `dynamodb`)              | `json`     |
| `--agent-path`      | Directory for dynamic agent discovery                   | `modules`  |
| `--log-level`       | Logging verbosity (`debug`, `info`, `warn`, `error`)    | `info`     |
| `--hosting-tier`    | Deployment profile (`mvp`, `scale`, `production`)       | `mvp`      |

---

## 📖 Examples

1. **Redis backend**  
   ```bash
   python -m cli_simulator --memory-backend redis
   ```
2. **Ping agent**  
   ```bash
   > {"user_id":"u1","business_type":"ping","intent":"ping","payload":{}}
   {"status":"ok","response":"pong"}
   ```
3. **Simulate production**  
   ```bash
   python -m cli_simulator --hosting-tier production
   ```

---

## 🐞 Troubleshooting

- **Missing API key**: Verify `API_KEY` or `--api-key` is provided.
- **Agent not found**: Confirm folder name under `modules/` matches `business_type`.
- **Memory errors**: Ensure your chosen backend service is running (e.g., Redis, DynamoDB).

---

## ⚙️ Extensibility & Future-Proofing

- **Dynamic Agents**: Drop new agent modules into `modules/<agent_name>/agent.py`.  
- **Hosting Profiles**: `--hosting-tier` toggles configuration presets for FastAPI on Railway, AWS Lambda/API Gateway, or Kubernetes.
- **Security Hooks**: Integrate OAuth2 or JWT flows by extending `cli_simulator.py`.
- **Observability**: Enable advanced telemetry or distributed tracing via flags (`--telemetry`, `--tracing`).

---

## 🔗 Further Reading

- **FastAPI Endpoints**: `docs/getting-started.md#api-endpoints`
- **Agent Contract & Schema**: `docs/schemas.md`
- **Persistence & Memory**: `docs/persistence.md`
- **Security & Auth**: `docs/security.md`

---

> *Nail it before you scale it.*  
> Designed for any vertical—health, finance, trading, nannies, or productivity—this CLI is your universal Copilot launchpad.