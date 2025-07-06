import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

CONFIG_PATH = Path("zsi_config.yaml")
USER_CONTEXT_PATH = Path("data/user_context.json")

def _load_config() -> Dict[str, Any]:
    if CONFIG_PATH.is_file() and yaml:
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}

def _read_user_context() -> Dict[str, Any]:
    if USER_CONTEXT_PATH.is_file():
        try:
            with USER_CONTEXT_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _write_user_context(ctx: Dict[str, Any]) -> None:
    USER_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_CONTEXT_PATH.open("w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2)

async def handle_user_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Route payloads to the appropriate orchestrator."""
    _load_config()  # Loaded for side effects if needed
    context = _read_user_context()

    orchestrator = payload.get("orchestrator")
    if orchestrator == "copilot":
        try:
            from core.copilot_orchestrator import handle_prompt
            result = handle_prompt(payload.get("prompt", ""))
        except Exception as exc:  # pragma: no cover - optional
            result = {"status": "error", "message": str(exc)}
    elif orchestrator == "advanced_smc":
        try:
            from core.advanced_smc_orchestrator import run_advanced_smc_strategy
            result = run_advanced_smc_strategy(**payload.get("args", {}))
        except Exception as exc:  # pragma: no cover - optional
            result = {"status": "error", "message": str(exc)}
    else:
        try:
            from main_orchestrator import MainOrchestrator
            mo = MainOrchestrator()
            mo.run_all_agents_from_yaml()
            result = {"status": "ok"}
        except Exception as exc:  # pragma: no cover - optional
            result = {"status": "error", "message": str(exc)}

    context["last_result"] = result
    _write_user_context(context)
    return result
