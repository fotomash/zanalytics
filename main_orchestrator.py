"""Simple YAML-driven orchestrator for the ZANALYTICS agents."""

import importlib
import os
from typing import Any, Dict

import yaml

from schemas.agent_profile_schemas import AgentProfileSchema

class MainOrchestrator:
    def __init__(self, logger=None):
        self.logger = logger or print
        # Modules are loaded lazily based on the agent profile's code_map
        self.modules: Dict[str, Any] = {}

    def _load_module(self, key: str, import_path: str) -> Any:
        """Import and instantiate a module based on its dotted path."""
        try:
            module_path, attr_name = import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            attr = getattr(module, attr_name)
            instance = attr() if callable(attr) else attr
            self.modules[key] = instance
            return instance
        except Exception as exc:  # pragma: no cover - import errors
            self.logger(f"[ERROR] Failed loading {key} from {import_path}: {exc}")
            return None

    def run_all_agents_from_yaml(self, profile_dir: str = "profiles/agents") -> None:
        for fname in os.listdir(profile_dir):
            if not fname.endswith(".yaml"):
                continue
            with open(os.path.join(profile_dir, fname)) as f:
                agent_config = yaml.safe_load(f)
                try:
                    profile = AgentProfileSchema(**agent_config)
                except Exception as e:
                    self.logger(f"[ERROR] Invalid profile {fname}: {e}")
                    continue
                self.logger(f"[START] Executing agent: {profile.meta_agent.agent_id}")
                self.run_pipeline(profile)

    def run_pipeline(self, profile: AgentProfileSchema) -> None:
        state = {"status": "START"}
        for step in profile.execution_sequence:
            module = self.modules.get(step)
            if not module:
                import_path = profile.code_map.get(step)
                if import_path:
                    module = self._load_module(step, import_path)
                if not module:
                    self.logger(f"[WARN] Step {step} not recognized.")
                    continue
            self.logger(f"[STEP] {step}")
            try:
                if hasattr(module, "run"):
                    state = module.run(profile, state)
                elif callable(module):
                    state = module(profile, state)
                else:
                    self.logger(f"[WARN] Module for {step} is not callable")
                    state["status"] = "FAIL"
                    break
                if state.get("status") == "FAIL":
                    self.logger(f"[FAIL] Step {step} failed.")
                    break
            except Exception as e:
                self.logger(f"[ERROR] Exception in {step}: {e}")
                state["status"] = "FAIL"
                break
        self.logger(f"[RESULT] Final status: {state.get('status')}")
