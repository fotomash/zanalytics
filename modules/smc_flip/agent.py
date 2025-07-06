"""SMC Structural Flip & POI Confirmation agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from core.agent import BaseAgent, register_agent
from schema.command_models import StructuredCommand
from schema.models import AgentResult, Intent

@register_agent("smc_flip")
class SMCFlipAgent(BaseAgent):
    """SMC Structural Flip & POI Confirmation agent."""

    def handle_intent(self, intent: Intent, memory) -> AgentResult:
        """Execute the workflow defined in the provided manifest if present."""
        memory.add_log({"intent": intent.dict(), "ts": datetime.utcnow().isoformat()})

        manifest: Optional[Dict[str, Any]] = None
        if isinstance(intent.payload, dict):
            manifest = intent.payload.get("manifest")
            if not manifest:
                ctx = intent.payload.get("context", {})
                manifest = ctx.get("manifest")

        command = None
        if manifest:
            command = self.execute_workflow(manifest)

        memory.update({"smc_flip_last_run": intent.timestamp.isoformat()})

        metadata = {"command": command.dict()} if command else {}
        return AgentResult(message="SMC Flip processed", updates={}, metadata=metadata)

    def execute_workflow(self, manifest: Dict[str, Any]) -> Optional[StructuredCommand]:
        """Log each workflow step and return final StructuredCommand if defined."""
        print("[SMCFlipAgent] Executing workflow steps...")
        for step in manifest.get("workflow", []):
            desc = step.get("description")
            print(f"  Step {step.get('step')}: {desc}")

        final_step = next((s for s in manifest.get("workflow", []) if s.get("action_type")), None)
        if final_step:
            cmd = StructuredCommand(
                request_id=manifest.get("strategy_id", "smc_flip"),
                action_type=final_step["action_type"],
                payload=final_step.get("payload", {}),
                human_readable_summary="SMC Flip trade idea",
            )
            print(f"[SMCFlipAgent] Emitting command: {cmd.json()}")
            return cmd
        return None
