from __future__ import annotations
from typing import Any, Dict, Optional
from schema.command_models import StructuredCommand

class LondonKillzoneAgent:
    """Specialist agent executing the London Kill Zone workflow."""

    def __init__(self, manifest: Dict[str, Any]):
        self.manifest = manifest

    def execute_workflow(self) -> Optional[StructuredCommand]:
        print("[LondonKillzoneAgent] Executing workflow steps...")
        for step in self.manifest.get("workflow", []):
            desc = step.get("description")
            print(f"  Step {step.get('step')}: {desc}")

        final_step = next(
            (s for s in self.manifest.get("workflow", []) if s.get("action_type")),
            None,
        )
        if final_step:
            cmd = StructuredCommand(
                request_id=self.manifest.get("strategy_id", "london_killzone"),
                action_type=final_step["action_type"],
                payload=final_step.get("payload", {}),
                human_readable_summary="London Kill Zone trade idea",
            )
            print(f"[LondonKillzoneAgent] Emitting command: {cmd.json()}")
            return cmd
        return None
