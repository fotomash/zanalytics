# xanflow/orchestrator/main_orchestrator.py
import os
import yaml
from xanflow.schemas.agent_profile_schemas import AgentProfileSchema
from xanflow.core.context_analyzer import ContextAnalyzer
from xanflow.core.liquidity_engine import LiquidityEngine
from xanflow.core.structure_validator import StructureValidator
from xanflow.core.fvg_locator import FVGLocator
from xanflow.core.risk_manager import RiskManager
from xanflow.core.confluence_stacker import ConfluenceStacker
from xanflow.core.executor import Executor
from xanflow.core.journal_logger import JournalLogger

class MainOrchestrator:
    def __init__(self, logger=None):
        self.logger = logger or print
        self.modules = {
            "context_analyzer": ContextAnalyzer(),
            "liquidity_engine": LiquidityEngine(),
            "structure_validator": StructureValidator(),
            "fvg_locator": FVGLocator(),
            "risk_manager": RiskManager(),
            "confluence_stacker": ConfluenceStacker(),
            "executor": Executor(),
            "journal_logger": JournalLogger()
        }

    def run_all_agents_from_yaml(self, profile_dir="profiles/agents"):
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

    def run_pipeline(self, profile):
        state = {"status": "START"}
        for step in profile.execution_sequence:
            if step not in self.modules:
                self.logger(f"[WARN] Step {step} not recognized.")
                continue
            self.logger(f"[STEP] {step}")
            module = self.modules[step]
            try:
                state = module.run(profile, state)
                if state.get("status") == "FAIL":
                    self.logger(f"[FAIL] Step {step} failed.")
                    break
            except Exception as e:
                self.logger(f"[ERROR] Exception in {step}: {e}")
                state["status"] = "FAIL"
                break
        self.logger(f"[RESULT] Final status: {state.get('status')}")