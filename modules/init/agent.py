from core.agent import BaseAgent, register_agent
from schema.models import InitIntent, InitResult
import pkg_resources
import yaml

@register_agent("init")
class InitAgent(BaseAgent):
    """
    The init agent:
     - Loads & returns your system_prompt
     - Seeds or migrates user memory
     - Reports framework metadata (version, loaded agents)
    """

    def handle_intent(self, intent: InitIntent, memory) -> InitResult:
        # 1. Load system prompt
        sys_md = pkg_resources.resource_string(__name__, "../../system/system_prompt.md").decode()

        # 2. Discover agents
        agents = list(self.agent_registry.keys())

        # 3. Optionally migrate or seed memory
        if not memory.exists():
            memory.seed_default()

        # 4. Return InitResult
        return InitResult(
            message="Framework initialized.",
            system_prompt=sys_md,
            available_agents=agents,
            version=self.framework_version,
        )