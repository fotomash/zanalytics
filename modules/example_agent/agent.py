from core.agent import BaseAgent, register_agent
from schema.models import Intent, AgentResult
from user_memory import UserMemory
from datetime import datetime

@register_agent("example")
class ExampleAgent(BaseAgent):
    def handle_intent(self, intent: Intent, memory: UserMemory) -> AgentResult:
        # Example processing logic
        memory.add_log({"intent": intent.dict(), "ts": datetime.utcnow().isoformat()})
        return AgentResult(message="Processed by ExampleAgent", updates={})
