from core.agent import BaseAgent, register_agent
from schema.models import Intent, AgentResult
from datetime import datetime

@register_agent("wyckoff_reversal")
class WyckoffReversalAgent(BaseAgent):
    """Wyckoff Spring/UTAD reversal agent."""

    def handle_intent(self, intent: Intent, memory) -> AgentResult:
        memory.add_log({"intent": intent.dict(), "ts": datetime.utcnow().isoformat()})
        memory.update({"wyckoff_reversal_last_run": intent.timestamp.isoformat()})
        return AgentResult(message="Wyckoff Reversal processed", updates={})
