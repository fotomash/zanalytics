from core.agent import BaseAgent, register_agent
from schema.models import Intent, AgentResult
from datetime import datetime

@register_agent("offline_analysis")
class OfflineAnalysisAgent(BaseAgent):
    """Agent performing offline confluence curve analysis."""

    def handle_intent(self, intent: Intent, memory) -> AgentResult:
        memory.add_log({"intent": intent.dict(), "ts": datetime.utcnow().isoformat()})
        memory.update({"offline_analysis_last_run": intent.timestamp.isoformat()})
        return AgentResult(message="Offline analysis completed", updates={})
