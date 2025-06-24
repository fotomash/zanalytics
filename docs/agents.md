# Agents

Defines agent modules implementing the ZSI pattern.

## Agent Contract
Each agent module must adhere to the following guidelines:

1. **Module Location & Auto-Discovery**  
   - File path: `modules/<agent_key>/agent.py`  
   - The orchestrator will dynamically discover any module using the `@register_agent` decorator.

2. **Registration Decorator**  
   - Import and apply:  
     ```python
     from core.agent import BaseAgent, register_agent
     @register_agent("<agent_key>")
     class YourAgent(BaseAgent):
         ...
     ```  
   - `<agent_key>` must match the `business_type` field in each incoming intent payload.

3. **Class Inheritance & Signature**  
   - Inherit from `BaseAgent`.  
   - Implement the method:  
     ```python
     def handle_intent(self, intent: Intent, memory: UserMemory) -> AgentResult:
         ...
     ```  
   - Use Pydantic models for `Intent`, `UserMemory`, and `AgentResult`.

4. **Memory API Usage**  
   - Access existing context: `memory.read(<key>)`  
   - Update or append context: `memory.update({<field>: <value>})`

5. **Standardized Output**  
   - Return an `AgentResult` dict with keys:  
     - `"message"`: `str` — user-facing response  
     - `"updates"`: `dict` — memory changes to persist  
     - *(optional)* `"suggestions"`: `List[Intent]` — follow-up actions or options  

## Example Agent

```python
# modules/example/agent.py
from core.agent import BaseAgent, register_agent
from schema.models import Intent, UserMemory, AgentResult

@register_agent("example")
class ExampleAgent(BaseAgent):
    def handle_intent(self, intent: Intent, memory: UserMemory) -> AgentResult:
        user_data = memory.read("profile") or {}
        # Business logic here...
        memory.update({"last_run": intent.timestamp})
        return AgentResult(
            message="Processed by ExampleAgent",
            updates={"profile": user_data},
            suggestions=[]
        )
```

## Available Agents
- **zbot**: Action-oriented responses
- **zse**: Signal and event analysis
- **zbar**: Behavioral journaling

> **Note:** Drop any new agent module into `modules/<agent_key>/agent.py` and register it via `@register_agent("<agent_key>")`. The orchestrator will automatically load it based on the `business_type`.
