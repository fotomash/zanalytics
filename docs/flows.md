# Flows

These flows define standard end-to-end processes for any vertical, enabling plug-and-play domain agents.

## 1. Intent Routing Flow  
**Actors:**  
User, Orchestrator, Agent  
**Triggers:**  
User submits input via `/log` endpoint  
**Inputs:**  
Intent, action, query, and an event payload containing `business_type` (e.g., "trading", "nanny") or `agent` field to route domain-specific handlers  
**Outputs:**  
Agent response, updated memory  

1. User sends input containing intent and metadata.  
2. Orchestrator parses `business_type` or `agent` to identify handler.  
3. Orchestrator dispatches to matching agent from registry.  
4. Agent processes intent, reads/writes to memory.  
5. Orchestrator persists memory and returns agent output.

## 2. Daily Logging Flow  
**Actors:**  
User, Orchestrator, Agent  
**Triggers:**  
User logs daily activity or status  
**Inputs:**  
User intent payload (activity, event, status)  
**Outputs:**  
Summary response, updated state  

1. User submits daily log entry.  
2. Orchestrator routes to appropriate agent via registry (`<agent_key>`).  
3. Agent processes data, updates memory accordingly.  
4. Orchestrator saves state and triggers optional follow-ups.  
5. Agent replies with summary and next-step suggestions.

## 3. Adjustment Flow  
**Actors:**  
Monitoring Agent, Orchestrator, Related Agents  
**Triggers:**  
Post-log memory review  
**Inputs:**  
Current memory state  
**Outputs:**  
Adjusted goals, recommendations  

1. Monitoring agent reviews memory after each log.  
2. Detects deviations (under/over targets, skipped events).  
3. Triggers relevant agents to recalculate goals or suggest actions.  
4. Orchestrator returns updated goals and recommendations.

## 4. Special Event Flow  
**Actors:**  
User, Orchestrator, Agents, Monitoring Agents  
**Triggers:**  
User logs special event with tag or `business_type`  
**Inputs:**  
Event data with special markers  
**Outputs:**  
Context prompts, pattern records, reinforcement actions  

1. User logs event flagged as special.  
2. Orchestrator dispatches to designated agent for handling.  
3. Related agents prompt for additional context or tagging.  
4. Monitoring agents analyze patterns and provide reinforcement.

## 5. Agent Registration & Discovery Flow  
**Actors:**  
Orchestrator, Agent Modules  
**Triggers:**  
System startup  
**Inputs:**  
Agent modules in `modules/<agent_name>/agent.py`  
**Outputs:**  
Agent registry mapping keys to handlers  

1. On startup, orchestrator scans `modules/` for agent modules.  
2. Each agent registers via `@register_agent(name="...")`.  
3. Registry maps agent keys to handlers.  
4. New modules auto-included without code changes.

## 6. Memory Persistence Flow  
**Actors:**  
Orchestrator, Agents, Persistence Layer  
**Triggers:**  
Agent read/write operations  
**Inputs:**  
User context from persistence store  
**Outputs:**  
Updated context persisted  

1. Load or initialize user context via abstract persistence layer (e.g., JSON, Firestore, DynamoDB).  
2. Agents read and update context fields.  
3. Orchestrator saves updated context back to store.  
4. Optional daily summaries accessible via `/summary` endpoint.

## 7. System Prompt & Routing Flow  
**Actors:**  
Orchestrator, LLM, Agents  
**Triggers:**  
Intent processing requiring LLM  
**Inputs:**  
System prompt, memory snapshot, agent instructions  
**Outputs:**  
Structured LLM JSON response, user-facing text  

1. System prompt (`system/system_prompt.md`) and agent files define LLM behavior.  
2. Orchestrator injects prompt and memory snapshot into GPT call.  
3. LLM returns structured JSON with response and next actions.  
4. Orchestrator parses output, updates memory, and sends user text.

## 8. Dynamic Extension Flow  
**Actors:**  
Orchestrator, Agent Modules  
**Triggers:**  
New module addition to `modules/` directory  
**Inputs:**  
Agent module files with registration decorators  
**Outputs:**  
Updated agent registry with new capabilities  

1. New modules dropped into `modules/` detected on orchestrator startup.  
2. Each registers agents via standard decorator.  
3. Agents become immediately available without code changes.  
4. Enables hot-plugging new verticals and features dynamically.

## 9. Security & Access Control Flow  
**Actors:**  
User, Orchestrator, Authentication Service, Authorization Layer  
**Triggers:**  
Incoming API requests  
**Inputs:**  
Authentication tokens, user credentials, access policies  
**Outputs:**  
Authenticated sessions, authorized actions, audit logs  

1. User requests authenticated via tokens or credentials.  
2. Orchestrator validates authentication with external or internal service.  
3. Authorization layer checks user permissions against access policies.  
4. Unauthorized requests are rejected with appropriate errors.  
5. Successful authentications enable action routing to agents.  
6. All access and actions are logged for auditing and compliance.  

---

*Last updated: CTO-ready v1.0*
