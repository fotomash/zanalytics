# Organic Intelligence Loop

This page describes the autonomous workflow added in **Phase&nbsp;5**.

## Structured Command System

Agents and LLMs exchange actions defined in [`command_schema.json`](../zanalytics_phase5_organic_intelligence/core/commands/command_schema.json). Each command includes:

- `request_id` – unique identifier
- `action_type` – routing flag such as `LOG_JOURNAL_ENTRY` or `EXECUTE_TRADE_IDEA`
- `payload` – action‑specific data
- optional metadata like `source` and `timestamp`

Supported `action_type` values are:

```json
{
  "enum": [
    "LOG_JOURNAL_ENTRY",
    "UPDATE_DASHBOARD_STATE",
    "TRIGGER_AGENT_ANALYSIS",
    "NOTIFY_USER",
    "EXECUTE_TRADE_IDEA",
    "UPDATE_MARKET_CONTEXT",
    "SCHEDULE_FOLLOWUP",
    "ARCHIVE_ANALYSIS"
  ]
}
```

All decisions become traceable commands that the dispatcher can log, schedule, or forward to other services.

## Intelligent Scheduling

The `SchedulingAgent` runs strategies on a fixed timetable or when conditions are met. The sample manifest [`London_Killzone_SMC.yml`](../zanalytics_phase5_organic_intelligence/knowledge/strategies/London_Killzone_SMC.yml) triggers every minute between **06:00** and **09:00 UTC** on weekdays. The agent checks market status and news before activating the workflow.

## London Kill Zone Workflow

The manifest defines a six‑step procedure:

1. **Identify Asian range** – determine highs and lows of the Asian session.
2. **Detect liquidity sweep** – watch for sweeps of Asian range levels.
3. **Confirm market structure** – look for BOS/CHOCH confirmation.
4. **Refine entry zone** – use MIDAS curve and order blocks.
5. **Calculate trade parameters** – compute entry, stop loss, and targets.
6. **Execute trade idea** – log the idea, update the dashboard, and notify the user.

A monitoring section tracks progress after execution.

## Example Command

When a valid setup is found, an `EXECUTE_TRADE_IDEA` command might look like:

```json
{
  "action_type": "EXECUTE_TRADE_IDEA",
  "payload": {
    "symbol": "EURUSD",
    "direction": "buy",
    "entry_price": 1.0850,
    "stop_loss": 1.0820,
    "take_profits": [1.0880, 1.0910, 1.0950]
  }
}
```

The journal is updated, the dashboard highlights key levels, and real‑time notifications are sent through the WebSocket service.

