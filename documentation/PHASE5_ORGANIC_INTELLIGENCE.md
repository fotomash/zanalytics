# Zanalytics Phase 5 - Organic Intelligence Loop

## Overview

Phase 5 introduces a proactive intelligence layer that schedules and executes strategies automatically. Commands produced by language models are processed in a structured way, enabling event driven workflows and hands free trading logic.

Key highlights:

- **Autonomous Strategy Execution** via a scheduling agent
- **Structured LLM Integration** using a common command schema
- **Event Driven Architecture** that reacts to monitoring triggers
- **Seamless User Experience** from analysis to execution

## Integration Steps

1. Run `integrate_organic_intelligence.py` from the repository root.
2. Review the generated `main_additions.py` and merge it into your entrypoint (the default is `run_system.py`).
3. Install new dependencies with `pip install -r requirements.txt`.
4. Ensure Redis is running.
5. Start the organic intelligence loop with `python run_organic_intelligence.py` or include it in your own startup process.

## Configuration

The integration script creates `config/organic_intelligence.json` with default settings. Adjust these parameters to control scheduling and agent behaviour.

## Files Added

- `core/agents/` – scheduling and strategy agents
- `core/dispatcher/` – action dispatcher
- `core/command_processor.py` – orchestrates command execution
- `knowledge/strategies/` – example strategy manifest
- `api/endpoints/organic_intelligence.py` – optional API router

## Next Steps

Continue expanding strategies and agents within `zanalytics_phase5_organic_intelligence` and iterate on the command schema as your system evolves.
