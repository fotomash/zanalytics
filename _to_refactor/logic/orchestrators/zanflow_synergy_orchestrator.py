
# zanflow_synergy_orchestrator.py

from orchestrators.pipeline_manager import run_pipeline
from orchestrators.signal_router import route_signal
from orchestrators.memory_bridge import update_shared_context
from journal.journal_logger import log_trade_decision

def zanflow_synergy_orchestrator(input_data):
    """
    Master orchestration function to handle:
    - Predictive scoring
    - Confluence resolution
    - Fallback pathing
    - Signal journaling
    """

    # Step 1: Update memory from shared context
    context = update_shared_context(input_data)

    # Step 2: Route signal through tick/M1 workflow
    routed = route_signal(context)

    # Step 3: Execute full agent pipeline
    result = run_pipeline(routed)

    # Step 4: Log decision, maturity, and rejection causes
    log_trade_decision(result)

    return result
