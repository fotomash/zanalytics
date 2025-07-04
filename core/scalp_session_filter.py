"""
scalp_session_filter.py

Session-based trading filter for ZANALYTICS scalper module.
Prevents entries during known low-liquidity times (e.g., rollover, post-settlement)
and activates only during predefined high-probability sessions (e.g., London, NY).

Returns a structured dict:
{
    "session_ok": True/False,
    "session_name": "London" or None,
    "utc_time": "13:45",
    "reason": "Inside London window (07:00–10:30)"
}
"""

from datetime import datetime, time

def is_within_session_time(current_utc: datetime, config: dict) -> dict:
    """
    Check if the current UTC time is within allowed trading sessions and outside of blocked windows.
    Returns a dict with session_ok flag and reason.
    """

    current_time = current_utc.time()
    session_ok = False
    active_session = None
    reason = ""

    # Parse allowed sessions
    for session in config.get("session_blocks", []):
        start = datetime.strptime(session["start"], "%H:%M").time()
        end = datetime.strptime(session["end"], "%H:%M").time()
        if start <= current_time <= end:
            session_ok = True
            active_session = session["name"]
            reason = f"Inside {active_session} window ({session['start']}–{session['end']})"
            break

    # Check blocked windows override
    for blocked in config.get("blocked_windows", []):
        block_start = datetime.strptime(blocked["start"], "%H:%M").time()
        block_end = datetime.strptime(blocked["end"], "%H:%M").time()
        if block_start <= current_time <= block_end:
            session_ok = False
            reason = f"Blocked window: {blocked['start']}–{blocked['end']}"
            break

    return {
        "session_ok": session_ok,
        "session_name": active_session if session_ok else None,
        "utc_time": current_time.strftime("%H:%M"),
        "reason": reason or "Outside allowed session windows"
    }

# --------------------------------------------------------------------------- #
# Extended utilities for session-based filtering                              #
# --------------------------------------------------------------------------- #
def evaluate_session_conditions(
    current_utc: datetime,
    config: dict,
    market_conditions: dict | None = None
) -> dict:
    """
    Combines the basic session‐time filter with optional market condition checks
    such as minimum volatility or volume.

    Parameters
    ----------
    current_utc : datetime
        Current UTC timestamp (typically `datetime.utcnow()` in calling code).
    config : dict
        Configuration dictionary that must include the same keys used by
        `is_within_session_time`, plus optional:
            - min_volatility (float) : minimum acceptable volatility metric
            - min_volume (float)     : minimum acceptable volume metric
    market_conditions : dict | None
        Dictionary that can hold real‑time market metrics, e.g.:
            {"volatility": 0.9, "volume": 1250}

    Returns
    -------
    dict
        The original session check dict, possibly updated with additional
        failure reasons if market conditions are not met.
    """
    session_info = is_within_session_time(current_utc, config)

    # Early exit if time filter already failed
    if not session_info["session_ok"]:
        return session_info

    # Enrich checks with volatility / volume if provided
    if market_conditions:
        if "min_volatility" in config:
            if market_conditions.get("volatility", 0) < config["min_volatility"]:
                session_info["session_ok"] = False
                session_info["reason"] += " | Volatility below threshold"

        if "min_volume" in config:
            if market_conditions.get("volume", 0) < config["min_volume"]:
                session_info["session_ok"] = False
                session_info["reason"] += " | Volume below threshold"

    return session_info


def pretty_print_session_report(session_info: dict) -> None:
    """
    Utility logger for quick console debugging of session checks.
    """
    print(
        f"[SESSION] OK={session_info['session_ok']} | "
        f"UTC={session_info['utc_time']} | "
        f"Session={session_info['session_name'] or '-'} | "
        f"Reason={session_info['reason']}"
    )