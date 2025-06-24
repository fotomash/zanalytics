
# risk_manager.py

import logging

log = logging.getLogger(__name__)


def calculate_risk(entry, stop, target):
    if entry == stop:
        log.warning("Entry price equals stop price. Unable to calculate risk/reward ratio.")
        raise ValueError("entry and stop prices must differ")

    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr_ratio = reward / risk
    return rr_ratio, risk, reward
