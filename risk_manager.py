
# risk_manager.py

def calculate_risk(entry, stop, target):
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr_ratio = reward / risk if risk != 0 else 0
    return rr_ratio, risk, reward
