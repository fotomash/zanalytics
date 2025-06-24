# optimizer_loop.py
# Iteratively adjusts strategy parameters based on feedback data

import json
import os
from feedback_analysis_engine import FeedbackAnalysisEngine

class OptimizerLoop:
    def __init__(self, config_path="config/strategy_profiles.json"):
        self.config_path = config_path
        self.engine = FeedbackAnalysisEngine()
        self.current_config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {}

    def update_config(self, suggestions):
        updated = self.current_config.copy()
        for key, new_value in suggestions.items():
            if key in updated:
                updated[key] = new_value
        self.current_config = updated
        return updated

    def optimize(self):
        feedback = self.engine.analyze_performance()

        suggestions = {}
        if feedback["win_rate"] < 0.4:
            suggestions["entry_threshold"] = max(0.6, self.current_config.get("entry_threshold", 0.5) + 0.1)
        if feedback["average_rr"] < 1.5:
            suggestions["min_rr_required"] = 2.0
        if feedback["worst_rr"] and feedback["worst_rr"] < 0.5:
            suggestions["poi_score_threshold"] = self.current_config.get("poi_score_threshold", 0.6) + 0.1

        updated_config = self.update_config(suggestions)
        return {
            "suggestions": suggestions,
            "new_config": updated_config
        }

    def save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self.current_config, f, indent=2)


if __name__ == "__main__":
    loop = OptimizerLoop()
    result = loop.optimize()
    loop.save_config()
    print(json.dumps(result, indent=2))

    # Optional: send Telegram alert with summary
    try:
        from telegram_alert_engine import send_simple_summary_alert
        summary_msg = f"📊 Strategy Updated:\n{json.dumps(result['suggestions'], indent=2)}"
        send_simple_summary_alert(summary_msg)
    except Exception as e:
        print(f"[WARN] Telegram alert failed: {e}")
