# feedback_analysis_engine.py
# Evaluates performance of entries and POIs using journal logs

import pandas as pd
import os

class FeedbackAnalysisEngine:
    def __init__(self, trade_log_path="journal/trade_log.csv"):
        self.trade_log_path = trade_log_path

    def load_trades(self):
        if not os.path.exists(self.trade_log_path):
            raise FileNotFoundError(f"No trade log found at: {self.trade_log_path}")
        return pd.read_csv(self.trade_log_path)

    def analyze_performance(self):
        df = self.load_trades()
        results = {
            "total_trades": len(df),
            "win_rate": 0.0,
            "average_rr": 0.0,
            "best_rr": None,
            "worst_rr": None
        }

        if "result" in df.columns and "rr" in df.columns:
            df["win"] = df["result"].str.lower() == "win"
            results["win_rate"] = df["win"].mean()
            results["average_rr"] = df["rr"].mean()
            results["best_rr"] = df["rr"].max()
            results["worst_rr"] = df["rr"].min()

        return results

    def get_failed_pois(self):
        df = self.load_trades()
        if "result" not in df.columns:
            return []
        failed = df[df["result"].str.lower() != "win"]
        return failed[["timestamp", "pair", "poi", "rr"]]


# To be called by journal summarizer, dashboard, or optimizer loop

