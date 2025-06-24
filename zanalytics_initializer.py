# agent_htfanalyst.py

class HTFPhaseAnalystAgent:
    def __init__(self, context=None):
        self.context = context or {}
        self.symbol = self.context.get("symbol", "XAUUSD")
        self.htf_phase_data = self.context.get("wyckoff_result", {})

    def evaluate_wyckoff_phase_context(self):
        """
        Analyze the H1/H4 Wyckoff phase classification and determine directional bias.
        """
        result = {
            "symbol": self.symbol,
            "phase": None,
            "bias": "neutral",
            "reason": ""
        }

        if not self.htf_phase_data or not isinstance(self.htf_phase_data, dict):
            result["reason"] = "No HTF Wyckoff data available"
            return result

        phase_class = self.htf_phase_data.get("phase_class", None)
        result["phase"] = phase_class

        if phase_class in ["C", "D"]:
            result["bias"] = "bullish"
            result["reason"] = f"HTF in Phase {phase_class} (Spring or LPS zone)"
        elif phase_class == "E":
            result["bias"] = "neutral"
            result["reason"] = "HTF in Phase E - distribution or exhaustion"
        else:
            result["reason"] = f"Unrecognized phase: {phase_class}"

        return result

    def to_markdown_summary(self):
        """
        Generate a Markdown-formatted summary of the Wyckoff phase evaluation.
        """
        eval_result = self.evaluate_wyckoff_phase_context()
        md = (
            f"# Wyckoff Phase Analysis Summary for {eval_result['symbol']}\n\n"
            f"- **Phase:** {eval_result['phase']}\n"
            f"- **Bias:** {eval_result['bias']}\n"
            f"- **Reason:** {eval_result['reason']}\n"
        )
        return md

    def to_dict(self):
        """
        Return the Wyckoff evaluation result as a plain dictionary.
        """
        return self.evaluate_wyckoff_phase_context()
