
class MacroAnalyzerAgent:
    def __init__(self, context=None):
        self.context = context or {}
        self.symbol = self.context.get("symbol", "XAUUSD")
        self.wyckoff_result = self.context.get("wyckoff_result", {})
        self.macro_snapshot = self.context.get("macro_snapshot", {})

    def evaluate_macro_bias(self):
        """Evaluate macro and HTF Wyckoff structure for bias."""
        result = {
            "symbol": self.symbol,
            "bias": "neutral",
            "phase": None,
            "macro_bias": None,
            "reason": "",
        }
        phase_class = self.wyckoff_result.get("phase_class")
        if phase_class:
            result["phase"] = phase_class
        if self.macro_snapshot:
            dxy = self.macro_snapshot.get("DXY", {}).get("bias")
            gold = self.macro_snapshot.get("XAUUSD", {}).get("bias")
            if gold:
                result["macro_bias"] = gold
            elif dxy:
                result["macro_bias"] = "bearish" if dxy == "bullish" else "bullish"
        if result["phase"] in ["C", "D"] and result["macro_bias"] == "bullish":
            result["bias"] = "bullish"
            result["reason"] = "HTF Phase C/D and macro alignment"
        elif result["phase"] == "E":
            result["bias"] = "neutral"
            result["reason"] = "Late phase - monitor for reversal"
        elif result["macro_bias"] == "bearish":
            result["bias"] = "bearish"
            result["reason"] = "Macro bearish pressure dominates"
        return result
