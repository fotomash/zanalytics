from .base_agent import BaseAgent

class MicroStrategistAgent(BaseAgent):
    """Generate micro-level trading signals based on real-time data."""

    def __init__(self, agent_id: str, config: dict, memory_manager):
        super().__init__(agent_id, config, memory_manager)
        self.symbol = config.get("symbol", "UNKNOWN")
        self.phase_info = None

    def set_phase_info(self, dataframe):
        """Attach microstructure dataframe to the agent."""
        self.phase_info = dataframe

    async def process(self, data):
        """Process incoming data and return a trading signal."""
        return self.evaluate_microstructure_phase_trigger()

    def evaluate_microstructure_phase_trigger(self):
        """Assess microstructure to decide if an entry signal is triggered."""
        result = {
            "entry_signal": False,
            "trigger": None,
            "confidence": 0.0,
            "reason": "",
            "symbol": self.symbol,
            "phase_context": None,
            "volatility": None,
            "execution_mode": "scalp",
        }

        if self.phase_info is None or getattr(self.phase_info, "empty", False):
            result["reason"] = "No microstructure data available"
            return result

        last = self.phase_info.iloc[-1]
        spring = last.get("SPRING", False)
        choch = last.get("CHoCH", False)
        bos = last.get("BOS", False)
        phase = last.get("PHASE", "")
        spread = last.get("SPREAD", 0)
        ret = last.get("RET", 0)

        weights = {
            "spring": 1.2,
            "choch_trap": 0.9,
            "bos_confirm": 0.5,
            "spread_compression": 0.3,
            "reversal_tick": 0.2,
        }

        score = 0.0
        reasons = []

        if spring:
            score += weights["spring"]
            reasons.append("Spring detected")
        if choch and not bos:
            score += weights["choch_trap"]
            reasons.append("CHoCH without BOS (trap zone)")
        if bos:
            score += weights["bos_confirm"]
            reasons.append("BOS confirms structure shift")
        if spread < 0.3:
            score += weights["spread_compression"]
            result["volatility"] = "compressed"
        if abs(ret) > 0.0004:
            score += weights["reversal_tick"]
            reasons.append("strong reversal tick")

        if score >= 1.5:
            result["entry_signal"] = True
            result["confidence"] = min(score / sum(weights.values()), 1.0)
            result["trigger"] = "+".join(reasons)
            result["reason"] = " + ".join(reasons)
            result["phase_context"] = phase
        else:
            result["reason"] = "Score too low: " + ", ".join(reasons)

        result["raw_score"] = score
        if ret > 0:
            result["tick_bias"] = "bullish"
        elif ret < 0:
            result["tick_bias"] = "bearish"
        else:
            result["tick_bias"] = "neutral"

        return result
