"""Utility agent for journaling trade decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TradeJournalistAgent:
    """Maintain a structured journal of trade decisions."""

    context: Optional[Dict[str, Any]] = None
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self, entry: Dict[str, Any]) -> str:
        """Return a concise string representation of a log entry."""
        return (
            f"[{entry['timestamp']}] {entry['symbol']} | {entry['trigger']} | "
            f"Risk: {entry.get('risk_level')} | Conf: {entry.get('confidence')} | "
            f"Bias: {entry.get('macro_bias')}"
        )

    def log_decision(
        self,
        strategist_result: Dict[str, Any],
        macro_result: Optional[Dict[str, Any]] = None,
        risk_result: Optional[Dict[str, Any]] = None,
        semantic_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a decision event and return the stored entry."""
        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": strategist_result.get("symbol"),
            "entry_signal": strategist_result.get("entry_signal"),
            "trigger": strategist_result.get("trigger"),
            "confidence": strategist_result.get("confidence"),
            "reason": strategist_result.get("reason"),
            "phase_context": strategist_result.get("phase_context"),
            "macro_bias": macro_result.get("bias") if macro_result else None,
            "macro_reason": macro_result.get("reason") if macro_result else None,
            "risk_level": risk_result.get("risk") if risk_result else None,
            "volatility": risk_result.get("volatility") if risk_result else None,
            "semantic_summary": (
                semantic_result.get("combined_interpretation")
                if semantic_result
                else None
            ),
            "semantic_bias": (
                semantic_result.get("macro_bias") if semantic_result else None
            ),
            "notes": [],
        }
        entry["summary"] = self.summary(entry)
        self.entries.append(entry)
        return entry

    def get_latest_entry(self) -> Optional[Dict[str, Any]]:
        """Return the most recent journal entry or ``None``."""
        return self.entries[-1] if self.entries else None

    def export_log(self) -> List[Dict[str, Any]]:
        """Return the full list of logged entries."""
        return self.entries

