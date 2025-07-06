from __future__ import annotations

"""Meta agent for analyzing journal logs and suggesting optimizations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import json
import pandas as pd


@dataclass
class OptimizationReport:
    """Summary of optimization metrics."""

    confluence_win_rates: Dict[str, float]
    rejection_effectiveness: Optional[float]
    suggestions: Dict[str, float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "confluence_win_rates": self.confluence_win_rates,
                "rejection_effectiveness": self.rejection_effectiveness,
                "suggestions": self.suggestions,
            },
            indent=2,
        )

    def to_markdown(self) -> str:
        lines = ["# Optimization Meta Report", ""]
        lines.append("## Confluence Path Win Rates")
        for path, rate in self.confluence_win_rates.items():
            lines.append(f"- **{path}**: {rate:.2%}")
        lines.append("")
        lines.append(
            f"**Rejection Effectiveness:** {self.rejection_effectiveness:.2%}"
            if self.rejection_effectiveness is not None
            else "**Rejection Effectiveness:** N/A"
        )
        lines.append("")
        lines.append("## Suggested Parameter Adjustments")
        for key, val in self.suggestions.items():
            lines.append(f"- **{key}**: {val}")
        return "\n".join(lines)


class OptimizationMetaAgent:
    """Reads journal entries and computes optimization metrics."""

    def __init__(self, journal_dir: str = "journal") -> None:
        self.journal_dir = Path(journal_dir)
        self.trade_log_path = self.journal_dir / "trade_log.csv"

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_trades(self) -> pd.DataFrame:
        if self.trade_log_path.is_file():
            try:
                return pd.read_csv(self.trade_log_path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Metric calculations
    # ------------------------------------------------------------------
    def _compute_confluence_win_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        rates: Dict[str, float] = {}
        if df.empty or "confluence_path" not in df.columns or "result" not in df.columns:
            return rates
        for _, row in df.iterrows():
            path = row["confluence_path"]
            if isinstance(path, str):
                path = path.strip()
                if path.startswith("[") and path.endswith("]"):
                    try:
                        path_list = json.loads(path.replace("'", '"'))
                        path = "->".join(path_list)
                    except Exception:
                        try:
                            import ast

                            path_list = ast.literal_eval(path)
                            if isinstance(path_list, list):
                                path = "->".join([str(p) for p in path_list])
                        except Exception:
                            pass
            win = str(row["result"]).lower() == "win"
            entry = rates.setdefault(path, {"wins": 0, "total": 0})
            entry["total"] += 1
            if win:
                entry["wins"] += 1
        return {k: v["wins"] / v["total"] if v["total"] else 0.0 for k, v in rates.items()}

    def _compute_rejection_effectiveness(self, df: pd.DataFrame) -> Optional[float]:
        if df.empty or "rejected" not in df.columns or "result" not in df.columns:
            return None
        rejected = df[df["rejected"] == True]
        if rejected.empty:
            return None
        losers = rejected[rejected["result"].str.lower() != "win"]
        return len(losers) / len(rejected)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def generate_report(self, as_markdown: bool = False) -> OptimizationReport | str:
        df = self._load_trades()
        confluence_rates = self._compute_confluence_win_rates(df)
        rejection_eff = self._compute_rejection_effectiveness(df)

        suggestions: Dict[str, float] = {}
        overall_win = (
            sum(r["wins"] for r in [
                {"wins": int(str(row["result"]).lower() == "win")}
                for _, row in df.iterrows()
            ]) / len(df)
            if not df.empty else 0
        )
        if overall_win < 0.5:
            suggestions["entry_threshold"] = 0.6
        if rejection_eff is not None and rejection_eff < 0.6:
            suggestions["rejection_sensitivity"] = 1.1

        report = OptimizationReport(
            confluence_win_rates=confluence_rates,
            rejection_effectiveness=rejection_eff,
            suggestions=suggestions,
        )
        return report.to_markdown() if as_markdown else report.to_json()
