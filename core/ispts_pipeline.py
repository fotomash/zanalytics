"""ISPTS Pipeline
-----------------
Deterministic pipeline combining Wyckoff phase detection,
SMC structure analysis, inducement sweep scanning and
microstructure gating. Configuration is provided via
`AgentProfileSchema` YAML files for reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from schemas.agent_profile_schemas import AgentProfileSchema
from core.phase_detector_wyckoff_v1 import detect_wyckoff_multi_tf
from market_structure_analyzer_smc import analyze_market_structure
from liquidity_detector import detect_swing_highs_lows
from core.tick_processor import ticks_to_bars


class ISPTSPipeline:
    """Simplified deterministic ISPTS pipeline."""

    def __init__(self, yaml_path: str):
        cfg_dict = yaml.safe_load(Path(yaml_path).read_text())
        self.profile = AgentProfileSchema(**cfg_dict)
        self.events: List[Dict[str, Any]] = []

    def _log(self, report_type: str, data: Dict[str, Any]):
        self.events.append({
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "report_type": report_type,
            "data": data,
        })

    def run(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame, tick_df: pd.DataFrame) -> Dict[str, Any]:
        """Execute pipeline using provided data frames."""
        results: Dict[str, Any] = {}

        # Wyckoff phase detection
        wyckoff_input = {"H1": htf_df}
        phase_cfg = getattr(self.profile, "phase_detection_config", {})
        wyckoff_res = detect_wyckoff_multi_tf(wyckoff_input, config=phase_cfg)
        results["wyckoff"] = wyckoff_res
        self._log("WYCKOFF_PHASE", wyckoff_res)

        # SMC structure analysis
        smc_cfg = getattr(self.profile, "structure_validator", {})
        swing_n = 5
        if smc_cfg and hasattr(smc_cfg, "swing_engine_config"):
            swing_n = smc_cfg.swing_engine_config.swing_n
        smc_res = analyze_market_structure(htf_df, swing_n=swing_n)
        results["smc_structure"] = smc_res
        self._log("SMC_STRUCTURE", {"points": len(smc_res.get("structure_points", []))})

        # Inducement / liquidity sweep detection on lower timeframe
        lookback = 3
        le_cfg = getattr(self.profile, "liquidity_engine", None)
        if le_cfg:
            lookback = le_cfg.min_lookback or lookback
        sweeps = detect_swing_highs_lows(ltf_df.to_dict("records"), lookback=lookback)
        results["inducement"] = sweeps
        self._log("LIQUIDITY_SWEEPS", {"count": len(sweeps)})

        # Microstructure gating using ticks
        de_cfg = getattr(self.profile, "data_enricher_config", None)
        if de_cfg and de_cfg.tick_context_settings and de_cfg.tick_context_settings.enable_tick_merge:
            bars = ticks_to_bars(tick_df, timeframe="1T")
            results["micro_gating"] = bars.tail(1).to_dict("records")
            self._log("MICRO_GATING", {"bars": len(bars)})
        else:
            results["micro_gating"] = None

        return results

    def save_events(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.events, indent=2))
