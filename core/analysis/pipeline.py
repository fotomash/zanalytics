import pandas as pd
from typing import Dict, Any, Optional

from .smc_enrichment_engine import tag_smc_zones
from .macro_enrichment_engine import fetch_macro_context
from .marker_enrichment_engine import add_all_indicators
from .divergence_engine import add_rsi_divergence
from .mentfx_ici_engine import tag_mentfx_ici
from .liquidity_engine_smc import detect_inducement_from_structure
from .liquidity_sweep_detector import tag_liquidity_sweeps
from .liquidity_vwap_detector import LiquidityVWAPDetector
from .wick_liquidity_monitor import analyze_wick_structure

class EnrichmentPipeline:
    """Apply enrichment engines sequentially based on a config dictionary."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results: Dict[str, Any] = {}

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enriched = df.copy()

        cfg = self.config

        if cfg.get("smc") is not False:
            params = cfg.get("smc", {})
            df_enriched = tag_smc_zones(df_enriched, **params)

        if cfg.get("divergence"):
            df_enriched = add_rsi_divergence(df_enriched, config=cfg.get("divergence"))

        if cfg.get("mentfx"):
            df_enriched = tag_mentfx_ici(df_enriched, **cfg.get("mentfx", {}))

        if cfg.get("markers"):
            timeframe = cfg["markers"].get("timeframe", "Unknown")
            df_enriched = add_all_indicators(df_enriched, timeframe=timeframe, config=cfg["markers"])

        if cfg.get("liquidity_sweep"):
            params = cfg.get("liquidity_sweep", {})
            tf = params.get("timeframe", "Unknown")
            df_enriched = tag_liquidity_sweeps(df_enriched, tf=tf, config=params)

        if cfg.get("liquidity_vwap"):
            detector = LiquidityVWAPDetector(cfg.get("liquidity_vwap"))
            df_vwap = detector.detect_deviation_sweeps(df_enriched)
            df_enriched = df_enriched.join(df_vwap.drop(columns=[c for c in df_enriched.columns if c in df_vwap.columns], errors="ignore"))

        if cfg.get("liquidity_smc"):
            params = cfg.get("liquidity_smc", {})
            tf = params.get("timeframe", "Unknown")
            structure = params.get("structure_data", {})
            self.results["liquidity_smc"] = detect_inducement_from_structure(df_enriched, tf, structure)

        if cfg.get("wick_monitor"):
            file_path = cfg["wick_monitor"].get("path")
            if file_path:
                analyze_wick_structure(file_path)

        if cfg.get("macro"):
            asset = cfg["macro"].get("asset_symbol", "")
            self.results["macro"] = fetch_macro_context(asset)

        return df_enriched


def apply(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Convenience wrapper around :class:`EnrichmentPipeline`."""
    pipeline = EnrichmentPipeline(config)
    return pipeline.apply(df)
