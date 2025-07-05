from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, Extra


class CommonConfig:
    orm_mode = True
    validate_assignment = True
    extra = Extra.ignore


class TrendMetrics(BaseModel):
    """Basic trend indicators."""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None

    class Config(CommonConfig):
        title = "TrendMetrics"


class MomentumMetrics(BaseModel):
    """Momentum oscillator metrics."""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_diff: Optional[float] = None

    class Config(CommonConfig):
        title = "MomentumMetrics"


class StatisticalMetrics(BaseModel):
    """General statistical indicators."""
    atr: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None

    class Config(CommonConfig):
        title = "StatisticalMetrics"


class WyckoffMetrics(BaseModel):
    """Micro Wyckoff analytics."""
    accumulation: Optional[bool] = None
    distribution: Optional[bool] = None
    spring: Optional[bool] = None
    upthrust: Optional[bool] = None
    effort: Optional[float] = None
    result: Optional[float] = None
    no_demand: Optional[bool] = None
    no_supply: Optional[bool] = None
    spread: Optional[float] = None
    spread_ma: Optional[float] = None
    volume_ma: Optional[float] = None
    vs_ratio: Optional[float] = None
    vs_ratio_ma: Optional[float] = None

    class Config(CommonConfig):
        title = "WyckoffMetrics"


class SMCMetrics(BaseModel):
    """Smart Money Concepts features."""
    order_block_bull: Optional[bool] = None
    order_block_bear: Optional[bool] = None
    fvg_bull: Optional[bool] = None
    fvg_bear: Optional[bool] = None
    liquidity_high: Optional[float] = None
    liquidity_low: Optional[float] = None
    bos_bull: Optional[bool] = None
    bos_bear: Optional[bool] = None

    class Config(CommonConfig):
        title = "SMCMetrics"


class UnifiedAnalyticsBar(BaseModel):
    """Standardized analytics object for a single bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Derived order-flow metrics
    bar_delta: Optional[int] = None
    poc_price: Optional[float] = None
    poi_price: Optional[float] = None
    bid_volume_total: Optional[int] = None
    ask_volume_total: Optional[int] = None

    trend: Optional[TrendMetrics] = None
    momentum: Optional[MomentumMetrics] = None
    statistical: Optional[StatisticalMetrics] = None
    wyckoff: Optional[WyckoffMetrics] = None
    smc: Optional[SMCMetrics] = None

    class Config(CommonConfig):
        title = "UnifiedAnalyticsBar"

    @classmethod
    def from_zbar(cls, zbar: "ZBar") -> "UnifiedAnalyticsBar":
        """Create a UnifiedAnalyticsBar from a ZBar dataclass."""
        return cls(
            timestamp=zbar.timestamp,
            open=zbar.open,
            high=zbar.high,
            low=zbar.low,
            close=zbar.close,
            volume=zbar.volume,
            bar_delta=getattr(zbar, "bar_delta", None),
            poc_price=getattr(zbar, "poc_price", None),
            poi_price=getattr(zbar, "poi_price", None),
            bid_volume_total=getattr(zbar, "bid_volume_total", None),
            ask_volume_total=getattr(zbar, "ask_volume_total", None),
        )

    @classmethod
    def from_series(cls, row: "pd.Series") -> "UnifiedAnalyticsBar":
        """Create a UnifiedAnalyticsBar from a pandas Series."""
        import pandas as pd  # Local import to avoid heavy dependency at module load

        if not isinstance(row, pd.Series):
            row = pd.Series(row)

        trend = TrendMetrics(
            sma_20=row.get("sma_20"),
            sma_50=row.get("sma_50"),
            sma_200=row.get("sma_200"),
            ema_20=row.get("ema_20"),
            ema_50=row.get("ema_50"),
        )

        momentum = MomentumMetrics(
            rsi=row.get("rsi") or row.get("rsi_14"),
            macd=row.get("macd"),
            macd_signal=row.get("macd_signal"),
            macd_diff=row.get("macd_diff"),
        )

        statistical = StatisticalMetrics(
            atr=row.get("atr"),
            bb_upper=row.get("bb_upper"),
            bb_middle=row.get("bb_middle"),
            bb_lower=row.get("bb_lower"),
            bb_width=row.get("bb_width"),
        )

        wyckoff = WyckoffMetrics(
            accumulation=row.get("wyckoff_accumulation"),
            distribution=row.get("wyckoff_distribution"),
            spring=row.get("wyckoff_spring"),
            upthrust=row.get("wyckoff_upthrust"),
            effort=row.get("wyckoff_effort"),
            result=row.get("wyckoff_result"),
            no_demand=row.get("wyckoff_no_demand"),
            no_supply=row.get("wyckoff_no_supply"),
            spread=row.get("wyckoff_spread"),
            spread_ma=row.get("wyckoff_spread_ma"),
            volume_ma=row.get("wyckoff_volume_ma"),
            vs_ratio=row.get("wyckoff_vs_ratio"),
            vs_ratio_ma=row.get("wyckoff_vs_ratio_ma"),
        )

        smc = SMCMetrics(
            order_block_bull=row.get("order_block_bull"),
            order_block_bear=row.get("order_block_bear"),
            fvg_bull=row.get("fvg_bull"),
            fvg_bear=row.get("fvg_bear"),
            liquidity_high=row.get("liquidity_high"),
            liquidity_low=row.get("liquidity_low"),
            bos_bull=row.get("bos_bull"),
            bos_bear=row.get("bos_bear"),
        )

        return cls(
            timestamp=pd.to_datetime(row.get("timestamp", row.get("index"))),
            open=float(row.get("open", row.get("Open", 0.0))),
            high=float(row.get("high", row.get("High", 0.0))),
            low=float(row.get("low", row.get("Low", 0.0))),
            close=float(row.get("close", row.get("Close", 0.0))),
            volume=float(row.get("volume", row.get("Volume", 0.0))),
            bar_delta=row.get("bar_delta"),
            poc_price=row.get("poc_price"),
            poi_price=row.get("poi_price"),
            bid_volume_total=row.get("bid_volume_total"),
            ask_volume_total=row.get("ask_volume_total"),
            trend=trend,
            momentum=momentum,
            statistical=statistical,
            wyckoff=wyckoff,
            smc=smc,
        )

