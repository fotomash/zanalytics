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

