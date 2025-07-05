"""Simple backtesting engine for Zanzibar strategies.

This module iterates through historical OHLCV data and feeds
bars to strategy orchestrators. Signals from ``entry_executor_smc.py``
are used to open and close virtual positions. Basic slippage and
commission models are supported. Standard performance metrics are
calculated after the simulation and saved to ``backtesting/results``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path when executed directly
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import orchestrator lazily to avoid heavy imports when not running tests
try:
    from core.advanced_smc_orchestrator import run_advanced_smc_strategy
except Exception:  # pragma: no cover - orchestrator may not be present
    run_advanced_smc_strategy = None


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    size: float
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None

    def is_long(self) -> bool:
        return self.direction.lower() == "buy"


@dataclass
class Portfolio:
    cash: float
    commission: float = 0.0
    slippage: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    open_trades: List[Trade] = field(default_factory=list)
    equity_curve: List[tuple[pd.Timestamp, float]] = field(default_factory=list)

    def open_trade(self, trade: Trade) -> None:
        self.cash -= self.commission
        if trade.is_long():
            trade.entry_price += self.slippage
        else:
            trade.entry_price -= self.slippage
        self.open_trades.append(trade)

    def close_trade(self, trade: Trade, price: float, time: pd.Timestamp) -> None:
        self.cash -= self.commission
        trade.exit_time = time
        if trade.is_long():
            trade.exit_price = price - self.slippage
            pnl = (trade.exit_price - trade.entry_price) * trade.size
        else:
            trade.exit_price = price + self.slippage
            pnl = (trade.entry_price - trade.exit_price) * trade.size
        self.cash += pnl
        self.open_trades.remove(trade)
        self.trades.append(trade)

    def update_equity(self, time: pd.Timestamp, price_lookup: Dict[str, float]) -> None:
        equity = self.cash
        for t in self.open_trades:
            price = price_lookup.get(t.symbol)
            if price is None:
                continue
            if t.is_long():
                unrealized = (price - t.entry_price) * t.size
            else:
                unrealized = (t.entry_price - price) * t.size
            equity += unrealized
        self.equity_curve.append((time, equity))

    def close_all(self, time: pd.Timestamp, price_lookup: Dict[str, float]) -> None:
        for t in list(self.open_trades):
            price = price_lookup.get(t.symbol, t.entry_price)
            self.close_trade(t, price, time)
        self.update_equity(time, price_lookup)


# --- Metric utilities -----------------------------------------------------

def compute_metrics(equity: pd.Series) -> Dict[str, float]:
    pnl = equity.iloc[-1] - equity.iloc[0]
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_drawdown = drawdown.min()
    returns = equity.pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if not returns.empty else np.nan
    return {
        "pnl": float(pnl),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
    }


# --- Data loading ---------------------------------------------------------

def load_symbol_data(
    data_dir: Path,
    symbol: str,
    timeframes: List[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Dict[str, pd.DataFrame]:
    data = {}
    for tf in timeframes:
        fname = f"{symbol}_{tf}.csv"
        path = data_dir / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        df.index = pd.to_datetime(df.index)
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        data[tf] = df
    return data


# --- Backtest runner ------------------------------------------------------

def run_backtest(
    symbols: List[str],
    data_dir: Path,
    start: Optional[str],
    end: Optional[str],
    variant: str = "default",
    timeframes: Optional[List[str]] = None,
    initial_balance: float = 10000.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    results_dir: Path = Path("backtesting/results"),
) -> Dict[str, Dict[str, float]]:
    if run_advanced_smc_strategy is None:
        raise ImportError("advanced_smc_orchestrator not available")

    if timeframes is None:
        timeframes = ["m1"]

    start_dt = pd.to_datetime(start) if start else None
    end_dt = pd.to_datetime(end) if end else None

    results_dir.mkdir(parents=True, exist_ok=True)

    symbol_data = {
        sym: load_symbol_data(data_dir, sym, timeframes, start_dt, end_dt)
        for sym in symbols
    }

    # Determine iteration timestamps from execution timeframe of first symbol
    exec_tf = timeframes[0]
    timestamps = None
    for data in symbol_data.values():
        df = data.get(exec_tf)
        if df is not None:
            timestamps = df.index if timestamps is None else timestamps.union(df.index)
    if timestamps is None:
        raise ValueError("No data found for given symbols/timeframes")
    timestamps = timestamps.sort_values()

    portfolio = Portfolio(cash=initial_balance, commission=commission, slippage=slippage)

    for ts in timestamps:
        price_lookup = {}
        for symbol, tf_data in symbol_data.items():
            df = tf_data.get(exec_tf)
            if df is None or ts not in df.index:
                continue
            bar = df.loc[ts]
            price_lookup[symbol] = bar["Close"]

            # Check exits first
            for trade in list(portfolio.open_trades):
                if trade.symbol != symbol:
                    continue
                if trade.is_long():
                    hit_sl = bar["Low"] <= trade.sl
                    hit_tp = bar["High"] >= trade.tp
                else:
                    hit_sl = bar["High"] >= trade.sl
                    hit_tp = bar["Low"] <= trade.tp
                if hit_sl:
                    portfolio.close_trade(trade, trade.sl, ts)
                elif hit_tp:
                    portfolio.close_trade(trade, trade.tp, ts)

            # Build slice for orchestrator
            data_slice = {
                tf: df_tf[df_tf.index <= ts] for tf, df_tf in tf_data.items()
            }
            if not all(tf in data_slice and not data_slice[tf].empty for tf in timeframes):
                continue
            orch_res = run_advanced_smc_strategy(
                all_tf_data=data_slice,
                strategy_variant=variant,
                target_timestamp=ts,
                symbol=symbol,
            )
            entry = orch_res.get("final_entry_result")
            if entry and entry.get("entry_confirmed"):
                trade = Trade(
                    symbol=symbol,
                    direction=entry.get("direction", "buy"),
                    entry_price=entry.get("entry_price"),
                    sl=entry.get("sl"),
                    tp=entry.get("tp"),
                    size=entry.get("lot_size", 1.0),
                    entry_time=ts,
                )
                portfolio.open_trade(trade)

        portfolio.update_equity(ts, price_lookup)

    # Close remaining trades at final prices
    final_prices = {
        sym: df[exec_tf]["Close"].iloc[-1]
        for sym, df in symbol_data.items()
        if exec_tf in df and not df[exec_tf].empty
    }
    if timestamps.any():
        portfolio.close_all(timestamps[-1], final_prices)

    equity_series = pd.Series(
        [eq for _, eq in portfolio.equity_curve],
        index=[ts for ts, _ in portfolio.equity_curve],
    )
    metrics = compute_metrics(equity_series)

    # Save outputs
    equity_series.to_csv(results_dir / "equity.csv")
    trades_df = pd.DataFrame([t.__dict__ for t in portfolio.trades])
    trades_df.to_csv(results_dir / "trades.csv", index=False)
    pd.Series(metrics).to_json(results_dir / "metrics.json")

    return metrics


# --- CLI interface --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with CSV files")
    parser.add_argument("--symbols", required=True, help="Comma separated list of symbols")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--variant", default="default", help="Strategy variant name")
    parser.add_argument("--timeframes", default="m1", help="Comma separated timeframes (e.g. m1,h1)")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--results-dir", type=Path, default=Path("backtesting/results"))
    args = parser.parse_args()

    metrics = run_backtest(
        symbols=[s.strip() for s in args.symbols.split(",")],
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        variant=args.variant,
        timeframes=[tf.strip() for tf in args.timeframes.split(",")],
        initial_balance=args.initial_balance,
        commission=args.commission,
        slippage=args.slippage,
        results_dir=args.results_dir,
    )
    print("Backtest complete:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
