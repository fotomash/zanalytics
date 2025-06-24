# Zanzibar v5.2 Core Module
# Version: 5.2.1 (Updated Resampling Logic)
# Module: data_pipeline.py
# Author: Captain Zanzibar Crew
# --------------------------------------------------
# Unified data-flow engine: fetch M1, fetch macro, resample to HTFs.
# Now uses core/resample_m1_to_htf.py for standardization.
# Assumes finnhub_data_fetcher is modified to SAVE raw M1 CSVs.
# --------------------------------------------------

# --- Zanzibar dynamic version tag and startup timestamp ---
from datetime import datetime
VERSION = "5.2.1"
STARTED_AT = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
print(f"[DataPipeline] Zanzibar {VERSION} initialized at {STARTED_AT} UTC")

from concurrent.futures import ThreadPoolExecutor  # Still useful for fetching
import os
import glob
import pandas as pd
from datetime import datetime, timezone, timedelta  # Added timezone, timedelta
import traceback  # For detailed error logging
from pathlib import Path  # Added Path for robust directory handling

# Unified data access layer
from core.data_manager import DataManager

# Local imports (massive_macro_fetcher is optional)
try:
    from core import massive_macro_fetcher
    MACRO_FETCHER_AVAILABLE = True
except ImportError:
    print("[WARN][DataPipeline] massive_macro_fetcher.py not found. Macro fetching disabled.")
    massive_macro_fetcher = None
    MACRO_FETCHER_AVAILABLE = False


# ----------------------------------------------------------------------------
# Helper: simple timestamped logger
# ----------------------------------------------------------------------------

def _log(msg: str):
    """Logs messages with a timestamp."""
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DataPipeline] {stamp} | {msg}")

# ----------------------------------------------------------------------------
# DataPipeline class
# ----------------------------------------------------------------------------

class DataPipeline:
    """Unified fetch + resample orchestrator."""

    def __init__(self, symbols_m1=None, macro_interval="5m", workers: int = 6, m1_fetch_days=5):
        """
        Initializes the DataPipeline.

        Args:
            symbols_m1 (list, optional): List of symbols for M1 fetching. Defaults to common FX/Indices.
            macro_interval (str, optional): Interval for macro asset fetching. Defaults to "5m".
            workers (int, optional): Number of parallel workers for fetching (if applicable). Defaults to 6.
            m1_fetch_days (int, optional): How many days back to fetch M1 data. Defaults to 5.
        """
        self.symbols_m1 = symbols_m1 or ["OANDA:EUR_USD", "OANDA:GBP_USD", "OANDA:XAU_USD"] # Updated defaults
        self.macro_interval = macro_interval
        self.workers = workers
        self.m1_fetch_days = m1_fetch_days
        # Define base paths using pathlib
        self.base_dir = Path(".") # Assuming script runs from project root
        self.log_dir = self.base_dir / "logs"
        self.m1_data_dir = self.base_dir / "tick_data" / "m1"
        self.ohlcv_data_dir = self.base_dir / "ohlcv_data"
        self.macro_data_dir = self.base_dir / "intel_data" / "macro" # Define macro dir

        # Ensure necessary directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.m1_data_dir.mkdir(parents=True, exist_ok=True)
        self.ohlcv_data_dir.mkdir(parents=True, exist_ok=True)
        self.macro_data_dir.mkdir(parents=True, exist_ok=True) # Ensure macro dir exists

        # Data manager handles fetching and resampling
        self.data_manager = DataManager(m1_dir=str(self.m1_data_dir))

    # --------------------------------------
    # 1. Fetch intraday pairs (M1) - Using new fetcher
    # --------------------------------------
    def fetch_pairs(self):
        """Fetch and cache M1 data for configured symbols using DataManager."""
        _log(f"Fetching M1 pairs for symbols: {self.symbols_m1}")
        successful_fetches = 0
        for symbol in self.symbols_m1:
            try:
                df = self.data_manager.get_data(symbol, 'm1', days_back=self.m1_fetch_days)
                if not df.empty:
                    successful_fetches += 1
                    safe_symbol = symbol.replace(':', '_').replace('/', '_')
                    filename = f"{safe_symbol}_M1_{datetime.utcnow():%Y%m%d%H%M}.csv"
                    filepath = self.m1_data_dir / filename
                    df.to_csv(filepath)
                    _log(f"Saved raw M1 data to {filepath}")
                else:
                    _log(f"No data returned for {symbol}")
            except Exception as e:
                _log(f"Exception during fetch for {symbol}: {e}")
                traceback.print_exc()

        _log(f"M1 Pair fetching complete. Successful fetches: {successful_fetches}/{len(self.symbols_m1)}")


    # --------------------------------------
    # 2. Fetch macro set (VIX, SPX…)
    # --------------------------------------
    def fetch_macro(self):
        """Fetches macro asset data using massive_macro_fetcher."""
        if not MACRO_FETCHER_AVAILABLE:
            _log("Skipping macro fetching: massive_macro_fetcher not available.")
            return

        _log(f"Fetching macro assets (Interval: {self.macro_interval})...")
        try:
            # Ensure massive_macro_fetcher saves to the correct directory
            # Modify massive_macro_fetcher.py if needed to use self.macro_data_dir
            # Check if the module has the attribute before setting it
            if hasattr(massive_macro_fetcher, 'MACRO_OUTPUT_DIR'):
                 massive_macro_fetcher.MACRO_OUTPUT_DIR = str(self.macro_data_dir) # Override if possible
            else:
                 _log(f"WARN: Cannot set MACRO_OUTPUT_DIR in massive_macro_fetcher. Using its default.")

            massive_macro_fetcher.batch_fetch_macro(interval=self.macro_interval)
            _log("Macro asset fetching complete.")
        except Exception as e:
            _log(f"Error during macro fetching: {e}")
            traceback.print_exc()

    # --------------------------------------
    # 3. Resample M1 to HTF (using standardized resampler)
    # --------------------------------------
    def resample_htf(self):
        """Resamples M1 data found in tick_data/m1/ to HTFs using resample_m1_to_htf."""
        _log(f"Resampling all M1 datasets found in {self.m1_data_dir} -> HTF...")
        try:
            self.data_manager.resample_csv_directory(
                m1_dir=str(self.m1_data_dir),
                output_dir=str(self.ohlcv_data_dir)
            )
            _log("HTF resampling complete.")
        except Exception as e:
            _log(f"Error during HTF resampling: {e}")
            traceback.print_exc()

    # --------------------------------------
    # 4. Run full chain
    # --------------------------------------
    def run_full(self):
        """Runs the full data pipeline: Fetch Pairs -> Fetch Macro -> Resample HTF."""
        _log("--- Starting Full Data Pipeline Run ---")
        self.fetch_pairs() # Assumes this step results in M1 CSVs being saved
        self.fetch_macro()
        self.resample_htf() # Resamples the saved M1 CSVs
        _log("--- DataPipeline Full Run Complete ✅ ---")

# ----------------------------------------------------------------------------
# CLI entry
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    _log("Initializing DataPipeline from CLI...")
    # Example: Customize symbols or fetch days if needed
    # pipeline = DataPipeline(symbols_m1=["OANDA:EUR_USD", "BINANCE:BTCUSDT"], m1_fetch_days=7)
    pipeline = DataPipeline()
    pipeline.run_full()
