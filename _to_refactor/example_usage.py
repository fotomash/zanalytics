#!/usr/bin/env python3
"""
Example usage of the Unified Market Microstructure Analyzer.
"""

import pandas as pd
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from unified_microstructure_analyzer import (
        AnalyzerConfig, UnifiedAnalyzer, 
        load_tick_data, load_bar_data, save_to_parquet
    )
except ImportError as e:
    print(f"Error: Could not import analyzer modules: {e}")
    print("Please ensure all files are in the same directory")
    sys.exit(1)


def main():
    """Run example analysis."""
    # Create configuration
    config = AnalyzerConfig(
        input_dir=Path("./data"),
        output_dir=Path("./output"),
        max_tick_rows=10000,
        max_bar_rows_per_timeframe=5000,
        enable_smc=True,
        enable_wyckoff=True,
        enable_order_flow=True,
        enable_advanced_tick_analysis=True,
        remove_duplicates=True
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = UnifiedAnalyzer(config)

    # Process tick data if available
    tick_files = list(config.input_dir.glob("*TICK*.csv"))
    if tick_files:
        print(f"\nFound {len(tick_files)} tick files")
        for tick_file in tick_files:
            print(f"Processing {tick_file.name}...")
            try:
                tick_df = load_tick_data(tick_file)
                processed_tick = analyzer.process_tick_data(tick_df)

                output_file = config.output_dir / f"{tick_file.stem}_analyzed.parquet"
                save_to_parquet(processed_tick, output_file)
                print(f"  ✓ Saved to {output_file}")

                # Show sample results
                print(f"  Processed {len(processed_tick)} tick records")
                if 'order_flow_trend' in processed_tick.columns:
                    trend_counts = processed_tick['order_flow_trend'].value_counts()
                    print(f"  Order flow: {trend_counts.to_dict()}")

            except Exception as e:
                print(f"  ✗ Error: {e}")

    # Process bar data if available
    bar_files = list(config.input_dir.glob("*bars*.csv")) + list(config.input_dir.glob("*M1*.csv"))
    if bar_files:
        print(f"\nFound {len(bar_files)} bar files")
        for bar_file in bar_files:
            print(f"Processing {bar_file.name}...")
            try:
                bar_df = load_bar_data(bar_file)

                # Detect timeframe
                if 'M1' in bar_file.name:
                    timeframe = 'M1'
                elif 'M5' in bar_file.name:
                    timeframe = 'M5'
                else:
                    timeframe = 'M1'  # default

                processed_bar = analyzer.process_bar_data(bar_df, timeframe)

                output_file = config.output_dir / f"{bar_file.stem}_analyzed.parquet"
                save_to_parquet(processed_bar, output_file)
                print(f"  ✓ Saved to {output_file}")

                # Show sample results
                print(f"  Processed {len(processed_bar)} {timeframe} bars")
                if 'market_structure' in processed_bar.columns:
                    structure_counts = processed_bar['market_structure'].value_counts()
                    print(f"  Market structure: {structure_counts.to_dict()}")
                if 'smc_trend' in processed_bar.columns:
                    smc_counts = processed_bar['smc_trend'].value_counts()
                    print(f"  SMC trend: {smc_counts.to_dict()}")

            except Exception as e:
                print(f"  ✗ Error: {e}")

    if not tick_files and not bar_files:
        print("\nNo data files found in ./data directory")
        print("Please add CSV files with 'TICK' or 'bars' or 'M1' in the filename")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
