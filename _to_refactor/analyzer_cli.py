"""
Command-line interface for the unified market microstructure analyzer.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import toml

from unified_microstructure_analyzer import AnalyzerConfig, UnifiedAnalyzer
from visualization import MarketVisualization


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Market Microstructure Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output arguments
    parser.add_argument(
        '--tick-data',
        type=str,
        nargs='+',
        help='Paths to tick data CSV files'
    )
    parser.add_argument(
        '--bar-data',
        type=str,
        nargs='+',
        help='Paths to bar data CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )

    # Processing arguments
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['XAUUSD'],
        help='Symbols to process (default: XAUUSD)'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        help='Maximum number of rows to process per file'
    )
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable duplicate removal'
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration TOML file'
    )

    # Features
    parser.add_argument(
        '--disable-smc',
        action='store_true',
        help='Disable SMC analysis'
    )
    parser.add_argument(
        '--disable-wyckoff',
        action='store_true',
        help='Disable Wyckoff analysis'
    )
    parser.add_argument(
        '--disable-order-flow',
        action='store_true',
        help='Disable order flow analysis'
    )
    parser.add_argument(
        '--disable-ml',
        action='store_true',
        help='Disable ML feature generation'
    )

    # Visualization
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--plot-types',
        type=str,
        nargs='+',
        choices=['price', 'order_flow', 'wyckoff', 'features', 'all'],
        default=['all'],
        help='Types of plots to generate'
    )

    # Other
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, 'r') as f:
        return toml.load(f)


def build_config(args: argparse.Namespace) -> AnalyzerConfig:
    """Build configuration from arguments and config file."""
    config_dict = {}

    # Load from config file if provided
    if args.config:
        config_dict = load_config_from_file(args.config)

    # Override with command-line arguments
    if args.tick_data:
        config_dict['tick_data_paths'] = [Path(p) for p in args.tick_data]

    if args.bar_data:
        config_dict['bar_data_paths'] = [Path(p) for p in args.bar_data]

    if args.output_dir:
        config_dict['output_dir'] = Path(args.output_dir)

    if args.symbols:
        config_dict['symbols'] = args.symbols

    if args.max_rows is not None:
        config_dict['max_rows'] = args.max_rows

    config_dict['remove_duplicates'] = not args.no_dedup

    # Features
    if args.disable_smc:
        config_dict['enable_smc'] = False

    if args.disable_wyckoff:
        config_dict['enable_wyckoff'] = False

    if args.disable_order_flow:
        config_dict['enable_order_flow'] = False

    if args.disable_ml:
        config_dict['enable_ml_features'] = False

    return AnalyzerConfig.from_dict(config_dict)


def create_visualizations(results: dict, config: AnalyzerConfig, 
                         plot_types: List[str]) -> None:
    """Create visualization plots."""
    logger = logging.getLogger(__name__)
    logger.info("Creating visualizations...")

    viz = MarketVisualization(config.output_dir / 'plots')

    # Load processed data
    tick_df = None
    bar_df = None

    if 'tick_data' in results:
        import pandas as pd
        tick_df = pd.read_parquet(results['tick_data'])

    if 'bar_data' in results:
        import pandas as pd
        bar_df = pd.read_parquet(results['bar_data'])

    # Create requested plots
    if 'all' in plot_types or 'price' in plot_types:
        if bar_df is not None:
            viz.plot_price_with_structure(bar_df)
            logger.info("Created price structure plot")

    if 'all' in plot_types or 'order_flow' in plot_types:
        if tick_df is not None and 'cumulative_delta' in tick_df.columns:
            viz.plot_order_flow_analysis(tick_df)
            logger.info("Created order flow plot")

    if 'all' in plot_types or 'wyckoff' in plot_types:
        if bar_df is not None and 'wyckoff_phase' in bar_df.columns:
            viz.plot_wyckoff_analysis(bar_df)
            logger.info("Created Wyckoff analysis plot")

    if 'all' in plot_types or 'features' in plot_types:
        if bar_df is not None:
            # Get ML feature columns
            feature_cols = [col for col in bar_df.columns 
                          if any(x in col for x in ['return_', 'rsi_', 'macd', 'stoch_'])]
            if feature_cols:
                viz.plot_ml_feature_importance(bar_df, feature_cols)
                logger.info("Created feature importance plot")

    # Create dashboard
    report_path = results.get('report')
    dashboard_path = viz.create_analysis_dashboard(tick_df, bar_df, report_path)
    logger.info(f"Created analysis dashboard: {dashboard_path}")


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        # Build configuration
        config = build_config(args)

        # Validate inputs
        if not config.tick_data_paths and not config.bar_data_paths:
            logger.error("No input data specified. Use --tick-data or --bar-data")
            sys.exit(1)

        # Create and run analyzer
        logger.info("Starting market microstructure analysis...")
        analyzer = UnifiedAnalyzer(config)
        results = analyzer.run_analysis()

        # Report results
        logger.info("Analysis complete!")
        for data_type, output_path in results.items():
            logger.info(f"{data_type}: {output_path}")

        # Create visualizations if requested
        if args.visualize:
            create_visualizations(results, config, args.plot_types)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
