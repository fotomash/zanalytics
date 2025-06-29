"""
Visualization components for market microstructure analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MarketVisualization:
    """Create visualizations for market analysis results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_price_with_structure(self, df: pd.DataFrame, save_name: str = "price_structure") -> Path:
        """Plot price with market structure overlays."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Price plot
        ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1)

        # Add EMAs if available
        for ema_col in [col for col in df.columns if col.startswith('ema_')]:
            ax1.plot(df.index, df[ema_col], label=ema_col.upper(), alpha=0.7)

        # Mark swing points
        if 'swing_high' in df.columns:
            swing_highs = df[df['swing_high']]
            ax1.scatter(swing_highs.index, swing_highs['high'], 
                       color='red', marker='v', s=100, label='Swing High')

        if 'swing_low' in df.columns:
            swing_lows = df[df['swing_low']]
            ax1.scatter(swing_lows.index, swing_lows['low'], 
                       color='green', marker='^', s=100, label='Swing Low')

        # Mark SMC patterns
        if 'bos_bullish' in df.columns:
            bos_bull = df[df['bos_bullish']]
            ax1.scatter(bos_bull.index, bos_bull['high'], 
                       color='lime', marker='o', s=150, label='Bullish BOS')

        if 'bos_bearish' in df.columns:
            bos_bear = df[df['bos_bearish']]
            ax1.scatter(bos_bear.index, bos_bear['low'], 
                       color='red', marker='o', s=150, label='Bearish BOS')

        ax1.set_title('Price Action with Market Structure', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Volume plot
        colors = ['green' if c > o else 'red' 
                 for c, o in zip(df['close'], df['open'])] if 'open' in df.columns else 'blue'
        ax2.bar(df.index, df['volume'], color=colors, alpha=0.7)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{save_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_order_flow_analysis(self, df: pd.DataFrame, save_name: str = "order_flow") -> Path:
        """Plot order flow analysis results."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Cumulative Delta
        if 'cumulative_delta' in df.columns:
            ax1 = axes[0]
            ax1.plot(df.index, df['cumulative_delta'], label='Cumulative Delta', 
                    color='blue', linewidth=2)
            ax1.fill_between(df.index, 0, df['cumulative_delta'], 
                           where=df['cumulative_delta'] > 0, color='green', alpha=0.3)
            ax1.fill_between(df.index, 0, df['cumulative_delta'], 
                           where=df['cumulative_delta'] < 0, color='red', alpha=0.3)
            ax1.set_title('Cumulative Delta', fontsize=14)
            ax1.set_ylabel('Delta', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()

        # Order Imbalance
        if 'order_imbalance' in df.columns:
            ax2 = axes[1]
            ax2.plot(df.index, df['order_imbalance'], label='Order Imbalance', 
                    color='purple', linewidth=1)
            ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Imbalance')
            ax2.axhline(y=-0.7, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title('Order Flow Imbalance', fontsize=14)
            ax2.set_ylabel('Imbalance', fontsize=12)
            ax2.set_ylim(-1, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        # Absorption Detection
        if 'absorption' in df.columns:
            ax3 = axes[2]
            absorption_points = df[df['absorption']]
            ax3.scatter(absorption_points.index, [1]*len(absorption_points), 
                       color='orange', s=50, label='Absorption')
            ax3.set_title('Absorption Detection', fontsize=14)
            ax3.set_ylabel('Detection', fontsize=12)
            ax3.set_ylim(0, 2)
            ax3.grid(True, alpha=0.3)
            ax3.legend()

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{save_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_wyckoff_analysis(self, df: pd.DataFrame, save_name: str = "wyckoff") -> Path:
        """Plot Wyckoff analysis results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]})

        # Price with Wyckoff phases
        ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1)

        # Color background by phase
        if 'wyckoff_phase' in df.columns:
            phase_colors = {
                'accumulation': 'green',
                'distribution': 'red',
                'none': 'gray'
            }

            for phase, color in phase_colors.items():
                phase_mask = df['wyckoff_phase'] == phase
                if phase_mask.any():
                    phase_regions = self._find_continuous_regions(phase_mask)
                    for start, end in phase_regions:
                        ax1.axvspan(start, end, alpha=0.2, color=color, label=phase)

        # Mark Wyckoff events
        if 'wyckoff_event' in df.columns:
            events = df[df['wyckoff_event'] != '']
            for idx, event in events.iterrows():
                ax1.annotate(event['wyckoff_event'], 
                           xy=(idx, event['close']),
                           xytext=(idx, event['close'] * 1.01),
                           fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

        ax1.set_title('Wyckoff Method Analysis', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Volume analysis
        if 'volume_analysis' in df.columns:
            volume_colors = {
                'high_volume_up': 'darkgreen',
                'high_volume_down': 'darkred',
                'low_volume': 'gray',
                'normal_volume': 'blue'
            }

            colors = [volume_colors.get(v, 'blue') for v in df['volume_analysis']]
            ax2.bar(df.index, df['volume'], color=colors, alpha=0.7)
        else:
            ax2.bar(df.index, df['volume'], color='blue', alpha=0.7)

        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{save_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_ml_feature_importance(self, df: pd.DataFrame, feature_cols: List[str], 
                                  save_name: str = "feature_importance") -> Path:
        """Plot ML feature importance based on correlation with returns."""
        if 'return_1' not in df.columns:
            return None

        # Calculate correlations
        correlations = {}
        for col in feature_cols:
            if col in df.columns and not df[col].isna().all():
                corr = df[col].corr(df['return_1'])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)

        if not correlations:
            return None

        # Sort by importance
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:20]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        features, importances = zip(*sorted_features)
        y_pos = np.arange(len(features))

        ax.barh(y_pos, importances, color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Absolute Correlation with Returns')
        ax.set_title('Feature Importance (Correlation Analysis)')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{save_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def create_analysis_dashboard(self, tick_df: Optional[pd.DataFrame] = None,
                                bar_df: Optional[pd.DataFrame] = None,
                                report_path: Optional[Path] = None) -> Path:
        """Create a comprehensive analysis dashboard."""
        # Create individual plots
        plots_created = []

        if bar_df is not None:
            # Price structure plot
            plot_path = self.plot_price_with_structure(bar_df)
            plots_created.append(('Price Structure', plot_path))

            # Wyckoff analysis
            if 'wyckoff_phase' in bar_df.columns:
                plot_path = self.plot_wyckoff_analysis(bar_df)
                plots_created.append(('Wyckoff Analysis', plot_path))

        if tick_df is not None:
            # Order flow analysis
            if 'cumulative_delta' in tick_df.columns:
                plot_path = self.plot_order_flow_analysis(tick_df)
                plots_created.append(('Order Flow', plot_path))

        # Create summary statistics plot
        if report_path and report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)

            self._plot_summary_statistics(report)
            plots_created.append(('Summary Statistics', 
                                self.output_dir / 'summary_statistics.png'))

        # Log created plots
        dashboard_info = {
            'created_at': datetime.now().isoformat(),
            'plots': [{'name': name, 'path': str(path)} for name, path in plots_created]
        }

        info_path = self.output_dir / 'dashboard_info.json'
        with open(info_path, 'w') as f:
            json.dump(dashboard_info, f, indent=2)

        return info_path

    def _find_continuous_regions(self, mask: pd.Series) -> List[Tuple[int, int]]:
        """Find continuous True regions in a boolean mask."""
        regions = []
        start = None

        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                regions.append((start, i-1))
                start = None

        if start is not None:
            regions.append((start, len(mask)-1))

        return regions

    def _plot_summary_statistics(self, report: Dict) -> None:
        """Plot summary statistics from analysis report."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Market metrics
        if 'market_metrics' in report:
            metrics = report['market_metrics']

            # Trend distribution
            if 'trend_distribution' in metrics:
                ax = axes[0, 0]
                trends = metrics['trend_distribution']
                ax.pie(trends.values(), labels=trends.keys(), autopct='%1.1f%%')
                ax.set_title('Market Trend Distribution')

            # Pattern counts
            if 'smc_pattern_counts' in metrics:
                ax = axes[0, 1]
                patterns = metrics['smc_pattern_counts']
                ax.bar(patterns.keys(), patterns.values())
                ax.set_title('SMC Pattern Counts')
                ax.set_xlabel('Pattern Type')
                ax.set_ylabel('Count')

        # SMC patterns summary
        if 'smc_patterns_summary' in report:
            summary = report['smc_patterns_summary']

            # Direction distribution
            if 'direction_distribution' in summary:
                ax = axes[1, 0]
                directions = summary['direction_distribution']
                ax.bar(directions.keys(), directions.values(), 
                      color=['green', 'red'])
                ax.set_title('Pattern Direction Distribution')
                ax.set_xlabel('Direction')
                ax.set_ylabel('Count')

            # Summary text
            ax = axes[1, 1]
            ax.axis('off')
            summary_text = f"Analysis Summary\n\n"
            summary_text += f"Total Patterns: {summary.get('total_patterns', 0)}\n"
            summary_text += f"Avg Pattern Strength: {summary.get('avg_pattern_strength', 0):.2f}\n"

            if 'market_metrics' in report:
                metrics = report['market_metrics']
                summary_text += f"\nAvg Volume: {metrics.get('avg_volume', 0):.0f}\n"
                summary_text += f"Price Range: {metrics.get('price_range', 0):.2f}\n"

            ax.text(0.1, 0.5, summary_text, fontsize=12, 
                   transform=ax.transAxes, verticalalignment='center')

        plt.suptitle('Market Analysis Summary', fontsize=16)
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / 'summary_statistics.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
