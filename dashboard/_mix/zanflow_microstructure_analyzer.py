#!/usr/bin/env python3
"""
XAUUSD Microstructure & Advanced Strategy Analysis Dashboard
Created for ZANFLOW v12 Trading System

This script analyzes tick data for:
1. Microstructure patterns and manipulation detection
2. Wyckoff analysis
3. Smart Money Concepts (SMC)
4. Inducement patterns
5. Rich market commentary

Author: AI Assistant for ZANFLOW v12
Date: 2025-06-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

class MicrostructureAnalyzer:
    def __init__(self, tick_file_path):
        """Initialize the analyzer with tick data"""
        self.tick_file_path = tick_file_path
        self.tick_df = None
        self.analysis_results = {}

    def load_data(self):
        """Load and preprocess tick data"""
        print("Loading tick data...")
        try:
            # Try tab-separated first, then comma-separated
            try:
                self.tick_df = pd.read_csv(self.tick_file_path, sep='\t')
            except:
                self.tick_df = pd.read_csv(self.tick_file_path)

            # Convert timestamp to datetime
            self.tick_df['datetime'] = pd.to_datetime(self.tick_df['timestamp'])

            # Calculate derived metrics
            self.tick_df['mid_price'] = (self.tick_df['bid'] + self.tick_df['ask']) / 2
            self.tick_df['price_change'] = self.tick_df['mid_price'].diff()
            self.tick_df['spread_pips'] = self.tick_df['spread_price'] * 100
            self.tick_df['volatility'] = self.tick_df['price_change'].abs()
            self.tick_df['ba_imbalance'] = (self.tick_df['ask'] - self.tick_df['bid']) / (self.tick_df['ask'] + self.tick_df['bid'])

            print(f"Successfully loaded {len(self.tick_df)} ticks")
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def detect_manipulation_patterns(self):
        """Detect various manipulation patterns in the tick data"""
        print("Detecting manipulation patterns...")

        # 1. Sudden spread widening
        spread_threshold = self.tick_df['spread_pips'].mean() + 2 * self.tick_df['spread_pips'].std()
        self.tick_df['spread_spike'] = self.tick_df['spread_pips'] > spread_threshold

        # 2. Price reversal after spread spike
        self.tick_df['price_reversal'] = False
        for i in range(3, len(self.tick_df)):
            if self.tick_df.iloc[i-3:i-1]['spread_spike'].any() and abs(self.tick_df.iloc[i]['price_change']) > 0.05:
                self.tick_df.loc[self.tick_df.index[i], 'price_reversal'] = True

        # 3. Stop hunts (sharp price movement followed by reversal)
        window_size = 10
        self.tick_df['stop_hunt'] = False
        for i in range(window_size, len(self.tick_df)):
            price_move = self.tick_df.iloc[i-window_size:i]['price_change'].sum()
            next_move = self.tick_df.iloc[i:i+5]['price_change'].sum() if i+5 < len(self.tick_df) else 0
            if abs(price_move) > 0.1 and price_move * next_move < 0:
                self.tick_df.loc[self.tick_df.index[i], 'stop_hunt'] = True

        # 4. Liquidity sweeps
        self.tick_df['liquidity_sweep'] = False
        for i in range(20, len(self.tick_df)):
            if i+10 < len(self.tick_df):
                pre_high = self.tick_df.iloc[i-20:i]['mid_price'].max()
                pre_low = self.tick_df.iloc[i-20:i]['mid_price'].min()
                curr_price = self.tick_df.iloc[i]['mid_price']
                next_price = self.tick_df.iloc[i+10]['mid_price']

                # High sweep or Low sweep
                if (curr_price > pre_high and next_price < pre_high) or \
                   (curr_price < pre_low and next_price > pre_low):
                    self.tick_df.loc[self.tick_df.index[i], 'liquidity_sweep'] = True

        # Store results
        self.analysis_results['spread_spikes'] = len(self.tick_df[self.tick_df['spread_spike']])
        self.analysis_results['price_reversals'] = len(self.tick_df[self.tick_df['price_reversal']])
        self.analysis_results['stop_hunts'] = len(self.tick_df[self.tick_df['stop_hunt']])
        self.analysis_results['liquidity_sweeps'] = len(self.tick_df[self.tick_df['liquidity_sweep']])

    def wyckoff_analysis(self):
        """Perform Wyckoff phase analysis"""
        print("Performing Wyckoff analysis...")

        self.tick_df['wyckoff_phase'] = np.nan
        window = 100

        if len(self.tick_df) > window:
            for i in range(window, len(self.tick_df), window//2):
                if i+window < len(self.tick_df):
                    segment = self.tick_df.iloc[i-window:i+window]

                    # Accumulation
                    if (segment['mid_price'].iloc[:window//2].mean() < segment['mid_price'].iloc[window//2:].mean() and
                        segment['volatility'].iloc[:window//2].mean() > segment['volatility'].iloc[window//2:].mean()):
                        self.tick_df.loc[self.tick_df.index[i-window//2:i+window//2], 'wyckoff_phase'] = 1

                    # Distribution
                    elif (segment['mid_price'].iloc[:window//2].mean() > segment['mid_price'].iloc[window//2:].mean() and
                          segment['volatility'].iloc[:window//2].mean() < segment['volatility'].iloc[window//2:].mean()):
                        self.tick_df.loc[self.tick_df.index[i-window//2:i+window//2], 'wyckoff_phase'] = 2

                    # Markup
                    elif (segment['mid_price'].diff().iloc[:window].mean() > 0 and
                          segment['mid_price'].diff().iloc[window:].mean() > 0):
                        self.tick_df.loc[self.tick_df.index[i-window//2:i+window//2], 'wyckoff_phase'] = 3

                    # Markdown
                    elif (segment['mid_price'].diff().iloc[:window].mean() < 0 and
                          segment['mid_price'].diff().iloc[window:].mean() < 0):
                        self.tick_df.loc[self.tick_df.index[i-window//2:i+window//2], 'wyckoff_phase'] = 4

        # Store results
        phases = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}
        self.analysis_results['wyckoff_phases'] = {}
        for phase in [1, 2, 3, 4]:
            count = len(self.tick_df[self.tick_df['wyckoff_phase'] == phase])
            self.analysis_results['wyckoff_phases'][phases[phase]] = count

    def smc_analysis(self):
        """Perform Smart Money Concepts analysis"""
        print("Performing SMC analysis...")

        # Resample to 1-minute for FVG detection
        minute_data = self.tick_df.resample('1Min', on='datetime').agg({
            'bid': 'first',
            'ask': 'first',
            'mid_price': 'first',
            'datetime': 'first'
        }).dropna()

        minute_data['fvg_up'] = False
        minute_data['fvg_down'] = False

        # Detect Fair Value Gaps
        if len(minute_data) > 3:
            for i in range(2, len(minute_data)):
                if minute_data['mid_price'].iloc[i-2] > minute_data['mid_price'].iloc[i]:
                    minute_data.iloc[i-1, minute_data.columns.get_loc('fvg_up')] = True

                if minute_data['mid_price'].iloc[i-2] < minute_data['mid_price'].iloc[i]:
                    minute_data.iloc[i-1, minute_data.columns.get_loc('fvg_down')] = True

        self.minute_data = minute_data
        self.analysis_results['bullish_fvgs'] = len(minute_data[minute_data['fvg_up'] == True])
        self.analysis_results['bearish_fvgs'] = len(minute_data[minute_data['fvg_down'] == True])

    def inducement_analysis(self):
        """Detect inducement patterns"""
        print("Performing inducement analysis...")

        self.tick_df['inducement'] = False
        self.tick_df['inducement_level'] = np.nan
        self.tick_df['inducement_type'] = ''

        window = 50
        if len(self.tick_df) > window:
            for i in range(window, len(self.tick_df)-window):
                pre_segment = self.tick_df.iloc[i-window:i]
                post_segment = self.tick_df.iloc[i:i+window]

                pre_high = pre_segment['mid_price'].max()
                pre_low = pre_segment['mid_price'].min()

                # High inducement
                if (self.tick_df.iloc[i]['mid_price'] > pre_high and 
                    post_segment['mid_price'].iloc[10:].mean() < self.tick_df.iloc[i]['mid_price']):
                    self.tick_df.loc[self.tick_df.index[i], 'inducement'] = True
                    self.tick_df.loc[self.tick_df.index[i], 'inducement_level'] = pre_high
                    self.tick_df.loc[self.tick_df.index[i], 'inducement_type'] = 'High'

                # Low inducement
                if (self.tick_df.iloc[i]['mid_price'] < pre_low and 
                    post_segment['mid_price'].iloc[10:].mean() > self.tick_df.iloc[i]['mid_price']):
                    self.tick_df.loc[self.tick_df.index[i], 'inducement'] = True
                    self.tick_df.loc[self.tick_df.index[i], 'inducement_level'] = pre_low
                    self.tick_df.loc[self.tick_df.index[i], 'inducement_type'] = 'Low'

        inducements = self.tick_df[self.tick_df['inducement']]
        self.analysis_results['high_inducements'] = len(inducements[inducements['inducement_type'] == 'High'])
        self.analysis_results['low_inducements'] = len(inducements[inducements['inducement_type'] == 'Low'])

    def create_dashboard(self):
        """Create the comprehensive analysis dashboard"""
        print("Creating dashboard...")

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 2, figure=fig)

        # 1. Main price chart with manipulation markers
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.tick_df['datetime'], self.tick_df['mid_price'], color='white', linewidth=1, label='Mid Price')

        # Mark manipulation patterns
        spread_spikes = self.tick_df[self.tick_df['spread_spike']]
        price_reversals = self.tick_df[self.tick_df['price_reversal']]
        stop_hunts = self.tick_df[self.tick_df['stop_hunt']]
        liquidity_sweeps = self.tick_df[self.tick_df['liquidity_sweep']]

        if not spread_spikes.empty:
            ax1.scatter(spread_spikes['datetime'], spread_spikes['mid_price'], 
                       color='yellow', s=50, marker='^', label='Spread Spike')
        if not price_reversals.empty:
            ax1.scatter(price_reversals['datetime'], price_reversals['mid_price'], 
                       color='red', s=50, marker='x', label='Price Reversal')
        if not stop_hunts.empty:
            ax1.scatter(stop_hunts['datetime'], stop_hunts['mid_price'], 
                       color='magenta', s=50, marker='o', label='Stop Hunt')
        if not liquidity_sweeps.empty:
            ax1.scatter(liquidity_sweeps['datetime'], liquidity_sweeps['mid_price'], 
                       color='cyan', s=50, marker='s', label='Liquidity Sweep')

        ax1.set_title('XAUUSD Price with Microstructure Manipulation Markers', fontsize=16)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Spread analysis
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.tick_df['datetime'], self.tick_df['spread_pips'], color='orange', linewidth=1)
        ax2.set_title('Spread Analysis (Pips)', fontsize=16)
        ax2.set_ylabel('Spread (Pips)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 3. Volatility
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.tick_df['datetime'], self.tick_df['volatility'].rolling(20).mean(), color='red', linewidth=1)
        ax3.set_title('Tick Volatility (20-tick Rolling Average)', fontsize=16)
        ax3.set_ylabel('Volatility', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 4. Bid-Ask imbalance
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(self.tick_df['datetime'], self.tick_df['ba_imbalance'].rolling(20).mean() * 10000, 
                color='purple', linewidth=1)
        ax4.set_title('Bid-Ask Imbalance (×10,000)', fontsize=16)
        ax4.set_ylabel('Imbalance', fontsize=12)
        ax4.grid(True, alpha=0.3)

        # 5. Price change distribution
        ax5 = fig.add_subplot(gs[2, 1])
        sns.histplot(self.tick_df['price_change'].dropna(), bins=50, kde=True, color='cyan', ax=ax5)
        ax5.set_title('Price Change Distribution (Microstructure)', fontsize=16)
        ax5.set_xlabel('Tick-to-Tick Price Change', fontsize=12)
        ax5.grid(True, alpha=0.3)

        # 6. Manipulation events timeline
        ax6 = fig.add_subplot(gs[3, :])
        events = []
        labels = []
        colors = []

        for idx, row in spread_spikes.iterrows():
            events.append((row['datetime'], 'Spread Spike', 'yellow'))
        for idx, row in price_reversals.iterrows():
            events.append((row['datetime'], 'Price Reversal', 'red'))
        for idx, row in stop_hunts.iterrows():
            events.append((row['datetime'], 'Stop Hunt', 'magenta'))
        for idx, row in liquidity_sweeps.iterrows():
            events.append((row['datetime'], 'Liquidity Sweep', 'cyan'))

        events.sort(key=lambda x: x[0])

        for i, (dt, event, color) in enumerate(events):
            ax6.scatter(dt, 0, color=color, s=100, marker='|')
            if i % 10 == 0:
                ax6.text(dt, 0.1, event, rotation=45, fontsize=8, color=color)

        ax6.set_title('Manipulation Events Timeline', fontsize=16)
        ax6.set_yticks([])
        ax6.grid(True, alpha=0.3)

        # 7. Wyckoff phases
        ax7 = fig.add_subplot(gs[4, 0])
        ax7.plot(self.tick_df['datetime'], self.tick_df['mid_price'], color='white', linewidth=1, alpha=0.5)

        colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}
        labels = {1: 'Accumulation', 2: 'Distribution', 3: 'Markup', 4: 'Markdown'}

        for phase in [1, 2, 3, 4]:
            phase_data = self.tick_df[self.tick_df['wyckoff_phase'] == phase]
            if not phase_data.empty:
                ax7.scatter(phase_data['datetime'], phase_data['mid_price'], 
                           color=colors[phase], label=labels[phase], alpha=0.7, s=10)

        ax7.set_title('Wyckoff Phase Analysis', fontsize=16)
        ax7.set_ylabel('Price', fontsize=12)
        ax7.legend(loc='upper left')
        ax7.grid(True, alpha=0.3)

        # 8. SMC analysis
        ax8 = fig.add_subplot(gs[4, 1])
        ax8.plot(self.minute_data['datetime'], self.minute_data['mid_price'], color='white', linewidth=1, alpha=0.7)

        fvg_up = self.minute_data[self.minute_data['fvg_up'] == True]
        fvg_down = self.minute_data[self.minute_data['fvg_down'] == True]

        if not fvg_up.empty:
            ax8.scatter(fvg_up['datetime'], fvg_up['mid_price'], color='green', s=50, marker='^', label='Bullish FVG')
        if not fvg_down.empty:
            ax8.scatter(fvg_down['datetime'], fvg_down['mid_price'], color='red', s=50, marker='v', label='Bearish FVG')

        ax8.set_title('Smart Money Concepts - Fair Value Gaps', fontsize=16)
        ax8.set_ylabel('Price', fontsize=12)
        ax8.legend(loc='upper left')
        ax8.grid(True, alpha=0.3)

        # 9. Inducement analysis
        ax9 = fig.add_subplot(gs[5, :])
        ax9.plot(self.tick_df['datetime'], self.tick_df['mid_price'], color='white', linewidth=1)

        inducements = self.tick_df[self.tick_df['inducement']]
        high_inducements = inducements[inducements['inducement_type'] == 'High']
        low_inducements = inducements[inducements['inducement_type'] == 'Low']

        if not high_inducements.empty:
            ax9.scatter(high_inducements['datetime'], high_inducements['mid_price'], 
                       color='red', s=80, marker='v', label='High Inducement')
        if not low_inducements.empty:
            ax9.scatter(low_inducements['datetime'], low_inducements['mid_price'], 
                       color='green', s=80, marker='^', label='Low Inducement')

        for idx, row in inducements.iterrows():
            ax9.axhline(y=row['inducement_level'], color='yellow', linestyle='--', alpha=0.3)

        ax9.set_title('Inducement Analysis - Institutional Order Flow', fontsize=16)
        ax9.set_ylabel('Price', fontsize=12)
        ax9.legend(loc='upper left')
        ax9.grid(True, alpha=0.3)

        # Format x-axis dates
        for ax in [ax1, ax2, ax3, ax4, ax6, ax7, ax8, ax9]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.suptitle('XAUUSD Microstructure & Advanced Strategy Analysis Dashboard', fontsize=24, y=1.02)
        plt.savefig('microstructure_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("XAUUSD MICROSTRUCTURE & ADVANCED STRATEGY ANALYSIS REPORT")
        print("="*80)

        print(f"\nAnalysis Period: {self.tick_df['datetime'].min()} to {self.tick_df['datetime'].max()}")
        print(f"Total Ticks Analyzed: {len(self.tick_df):,}")
        print(f"Price Range: {self.tick_df['mid_price'].min():.2f} - {self.tick_df['mid_price'].max():.2f}")

        overall_trend = "BULLISH" if self.tick_df['mid_price'].iloc[-1] > self.tick_df['mid_price'].iloc[0] else "BEARISH"
        price_change = self.tick_df['mid_price'].iloc[-1] - self.tick_df['mid_price'].iloc[0]
        print(f"Overall Trend: {overall_trend} ({price_change:+.2f} points)")

        print("\n" + "-"*50)
        print("MICROSTRUCTURE ANALYSIS")
        print("-"*50)

        print(f"Average Spread: {self.tick_df['spread_pips'].mean():.2f} pips")
        print(f"Maximum Spread: {self.tick_df['spread_pips'].max():.2f} pips")
        print(f"Spread Volatility: {self.tick_df['spread_pips'].std():.2f} pips")
        print(f"Average Tick Volatility: {self.tick_df['volatility'].mean():.4f}")

        print("\n" + "-"*50)
        print("MANIPULATION DETECTION")
        print("-"*50)

        print(f"Spread Spikes: {self.analysis_results['spread_spikes']} detected")
        print(f"Price Reversals: {self.analysis_results['price_reversals']} detected")
        print(f"Stop Hunts: {self.analysis_results['stop_hunts']} detected")
        print(f"Liquidity Sweeps: {self.analysis_results['liquidity_sweeps']} detected")

        manipulation_score = (self.analysis_results['spread_spikes'] + 
                             self.analysis_results['stop_hunts'] + 
                             self.analysis_results['liquidity_sweeps']) / len(self.tick_df) * 100

        print(f"\nManipulation Activity Score: {manipulation_score:.2f}%")

        if manipulation_score > 5:
            print("🔴 HIGH MANIPULATION ACTIVITY - Institutional traders very active")
        elif manipulation_score > 2:
            print("🟡 MODERATE MANIPULATION ACTIVITY - Some institutional involvement")
        else:
            print("🟢 LOW MANIPULATION ACTIVITY - Relatively clean price action")

        print("\n" + "-"*50)
        print("WYCKOFF ANALYSIS")
        print("-"*50)

        for phase, count in self.analysis_results['wyckoff_phases'].items():
            percentage = (count / len(self.tick_df)) * 100
            print(f"{phase}: {count:,} ticks ({percentage:.1f}%)")

        # Determine dominant phase
        dominant_phase = max(self.analysis_results['wyckoff_phases'], 
                           key=self.analysis_results['wyckoff_phases'].get)
        print(f"\nDominant Phase: {dominant_phase}")

        print("\n" + "-"*50)
        print("SMART MONEY CONCEPTS (SMC)")
        print("-"*50)

        print(f"Bullish Fair Value Gaps: {self.analysis_results['bullish_fvgs']}")
        print(f"Bearish Fair Value Gaps: {self.analysis_results['bearish_fvgs']}")

        fvg_bias = "BULLISH" if self.analysis_results['bullish_fvgs'] > self.analysis_results['bearish_fvgs'] else "BEARISH"
        print(f"SMC Bias: {fvg_bias}")

        print("\n" + "-"*50)
        print("INDUCEMENT ANALYSIS")
        print("-"*50)

        print(f"High Inducements (Trapping Longs): {self.analysis_results['high_inducements']}")
        print(f"Low Inducements (Trapping Shorts): {self.analysis_results['low_inducements']}")

        total_inducements = self.analysis_results['high_inducements'] + self.analysis_results['low_inducements']
        inducement_rate = (total_inducements / len(self.tick_df)) * 100
        print(f"Inducement Rate: {inducement_rate:.2f}%")

        print("\n" + "-"*50)
        print("EXPERT COMMENTARY")
        print("-"*50)

        commentary = []

        if manipulation_score > 5:
            commentary.append("🚨 HEAVY institutional manipulation detected. Price action is being heavily engineered.")

        if self.analysis_results['stop_hunts'] > len(self.tick_df) * 0.1:
            commentary.append("🎯 Frequent stop hunting indicates aggressive institutional order flow.")

        if self.analysis_results['liquidity_sweeps'] > 100:
            commentary.append("💧 Multiple liquidity sweeps show smart money collecting orders at key levels.")

        if dominant_phase == "Accumulation":
            commentary.append("📈 Accumulation phase suggests preparation for potential markup.")
        elif dominant_phase == "Distribution":
            commentary.append("📉 Distribution phase indicates potential bearish reversal ahead.")

        if fvg_bias == "BULLISH" and overall_trend == "BEARISH":
            commentary.append("⚔️ SMC shows bullish bias while trend is bearish - potential reversal setup.")

        if inducement_rate > 5:
            commentary.append("🪤 High inducement activity - many retail traders being trapped.")

        for comment in commentary:
            print(comment)

        print("\n" + "-"*50)
        print("TRADING STRATEGY RECOMMENDATIONS")
        print("-"*50)

        recommendations = []

        if fvg_bias == "BULLISH":
            recommendations.append("✅ Focus on LONG setups at Fair Value Gap levels")
        else:
            recommendations.append("✅ Focus on SHORT setups at Fair Value Gap levels")

        if dominant_phase == "Accumulation":
            recommendations.append("✅ Prepare for potential MARKUP phase - look for breakout setups")
        elif dominant_phase == "Distribution":
            recommendations.append("✅ Be cautious of reversals - consider SHORT bias")

        if manipulation_score > 5:
            recommendations.append("⚠️ Use wider stops due to high manipulation activity")
            recommendations.append("⚠️ Consider smaller position sizes in this volatile environment")

        if self.analysis_results['high_inducements'] > self.analysis_results['low_inducements']:
            recommendations.append("📍 Watch for shorts around recent highs (inducement levels)")
        else:
            recommendations.append("📍 Watch for longs around recent lows (inducement levels)")

        recommendations.append("🔄 Monitor price reaction at identified Fair Value Gap levels")
        recommendations.append("🎯 Use liquidity sweep levels as potential reversal zones")

        for rec in recommendations:
            print(rec)

        print("\n" + "-"*50)
        print("V5, V10, V12 STRATEGY INSIGHTS")
        print("-"*50)

        print("V5 Strategy (Basic SMC):")
        print(f"  - {self.analysis_results['bullish_fvgs'] + self.analysis_results['bearish_fvgs']} FVG opportunities identified")
        print(f"  - Bias: {fvg_bias}")

        print("\nV10 Strategy (Advanced Wyckoff):")
        print(f"  - Current market phase: {dominant_phase}")
        print(f"  - Phase distribution suggests {'trending' if max(self.analysis_results['wyckoff_phases'].values()) > len(self.tick_df) * 0.4 else 'ranging'} market")

        print("\nV12 Strategy (Full ZANFLOW):")
        print(f"  - Manipulation score: {manipulation_score:.2f}%")
        print(f"  - Inducement activity: {inducement_rate:.2f}%")
        print(f"  - Market complexity: {'HIGH' if manipulation_score > 5 else 'MODERATE' if manipulation_score > 2 else 'LOW'}")

        print("\n" + "="*80)
        print("END OF ANALYSIS REPORT")
        print("="*80)

    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        if not self.load_data():
            return False

        self.detect_manipulation_patterns()
        self.wyckoff_analysis()
        self.smc_analysis()
        self.inducement_analysis()
        self.create_dashboard()
        self.generate_report()

        return True

# Main execution
if __name__ == "__main__":
    print("ZANFLOW v12 Microstructure Analysis System")
    print("==========================================")

    # Initialize analyzer
    analyzer = MicrostructureAnalyzer('XAUUSD_TICK.csv')

    # Run full analysis
    success = analyzer.run_full_analysis()

    if success:
        print("\n✅ Analysis completed successfully!")
        print("📊 Dashboard saved as 'microstructure_analysis_dashboard.png'")
        print("📋 Full report displayed above")
    else:
        print("\n❌ Analysis failed. Please check your data file.")
