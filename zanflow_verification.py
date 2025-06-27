#!/usr/bin/env python3
"""
ZANFLOW Data Processing Verification Script
Ensures all components are correctly processing data
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import traceback
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

class DataProcessingVerifier:
    """Verify data processing pipeline integrity"""

    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.verification_results = {}
        self.errors = []
        self.warnings = []

    def run_full_verification(self):
        """Run complete verification suite"""
        print("🔍 ZANFLOW Data Processing Verification Starting...")
        print("=" * 60)

        # 1. Check data directory structure
        self.verify_directory_structure()

        # 2. Verify data files
        self.verify_data_files()

        # 3. Test data loading
        self.test_data_loading()

        # 4. Verify indicator calculations
        self.verify_indicators()

        # 5. Test SMC analysis
        self.test_smc_analysis()

        # 6. Test Wyckoff analysis
        self.test_wyckoff_analysis()

        # 7. Verify microstructure analysis
        self.verify_microstructure()

        # 8. Generate verification report
        self.generate_report()

    def verify_directory_structure(self):
        """Verify expected directory structure"""
        print("\n📁 Verifying Directory Structure...")

        if not self.data_dir.exists():
            self.errors.append(f"Data directory {self.data_dir} does not exist!")
            return

        # Check for expected subdirectories
        expected_patterns = ['*_ticks', '*_tick', '*TICK*']
        found_dirs = []

        for pattern in expected_patterns:
            found_dirs.extend(list(self.data_dir.glob(pattern)))

        if found_dirs:
            print(f"✅ Found {len(found_dirs)} tick data directories")
            for d in found_dirs:
                print(f"   - {d.name}")
        else:
            self.warnings.append("No tick data directories found")

        # Check for CSV files
        csv_files = list(self.data_dir.rglob("*.csv"))
        json_files = list(self.data_dir.rglob("*.json"))

        print(f"\n📊 Found {len(csv_files)} CSV files and {len(json_files)} JSON files")

        self.verification_results['directory_structure'] = {
            'data_dir_exists': self.data_dir.exists(),
            'tick_dirs_found': len(found_dirs),
            'csv_files_found': len(csv_files),
            'json_files_found': len(json_files)
        }

    def verify_data_files(self):
        """Verify data file integrity"""
        print("\n📋 Verifying Data Files...")

        csv_files = list(self.data_dir.rglob("*.csv"))
        verified_files = []

        for csv_file in csv_files[:5]:  # Check first 5 files
            try:
                # Try to read the file
                df = pd.read_csv(csv_file, nrows=10)

                # Check required columns
                required_cols = {
                    'tick': ['timestamp', 'bid', 'ask'],
                    'bar': ['timestamp', 'open', 'high', 'low', 'close']
                }

                file_type = 'tick' if 'tick' in csv_file.name.lower() else 'bar'

                if file_type == 'tick':
                    has_required = all(col in df.columns for col in required_cols['tick'])
                else:
                    has_required = all(col in df.columns for col in required_cols['bar'])

                verified_files.append({
                    'file': csv_file.name,
                    'type': file_type,
                    'columns': list(df.columns),
                    'rows': len(df),
                    'has_required_columns': has_required
                })

                if has_required:
                    print(f"✅ {csv_file.name}: Valid {file_type} data")
                else:
                    print(f"⚠️  {csv_file.name}: Missing required columns")
                    self.warnings.append(f"{csv_file.name} missing required columns")

            except Exception as e:
                self.errors.append(f"Error reading {csv_file.name}: {str(e)}")
                print(f"❌ {csv_file.name}: Error - {str(e)}")

        self.verification_results['file_verification'] = verified_files

    def test_data_loading(self):
        """Test data loading functionality"""
        print("\n🔄 Testing Data Loading...")

        # Test loading different file types
        test_results = []

        # Test CSV loading
        csv_files = list(self.data_dir.rglob("*.csv"))[:3]
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Check timestamp parsing
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Add calculated columns
                if 'bid' in df.columns and 'ask' in df.columns:
                    df['mid'] = (df['bid'] + df['ask']) / 2
                    df['spread'] = df['ask'] - df['bid']

                test_results.append({
                    'file': csv_file.name,
                    'loaded': True,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'has_timestamps': 'timestamp' in df.columns,
                    'memory_usage': df.memory_usage().sum() / 1024**2  # MB
                })

                print(f"✅ Loaded {csv_file.name}: {len(df)} rows")

            except Exception as e:
                test_results.append({
                    'file': csv_file.name,
                    'loaded': False,
                    'error': str(e)
                })
                print(f"❌ Failed to load {csv_file.name}: {str(e)}")

        self.verification_results['data_loading'] = test_results

    def verify_indicators(self):
        """Verify indicator calculations"""
        print("\n📈 Verifying Indicator Calculations...")

        # Load a sample file for testing
        csv_files = list(self.data_dir.rglob("*bars*.csv"))
        if not csv_files:
            csv_files = list(self.data_dir.rglob("*.csv"))

        if not csv_files:
            self.warnings.append("No files found for indicator testing")
            return

        test_file = csv_files[0]

        try:
            df = pd.read_csv(test_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # Test basic indicators
            indicators_tested = []

            # Moving averages
            if 'close' in df.columns:
                for period in [8, 21, 55]:
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                    df[f'sma_{period}'] = df['close'].rolling(period).mean()
                    indicators_tested.extend([f'ema_{period}', f'sma_{period}'])

                # RSI
                df['rsi_14'] = self.calculate_rsi(df['close'], 14)
                indicators_tested.append('rsi_14')

                # ATR
                if all(col in df.columns for col in ['high', 'low', 'close']):
                    df['atr_14'] = self.calculate_atr(df, 14)
                    indicators_tested.append('atr_14')

                # Bollinger Bands
                df['bb_middle'] = df['close'].rolling(20).mean()
                df['bb_std'] = df['close'].rolling(20).std()
                df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
                df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
                indicators_tested.extend(['bb_upper', 'bb_middle', 'bb_lower'])

            print(f"✅ Successfully calculated {len(indicators_tested)} indicators")

            # Verify indicator values
            nan_counts = {}
            for ind in indicators_tested:
                if ind in df.columns:
                    nan_count = df[ind].isna().sum()
                    nan_counts[ind] = nan_count

            self.verification_results['indicators'] = {
                'file_tested': test_file.name,
                'indicators_calculated': indicators_tested,
                'nan_counts': nan_counts,
                'total_rows': len(df)
            }

        except Exception as e:
            self.errors.append(f"Error testing indicators: {str(e)}")
            print(f"❌ Indicator calculation failed: {str(e)}")

    def test_smc_analysis(self):
        """Test SMC analysis functionality"""
        print("\n🎯 Testing SMC Analysis...")

        # Find processed SMC files
        smc_files = list(self.data_dir.rglob("*bars*processed.csv"))

        if not smc_files:
            self.warnings.append("No SMC processed files found")
            print("⚠️  No SMC processed files found")
            return

        test_file = smc_files[0]

        try:
            df = pd.read_csv(test_file)

            # Check for SMC columns
            smc_columns = [
                'bullish_fvg', 'bearish_fvg',
                'bullish_order_block', 'bearish_order_block',
                'structure_break', 'liquidity_high', 'liquidity_low'
            ]

            found_smc_cols = [col for col in smc_columns if col in df.columns]

            if found_smc_cols:
                print(f"✅ Found {len(found_smc_cols)} SMC indicators")

                # Count occurrences
                smc_counts = {}
                for col in found_smc_cols:
                    if df[col].dtype == bool:
                        smc_counts[col] = df[col].sum()
                    else:
                        smc_counts[col] = (df[col] != 0).sum()

                for col, count in smc_counts.items():
                    print(f"   - {col}: {count} occurrences")

                self.verification_results['smc_analysis'] = {
                    'file_tested': test_file.name,
                    'smc_columns_found': found_smc_cols,
                    'smc_counts': smc_counts
                }
            else:
                self.warnings.append(f"No SMC columns found in {test_file.name}")
                print(f"⚠️  No SMC columns found in {test_file.name}")

        except Exception as e:
            self.errors.append(f"Error testing SMC analysis: {str(e)}")
            print(f"❌ SMC analysis test failed: {str(e)}")

    def test_wyckoff_analysis(self):
        """Test Wyckoff analysis functionality"""
        print("\n📊 Testing Wyckoff Analysis...")

        # Check for Wyckoff JSON file
        wyckoff_json = self.data_dir / "wyckoff_smc_chart.json"

        if wyckoff_json.exists():
            try:
                with open(wyckoff_json, 'r') as f:
                    wyckoff_data = json.load(f)

                print(f"✅ Loaded Wyckoff configuration:")
                print(f"   - Pair: {wyckoff_data.get('pair')}")
                print(f"   - Schema: {wyckoff_data.get('schema')}")
                print(f"   - Entry: {wyckoff_data.get('entry', {}).get('price')}")

                self.verification_results['wyckoff_config'] = wyckoff_data

            except Exception as e:
                self.errors.append(f"Error loading Wyckoff config: {str(e)}")

        # Test Wyckoff phase detection
        csv_files = list(self.data_dir.rglob("*.csv"))[:1]
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0])

                # Check for Wyckoff columns
                wyckoff_cols = ['wyckoff_phase', 'wyckoff_event', 'accumulation', 'distribution']
                found_wyckoff = [col for col in wyckoff_cols if col in df.columns]

                if found_wyckoff:
                    print(f"✅ Found Wyckoff indicators: {found_wyckoff}")
                else:
                    print("⚠️  No Wyckoff indicators found in data")

            except Exception as e:
                print(f"❌ Wyckoff test failed: {str(e)}")

    def verify_microstructure(self):
        """Verify microstructure analysis"""
        print("\n🔬 Verifying Microstructure Analysis...")

        # Find tick data files
        tick_files = list(self.data_dir.rglob("*tick*.csv"))

        if not tick_files:
            self.warnings.append("No tick data files found")
            print("⚠️  No tick data files found")
            return

        test_file = tick_files[0]

        try:
            df = pd.read_csv(test_file, nrows=1000)  # Load first 1000 ticks

            # Check for microstructure columns
            micro_cols = ['bid', 'ask', 'spread', 'mid', 'volume']
            found_cols = [col for col in micro_cols if col in df.columns]

            if 'bid' in df.columns and 'ask' in df.columns:
                # Calculate microstructure metrics
                df['spread'] = df['ask'] - df['bid']
                df['mid'] = (df['bid'] + df['ask']) / 2

                metrics = {
                    'avg_spread': df['spread'].mean(),
                    'max_spread': df['spread'].max(),
                    'min_spread': df['spread'].min(),
                    'spread_volatility': df['spread'].std()
                }

                print(f"✅ Microstructure metrics calculated:")
                for metric, value in metrics.items():
                    print(f"   - {metric}: {value:.6f}")

                self.verification_results['microstructure'] = {
                    'file_tested': test_file.name,
                    'columns_found': found_cols,
                    'metrics': metrics
                }

            else:
                self.warnings.append(f"Missing bid/ask columns in {test_file.name}")
                print(f"⚠️  Missing bid/ask columns in {test_file.name}")

        except Exception as e:
            self.errors.append(f"Error testing microstructure: {str(e)}")
            print(f"❌ Microstructure test failed: {str(e)}")

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df, period=14):
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    def generate_report(self):
        """Generate comprehensive verification report"""
        print("\n📝 Generating Verification Report...")
        print("=" * 60)

        # Summary
        print("\n🎯 VERIFICATION SUMMARY")
        print(f"✅ Successful checks: {len([k for k, v in self.verification_results.items() if v])}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")

        # Detailed results
        if self.verification_results.get('directory_structure'):
            print("\n📁 Directory Structure:")
            for key, value in self.verification_results['directory_structure'].items():
                print(f"   - {key}: {value}")

        if self.verification_results.get('file_verification'):
            print("\n📋 File Verification:")
            for file_info in self.verification_results['file_verification'][:3]:
                print(f"   - {file_info['file']}: {file_info['type']} data, {file_info['rows']} rows")

        if self.verification_results.get('indicators'):
            print("\n📈 Indicators:")
            ind_info = self.verification_results['indicators']
            print(f"   - Tested on: {ind_info['file_tested']}")
            print(f"   - Indicators calculated: {len(ind_info['indicators_calculated'])}")

        if self.verification_results.get('smc_analysis'):
            print("\n🎯 SMC Analysis:")
            smc_info = self.verification_results['smc_analysis']
            print(f"   - SMC columns found: {len(smc_info['smc_columns_found'])}")

        # Warnings
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings[:5]:
                print(f"   - {warning}")

        # Errors
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors[:5]:
                print(f"   - {error}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")

        if not self.verification_results.get('directory_structure', {}).get('tick_dirs_found'):
            print("   - Create tick data directories with naming pattern: PAIR_ticks/")

        if self.errors:
            print("   - Fix data loading errors before running dashboard")

        if not self.verification_results.get('smc_analysis'):
            print("   - Run SMC analysis scripts to generate processed files")

        print("\n✅ Verification Complete!")

        # Save report to file
        report_path = Path("verification_report.json")
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'results': self.verification_results,
            'warnings': self.warnings,
            'errors': self.errors
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n📄 Full report saved to: {report_path}")

# Create test data generator
def generate_test_data():
    """Generate test data for verification"""
    print("\n🔧 Generating Test Data...")

    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # Create test tick data
    tick_dir = data_dir / "EURUSD_ticks"
    tick_dir.mkdir(exist_ok=True)

    # Generate tick data
    timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=1000, freq='100ms')
    base_price = 1.0850

    tick_data = {
        'timestamp': timestamps,
        'bid': base_price + np.random.randn(1000) * 0.0001,
        'ask': base_price + 0.0002 + np.random.randn(1000) * 0.0001,
        'volume': np.random.randint(1, 100, 1000)
    }

    tick_df = pd.DataFrame(tick_data)
    tick_df['mid'] = (tick_df['bid'] + tick_df['ask']) / 2
    tick_df['spread'] = tick_df['ask'] - tick_df['bid']

    tick_file = tick_dir / "EURUSD_tick_processed.csv"
    tick_df.to_csv(tick_file, index=False)
    print(f"✅ Created test tick data: {tick_file}")

    # Generate bar data with SMC columns
    bar_timestamps = pd.date_range(start='2024-01-01', periods=500, freq='1min')

    bar_data = {
        'timestamp': bar_timestamps,
        'open': base_price + np.random.randn(500) * 0.001,
        'high': base_price + 0.001 + np.abs(np.random.randn(500) * 0.001),
        'low': base_price - 0.001 - np.abs(np.random.randn(500) * 0.001),
        'close': base_price + np.random.randn(500) * 0.001,
        'volume': np.random.randint(100, 1000, 500)
    }

    bar_df = pd.DataFrame(bar_data)

    # Add SMC columns
    bar_df['bullish_fvg'] = np.random.choice([True, False], 500, p=[0.05, 0.95])
    bar_df['bearish_fvg'] = np.random.choice([True, False], 500, p=[0.05, 0.95])
    bar_df['bullish_order_block'] = np.random.choice([True, False], 500, p=[0.03, 0.97])
    bar_df['bearish_order_block'] = np.random.choice([True, False], 500, p=[0.03, 0.97])
    bar_df['structure_break'] = np.random.choice([True, False], 500, p=[0.02, 0.98])

    # Add indicators
    bar_df['ema_8'] = bar_df['close'].ewm(span=8).mean()
    bar_df['ema_21'] = bar_df['close'].ewm(span=21).mean()
    bar_df['rsi_14'] = 50 + np.random.randn(500) * 20  # Simplified RSI

    bar_file = tick_dir / "EURUSD_bars_1min_processed.csv"
    bar_df.to_csv(bar_file, index=False)
    print(f"✅ Created test bar data with SMC: {bar_file}")

    print("✅ Test data generation complete!")

if __name__ == "__main__":
    print("🚀 ZANFLOW Data Processing Verification Tool")
    print("=" * 60)

    # Check if we should generate test data
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-test-data":
        generate_test_data()

    # Run verification
    verifier = DataProcessingVerifier()
    verifier.run_full_verification()
