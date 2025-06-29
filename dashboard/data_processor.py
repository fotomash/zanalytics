# Create the updated data processor utility
#data_processor_update = '''"""
# Data Processor Module
# Handles loading and preprocessing of market data, specifically for tab-separated formats.
# """

import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback

class DataProcessor:
    def __init__(self):
        self.data = None
        self.metadata = {}
    
    def load_data(self, filepath):
        """Load market data, specifically handling tab-separated formats."""
        try:
            # Explicitly read as tab-separated, as seen in XRPUSD_M1_bars.csv
            df = pd.read_csv(filepath, sep='\t', engine='python')

            # Check if parsing failed and resulted in a single column
            if len(df.columns) == 1:
                # This happens when the header is also tab-separated.
                # We need to split the single column into multiple columns.
                df = df.iloc[:, 0].str.split('\t', expand=True)

                # The header is now the first row, so we promote it.
                new_header = df.iloc[0]
                df = df[1:]
                df.columns = new_header
                df.reset_index(drop=True, inplace=True)

            # Standardize column names (make them lowercase and strip whitespace)
            df.columns = [str(col).lower().strip() for col in df.columns]

            # Rename 'timestamp' to 'datetime' for consistency
            if 'timestamp' in df.columns:
                df.rename(columns={'timestamp': 'datetime'}, inplace=True)

            # Ensure required columns exist
            required_cols = ['datetime', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Data file must contain columns: {required_cols}. Found: {df.columns.tolist()}")

            # Handle volume - use 'tickvol' if 'volume' is 0 or not present
            if 'volume' not in df.columns or df['volume'].astype(float).sum() == 0:
                if 'tickvol' in df.columns:
                    print("Info: 'volume' column is empty or missing. Using 'tickvol' as volume source.")
                    df.rename(columns={'tickvol': 'volume'}, inplace=True)
                else:
                    # If no volume data, create a column of ones to avoid errors
                    print("Warning: No volume data found. Using 1 as a placeholder.")
                    df['volume'] = 1

            # Convert data types
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Clean up data
            df.dropna(inplace=True)
            df.sort_index(inplace=True)

            self.metadata = {
                'symbol': os.path.basename(filepath).split('_')[0],
                'start_date': df.index[0],
                'end_date': df.index[-1],
                'total_bars': len(df),
                'timeframe': self._detect_timeframe(df)
            }

            self.data = df
            return df

        except Exception as e:
            tb = traceback.format_exc()
            raise Exception(f"Error loading data from {filepath}: {str(e)}\nTraceback:\n{tb}")

    def load_tick_data(self, filepath):
        """Load and process tick data from a CSV file."""
        try:
            # Load the data, assuming tab separation
            ticks_df = pd.read_csv(filepath, sep='\t', engine='python')

            # As before, handle the case where the header is part of the data
            if len(ticks_df.columns) == 1:
                ticks_df = ticks_df.iloc[:, 0].str.split('\t', expand=True)
                new_header = ticks_df.iloc[0]
                ticks_df = ticks_df[1:]
                ticks_df.columns = new_header
                ticks_df.reset_index(drop=True, inplace=True)

            # Standardize column names
            ticks_df.columns = [str(col).lower().strip() for col in ticks_df.columns]

            # Rename 'timestamp' to 'datetime'
            if 'timestamp' in ticks_df.columns:
                ticks_df.rename(columns={'timestamp': 'datetime'}, inplace=True)

            # Convert data types
            ticks_df['datetime'] = pd.to_datetime(ticks_df['datetime'])
            ticks_df.set_index('datetime', inplace=True)

            numeric_cols = ['bid', 'ask', 'last', 'volume', 'spread']
            for col in numeric_cols:
                if col in ticks_df.columns:
                    ticks_df[col] = pd.to_numeric(ticks_df[col], errors='coerce')

            # Clean up
            ticks_df.dropna(subset=['bid', 'ask'], inplace=True)
            ticks_df.sort_index(inplace=True)

            print(f"Successfully loaded and processed {len(ticks_df)} ticks from {filepath}.")
            return ticks_df

        except Exception as e:
            tb = traceback.format_exc()
            raise Exception(f"Error loading tick data from {filepath}: {str(e)}\nTraceback:\n{tb}")

    def _detect_timeframe(self, df):
        """Detect the timeframe of the data"""
        if len(df) < 2:
            return "Unknown"

        time_diffs = df.index[1:] - df.index[:-1]
        avg_diff = time_diffs.mean()
        minutes = avg_diff.total_seconds() / 60

        timeframe_map = {
            1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1",
            240: "H4", 1440: "D1", 10080: "W1"
        }

        closest_tf = min(timeframe_map.keys(), key=lambda x: abs(x - minutes))
        return timeframe_map[closest_tf]