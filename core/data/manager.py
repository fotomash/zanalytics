"""DataManager: Unified data access layer for ZAnalytics
Provides standardized access to all data sources through the data manifest.
"""
import os
import yaml
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import boto3
from urllib.parse import urlparse


class DataManager:
    """Centralized data management system for ZAnalytics."""

    def __init__(self, manifest_path: str = "data_manifest.yml", use_cloud: bool = False):
        """
        Initialize DataManager with manifest configuration.

        Args:
            manifest_path: Path to data_manifest.yml
            use_cloud: Whether to use cloud storage (S3) instead of local
        """
        self.logger = logging.getLogger(__name__)
        self.manifest_path = manifest_path
        self.use_cloud = use_cloud
        self.manifest = self._load_manifest()
        self.base_path = self._get_base_path()
        self._s3_client = None

    def _load_manifest(self) -> Dict[str, Any]:
        """Load and validate the data manifest."""
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            self.logger.info(f"Loaded data manifest from {self.manifest_path}")
            return manifest
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {e}")
            raise

    def _get_base_path(self) -> str:
        """Get the appropriate base path based on environment."""
        if self.use_cloud:
            return self.manifest.get('cloud_base_path', 's3://zanalytics-data')
        return self.manifest.get('base_path', '/var/data/zanalytics')

    @property
    def s3_client(self):
        """Lazy-load S3 client when needed."""
        if self._s3_client is None and self.use_cloud:
            self._s3_client = boto3.client('s3')
        return self._s3_client

    def resolve_path(self, data_type: str, **kwargs) -> str:
        """
        Resolve the full path for a data file based on manifest configuration.

        Args:
            data_type: Type of data (tick_data, ohlc_data, indicator_data)
            **kwargs: Parameters like symbol, date, timeframe, format, indicator

        Returns:
            Full path to the data file
        """
        if data_type not in self.manifest['data_sources']:
            raise ValueError(f"Unknown data type: {data_type}")

        config = self.manifest['data_sources'][data_type]

        # Get path template and file pattern
        path_template = config.get('path_template', '')
        file_pattern = config.get('file_pattern', '')

        # Fill in the templates
        path = path_template.format(**kwargs)
        filename = file_pattern.format(**kwargs)

        # Combine with base path
        full_path = os.path.join(self.base_path, path, filename)

        return full_path

    def get_available_symbols(self, data_type: str) -> List[str]:
        """Get list of available symbols for a data type."""
        if data_type not in self.manifest['data_sources']:
            return []
        return self.manifest['data_sources'][data_type].get('symbols', [])

    def get_available_timeframes(self, data_type: str = 'ohlc_data') -> List[str]:
        """Get list of available timeframes."""
        if data_type not in self.manifest['data_sources']:
            return []
        return self.manifest['data_sources'][data_type].get('timeframes', [])

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names based on schema mapping.

        Args:
            df: DataFrame with potentially non-standard column names

        Returns:
            DataFrame with standardized column names
        """
        column_aliases = self.manifest['schema_mapping']['column_aliases']

        # Create reverse mapping for current columns
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in column_aliases:
                rename_map[col] = column_aliases[col_lower]

        if rename_map:
            df = df.rename(columns=rename_map)
            self.logger.debug(f"Renamed columns: {rename_map}")

        return df

    def validate_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Validate data against quality checks defined in manifest.

        Args:
            df: DataFrame to validate
            data_type: Type of data for validation rules

        Returns:
            Validated DataFrame (may have rows removed)
        """
        if data_type not in self.manifest.get('quality_checks', {}):
            return df

        checks = self.manifest['quality_checks'][data_type]
        original_len = len(df)

        if data_type == 'tick_data' and 'max_spread_pips' in checks:
            if 'spread' in df.columns:
                max_spread = checks['max_spread_pips']
                df = df[df['spread'] <= max_spread]

        if data_type == 'ohlc_data' and checks.get('validate_ohlc_relationship', False):
            # Ensure high >= low and high >= open/close, low <= open/close
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                df = df[(df['high'] >= df['low']) &
                       (df['high'] >= df['open']) &
                       (df['high'] >= df['close']) &
                       (df['low'] <= df['open']) &
                       (df['low'] <= df['close'])]

        removed = original_len - len(df)
        if removed > 0:
            self.logger.warning(f"Removed {removed} invalid rows during validation")

        return df

    def get_data(self,
                 data_type: str,
                 symbol: str,
                 start_date: Optional[Union[str, datetime]] = None,
                 end_date: Optional[Union[str, datetime]] = None,
                 timeframe: Optional[str] = None,
                 format: str = 'parquet',
                 indicator: Optional[str] = None,
                 validate: bool = True) -> pd.DataFrame:
        """
        Get data from the unified data catalog.

        Args:
            data_type: Type of data (tick_data, ohlc_data, indicator_data)
            symbol: Trading symbol (e.g., 'XAUUSD')
            start_date: Start date for data range
            end_date: End date for data range
            timeframe: Timeframe for OHLC data (e.g., '1h')
            format: File format (parquet or csv)
            indicator: Indicator name for indicator_data
            validate: Whether to validate data quality

        Returns:
            DataFrame with requested data
        """
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # If no dates specified, default to last 30 days
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        all_data = []
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')

            # Build kwargs for path resolution
            path_kwargs = {
                'symbol': symbol,
                'date': date_str,
                'format': format
            }

            if timeframe:
                path_kwargs['timeframe'] = timeframe
            if indicator:
                path_kwargs['indicator'] = indicator

            try:
                file_path = self.resolve_path(data_type, **path_kwargs)

                # Read data based on location and format
                if self.use_cloud:
                    df = self._read_from_s3(file_path, format)
                else:
                    df = self._read_from_local(file_path, format)

                if df is not None and not df.empty:
                    # Standardize columns
                    df = self.standardize_columns(df)

                    # Filter by date range if timestamp column exists
                    if 'timestamp' in df.columns:
                        df = df[(df['timestamp'] >= start_date) &
                               (df['timestamp'] <= end_date)]

                    all_data.append(df)

            except Exception as e:
                self.logger.debug(f"No data for {current_date}: {e}")

            current_date += timedelta(days=1)

        if not all_data:
            self.logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Combine all data
        result = pd.concat(all_data, ignore_index=True)

        # Remove duplicates if any
        if 'timestamp' in result.columns:
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        # Validate data if requested
        if validate:
            result = self.validate_data(result, data_type)

        return result

    def _read_from_local(self, file_path: str, format: str) -> Optional[pd.DataFrame]:
        """Read data from local filesystem."""
        if not os.path.exists(file_path):
            return None

        try:
            if format == 'parquet':
                return pd.read_parquet(file_path)
            elif format == 'csv':
                return pd.read_csv(file_path, parse_dates=['timestamp'])
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None

    def _read_from_s3(self, file_path: str, format: str) -> Optional[pd.DataFrame]:
        """Read data from S3."""
        try:
            # Parse S3 path
            parsed = urlparse(file_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')

            # Check if object exists
            try:
                self.s3_client.head_object(Bucket=bucket, Key=key)
            except Exception:
                return None

            # Read based on format
            if format == 'parquet':
                return pd.read_parquet(f"s3://{bucket}/{key}")
            elif format == 'csv':
                return pd.read_csv(f"s3://{bucket}/{key}", parse_dates=['timestamp'])
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            self.logger.error(f"Error reading from S3 {file_path}: {e}")
            return None

    def save_data(self,
                  df: pd.DataFrame,
                  data_type: str,
                  symbol: str,
                  date: Optional[Union[str, datetime]] = None,
                  timeframe: Optional[str] = None,
                  format: str = 'parquet',
                  indicator: Optional[str] = None) -> str:
        """
        Save data to the unified data catalog.

        Args:
            df: DataFrame to save
            data_type: Type of data
            symbol: Trading symbol
            date: Date for the data (defaults to today)
            timeframe: Timeframe for OHLC data
            format: File format
            indicator: Indicator name for indicator_data

        Returns:
            Path where data was saved
        """
        # Default to today if no date specified
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = pd.to_datetime(date)

        date_str = date.strftime('%Y%m%d')

        # Build kwargs for path resolution
        path_kwargs = {
            'symbol': symbol,
            'date': date_str,
            'format': format
        }

        if timeframe:
            path_kwargs['timeframe'] = timeframe
        if indicator:
            path_kwargs['indicator'] = indicator

        file_path = self.resolve_path(data_type, **path_kwargs)

        # Ensure directory exists for local storage
        if not self.use_cloud:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save based on format
        try:
            if format == 'parquet':
                df.to_parquet(file_path, index=False)
            elif format == 'csv':
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Saved data to {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {e}")
            raise

    @lru_cache(maxsize=128)
    def get_schema(self, data_type: str, indicator: Optional[str] = None) -> Dict[str, str]:
        """Get the expected schema for a data type."""
        if data_type not in self.manifest['data_sources']:
            return {}

        config = self.manifest['data_sources'][data_type]

        if data_type == 'indicator_data' and indicator:
            return config.get('schema_mapping', {}).get(indicator, {})
        else:
            return config.get('schema', {})

    def list_available_data(self, data_type: str, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available data files for a given data type and optional symbol.

        Returns:
            List of dictionaries with file information
        """
        available_files = []

        if data_type not in self.manifest['data_sources']:
            return available_files

        config = self.manifest['data_sources'][data_type]
        symbols = [symbol] if symbol else config.get('symbols', [])

        for sym in symbols:
            if data_type == 'ohlc_data':
                for tf in config.get('timeframes', []):
                    # Check for existing files
                    # Implementation depends on storage backend
                    pass
            elif data_type == 'tick_data':
                # Check for tick data files
                pass

        return available_files


# Singleton instance for easy access
_data_manager_instance = None

def get_data_manager(manifest_path: str = "data_manifest.yml", use_cloud: bool = False) -> DataManager:
    """Get or create the global DataManager instance."""
    global _data_manager_instance
    if _data_manager_instance is None:
        _data_manager_instance = DataManager(manifest_path, use_cloud)
    return _data_manager_instance
