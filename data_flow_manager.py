# data_flow_manager.py - v2 (with Startup Scan)
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
from threading import Lock
import queue
import time
import os

@dataclass
class DataFlowEvent:
    """Represents a data flow event"""
    event_type: str  # 'new_data', 'analysis_complete', 'tick_update', etc.
    source: str      # 'csv', 'json', 'tick', 'api'
    symbol: str
    timeframe: str
    timestamp: datetime
    data: Dict[Any, Any]
    file_path: Optional[str] = None

class DataFlowHandler(FileSystemEventHandler):
    """Monitors file system for new data and performs initial scan."""
    
    def __init__(self, callback: Callable[[DataFlowEvent], None]):
        self.callback = callback
        self.processed_files = set()
        self.lock = Lock()
        
    def on_created(self, event):
        if not event.is_directory:
            self._process_file(event.src_path, 'file_created')
    
    def on_modified(self, event):
        if not event.is_directory:
            # Optional: handle modifications if needed, for now we focus on creation
            # self._process_file(event.src_path, 'file_modified')
            pass
    
    def _process_file(self, file_path: str, event_type: str):
        """Deduplicates and processes a file path."""
        file_path_str = str(file_path)
        with self.lock:
            if file_path_str in self.processed_files:
                return
            # Add to processed set immediately to prevent race conditions
            self.processed_files.add(file_path_str)
        
        try:
            path = Path(file_path_str)
            
            # Detect file type and extract metadata
            if path.suffix.lower() == '.csv':
                self._handle_csv_file(path, event_type)
            elif path.suffix.lower() == '.json':
                self._handle_json_file(path, event_type)
                
        except Exception as e:
            logging.error(f"Error processing file {file_path_str}: {e}")
    
    def _handle_csv_file(self, path: Path, event_type: str):
        """Handle CSV data files"""
        try:
            # Extract symbol and timeframe from filename
            filename = path.stem
            symbol = self._extract_symbol(filename)
            timeframe = self._extract_timeframe(filename)
            
            # Quick peek at data structure
            df = pd.read_csv(path, nrows=5)
            data_type = 'tick' if 'bid' in df.columns else 'ohlc'
            
            flow_event = DataFlowEvent(
                event_type='new_csv_data',
                source='csv',
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                data={
                    'file_path': str(path),
                    'data_type': data_type,
                    'columns': list(df.columns),
                    'row_count': len(pd.read_csv(path))
                },
                file_path=str(path)
            )
            
            self.callback(flow_event)
            
        except Exception as e:
            logging.error(f"Error handling CSV file {path}: {e}")
    
    def _handle_json_file(self, path: Path, event_type: str):
        """Handle JSON analysis files"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Extract metadata from JSON structure
            symbol = data.get('metadata', {}).get('symbol', 'UNKNOWN')
            timeframes = list(data.get('analysis', {}).keys()) if 'analysis' in data else []
            
            flow_event = DataFlowEvent(
                event_type='new_analysis_data',
                source='json',
                symbol=symbol,
                timeframe=','.join(timeframes),
                timestamp=datetime.now(),
                data={
                    'file_path': str(path),
                    'analysis_keys': list(data.keys()),
                    'timeframes': timeframes,
                    'indicators': self._extract_indicators(data)
                },
                file_path=str(path)
            )
            
            self.callback(flow_event)
            
        except Exception as e:
            logging.error(f"Error handling JSON file {path}: {e}")
    
    def _extract_symbol(self, filename: str) -> str:
        """Extract trading symbol from filename"""
        parts = filename.upper().split('_')
        for part in parts:
            if any(pair in part for pair in ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'XAU', 'XAG']):
                return part
        return parts[0] if parts else 'UNKNOWN'
    
    def _extract_timeframe(self, filename: str) -> str:
        """Extract timeframe from filename"""
        filename_upper = filename.upper()
        if 'TICK' in filename_upper: return 'TICK'
        timeframe_map = {'M1': 'M1', 'M5': 'M5', 'M15': 'M15', 'H1': 'H1', 'H4': 'H4', 'D1': 'D1'}
        for key, value in timeframe_map.items():
            if key in filename_upper:
                return value
        return 'UNKNOWN'
    
    def _extract_indicators(self, data: Dict) -> List[str]:
        """Extract available indicators from analysis data"""
        indicators = []
        if 'analysis' in data:
            for tf_data in data['analysis'].values():
                if isinstance(tf_data, dict) and 'indicators' in tf_data:
                    indicators.extend(tf_data['indicators'].keys())
        return list(set(indicators))

class DataFlowManager:
    """Main data flow awareness manager"""
    
    def __init__(self, watch_directories: List[str], zanalytics_callback: Optional[Callable] = None):
        self.watch_directories = watch_directories
        self.zanalytics_callback = zanalytics_callback
        self.observers = []
        self.event_queue = queue.Queue()
        self.is_running = False
        self.data_cache = {}
        self.agent_subscriptions = {}
        self.handler = DataFlowHandler(self._on_data_event) # --- MODIFIED ---
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    # --- NEW METHOD ---
    def _scan_existing_files(self, directory: str):
        """Scans for files that already exist on startup."""
        self.logger.info(f"Performing initial scan of directory: {directory}...")
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                self.handler._process_file(file_path, 'initial_scan')
        self.logger.info(f"Initial scan of {directory} complete.")

    def start_monitoring(self):
        """Start monitoring directories for data flow"""
        self.is_running = True
        
        for directory in self.watch_directories:
            path_obj = Path(directory)
            if path_obj.exists() and path_obj.is_dir():
                # --- NEW ---: Perform the initial scan first
                self._scan_existing_files(directory)
                
                # Now, schedule the observer to watch for new files
                observer = Observer()
                observer.schedule(self.handler, directory, recursive=True)
                observer.start()
                self.observers.append(observer)
                self.logger.info(f"Live monitoring started for: {directory}")
            else:
                self.logger.warning(f"Directory not found, cannot monitor: {directory}")
        
        asyncio.create_task(self._process_events())
        self.logger.info("DataFlowManager started successfully")
    
    def stop_monitoring(self):
        """Stop all monitors"""
        self.is_running = False
        for observer in self.observers:
            observer.stop()
            observer.join()
        self.logger.info("DataFlowManager stopped")
    
    def _on_data_event(self, event: DataFlowEvent):
        """Handle incoming data events"""
        self.event_queue.put(event)
        self.logger.info(f"New data event queued: {event.symbol} {event.timeframe} - {event.event_type}")
    
    async def _process_events(self):
        """Process data events asynchronously"""
        while self.is_running:
            try:
                if not self.event_queue.empty():
                    event = self.event_queue.get_nowait()
                    await self._handle_event(event)
                await asyncio.sleep(0.1)
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing event queue: {e}")
    
    async def _handle_event(self, event: DataFlowEvent):
        """Handle individual data events"""
        cache_key = f"{event.symbol}_{event.timeframe}"
        self.data_cache[cache_key] = {
            'last_update': event.timestamp,
            'data': event.data,
            'source': event.source
        }
        
        if cache_key in self.agent_subscriptions:
            for agent_callback in self.agent_subscriptions[cache_key]:
                try:
                    await agent_callback(event)
                except Exception as e:
                    self.logger.error(f"Error notifying agent: {e}")
        
        if self.zanalytics_callback:
            try:
                await self.zanalytics_callback(event)
            except Exception as e:
                self.logger.error(f"Error notifying ZANALYTICS: {e}")
    
    def subscribe_agent(self, symbol: str, timeframe: str, callback: Callable):
        """Subscribe an agent to data updates"""
        cache_key = f"{symbol}_{timeframe}"
        if cache_key not in self.agent_subscriptions:
            self.agent_subscriptions[cache_key] = []
        self.agent_subscriptions[cache_key].append(callback)
        self.logger.info(f"Agent subscribed to {cache_key}")
    
    def get_latest_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get latest data for symbol/timeframe"""
        cache_key = f"{symbol}_{timeframe}"
        return self.data_cache.get(cache_key)
    
    def get_data_status(self) -> Dict:
        """Get overall data flow status"""
        return {
            'active_streams': len(self.data_cache),
            'monitored_directories': len(self.watch_directories),
            'last_events': {k: v['last_update'].isoformat() for k, v in self.data_cache.items()},
            'is_running': self.is_running
        }