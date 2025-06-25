# run_system.py - The All-in-One ZANALYTICS Launcher

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
from threading import Lock
import queue

# ==============================================================================
# SECTION 1: CORE LOGIC (Adapted from data_flow_manager.py)
# ==============================================================================

@dataclass
class DataFlowEvent:
    event_type: str
    source: str
    symbol: str
    timeframe: str
    timestamp: Any
    data: Dict[Any, Any]
    file_path: Optional[str] = None

class DataFlowHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[DataFlowEvent], None]):
        self.callback = callback
        self.processed_files = set()
        self.lock = Lock()

    def on_created(self, event):
        if not event.is_directory: self._process_file(event.src_path)
    
    def _process_file(self, file_path: str):
        file_path_str = str(file_path)
        with self.lock:
            if file_path_str in self.processed_files: return
            self.processed_files.add(file_path_str)
        
        try:
            path = Path(file_path_str)
            if path.suffix.lower() == '.csv': self._handle_csv_file(path)
            elif path.suffix.lower() == '.json': self._handle_json_file(path)
        except Exception as e:
            logging.error(f"Error processing file {file_path_str}: {e}")

    def _handle_csv_file(self, path: Path):
        try:
            filename = path.stem
            symbol = self._extract_symbol(filename)
            timeframe = self._extract_timeframe(filename)
            df = pd.read_csv(path, nrows=5)
            data_type = 'tick' if 'bid' in df.columns else 'ohlc'
            
            self.callback(DataFlowEvent(
                event_type='new_csv_data', source='csv', symbol=symbol, timeframe=timeframe,
                timestamp=pd.to_datetime('now', utc=True),
                data={'file_path': str(path), 'data_type': data_type},
                file_path=str(path)
            ))
        except Exception as e: logging.error(f"Error handling CSV {path}: {e}")

    def _handle_json_file(self, path: Path):
        try:
            with open(path, 'r') as f: data = json.load(f)
            symbol = data.get('metadata', {}).get('symbol', 'UNKNOWN')
            timeframes = list(data.get('analysis', {}).keys())
            
            self.callback(DataFlowEvent(
                event_type='new_analysis_data', source='json', symbol=symbol, timeframe=','.join(timeframes),
                timestamp=pd.to_datetime('now', utc=True),
                data={'file_path': str(path), 'timeframes': timeframes},
                file_path=str(path)
            ))
        except Exception as e: logging.error(f"Error handling JSON {path}: {e}")

    def _extract_symbol(self, filename: str) -> str:
        parts = filename.upper().split('_')
        for part in parts:
            if any(p in part for p in ['EUR', 'USD', 'GBP', 'JPY', 'XAU']): return part
        return 'UNKNOWN'

    def _extract_timeframe(self, filename: str) -> str:
        filename_upper = filename.upper()
        if 'TICK' in filename_upper: return 'TICK'
        for tf in ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']:
            if tf in filename_upper: return tf
        return 'UNKNOWN'

class DataFlowManager:
    def __init__(self, watch_directories: List[str], event_callback: Callable):
        self.watch_directories = watch_directories
        self.event_callback = event_callback
        self.observers = []
        self.handler = DataFlowHandler(self.event_callback)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _scan_existing_files(self, directory: str):
        self.logger.info(f"Performing initial scan of directory: {directory}...")
        for root, _, files in os.walk(directory):
            for filename in files:
                self.handler._process_file(os.path.join(root, filename))
        self.logger.info(f"Initial scan of {directory} complete.")

    def start_monitoring(self):
        for directory in self.watch_directories:
            if os.path.exists(directory):
                self._scan_existing_files(directory)
                observer = Observer()
                observer.schedule(self.handler, directory, recursive=True)
                observer.start()
                self.observers.append(observer)
                self.logger.info(f"Live monitoring started for: {directory}")
            else:
                self.logger.warning(f"Directory not found, cannot monitor: {directory}")
    
    def stop_monitoring(self):
        for observer in self.observers:
            observer.stop()
            observer.join()

# ==============================================================================
# SECTION 2: ZANALYTICS BRIDGE & AGENT INITIALIZATION
# ==============================================================================

# --- CORRECTED IMPORTS FOR FLAT STRUCTURE ---
# Python can now find these files because they are in the same folder.
from agent_initializer import initialize_agents
from agent_registry import AgentRegistry

class ZAnalyticsSystem:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._load_config()
        self.agent_registry = AgentRegistry()
        self.event_queue = asyncio.Queue()
        
        self.data_flow_manager = DataFlowManager(
            watch_directories=self.config.get('watch_directories', []),
            event_callback=self.on_new_data_event
        )
        
        initialize_agents(self.agent_registry, self.config.get('agents', {}))
        self.logger.info(f"Initialized {len(self.agent_registry.get_all_agents())} agents.")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open('zanalytics_config.json', 'r') as f: return json.load(f)
        except Exception as e:
            self.logger.critical(f"Could not load or parse zanalytics_config.json: {e}")
            return {}

    def on_new_data_event(self, event: DataFlowEvent):
        self.logger.info(f"Event received: {event.symbol} {event.timeframe}. Queuing for processing.")
        self.event_queue.put_nowait(event)

    async def event_processor_loop(self):
        self.logger.info("Event processor loop started.")
        while True:
            event = await self.event_queue.get()
            self.logger.info(f"Processing event for {event.symbol}...")
            await self.agent_registry.broadcast_event(event)
            self.event_queue.task_done()

    async def run(self):
        self.data_flow_manager.start_monitoring()
        asyncio.create_task(self.event_processor_loop())
        self.logger.info("ZANALYTICS System is now running.")
        try:
            while True: await asyncio.sleep(3600) # Keep running
        except asyncio.CancelledError:
            self.logger.info("System run task cancelled.")
        finally:
            self.data_flow_manager.stop_monitoring()
            self.logger.info("DataFlowManager stopped.")

# ==============================================================================
# SECTION 3: SYSTEM STARTUP
# ==============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.FileHandler('zanalytics_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║              ZANALYTICS - ALL-IN-ONE                 ║
║                Real-Time Trading Intel               ║
╚══════════════════════════════════════════════════════╝
    """)

async def main():
    print_banner()
    setup_logging()
    
    try:
        system = ZAnalyticsSystem()
        await system.run()
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested by user.")
