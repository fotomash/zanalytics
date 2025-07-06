#!/usr/bin/env python3
"""
🌐 WebSocket Server - Real-time Data Broadcasting
Streams enriched trading data to dashboard and clients
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class WebSocketServer:
    """Real-time data broadcasting via WebSocket"""

    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.data_cache = {}

    async def register_client(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        logger.info(f"📱 Client connected: {websocket.remote_address}")

        # Send initial data
        await self.send_initial_data(websocket)

    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        logger.info(f"📱 Client disconnected: {websocket.remote_address}")

    async def send_initial_data(self, websocket):
        """Send current data to newly connected client"""
        try:
            # Load latest enriched data
            data = self.load_latest_data()
            if data:
                await websocket.send(json.dumps({
                    'type': 'initial_data',
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }))
        except Exception as e:
            logger.error(f"❌ Error sending initial data: {e}")

    def load_latest_data(self):
        """Load the latest enriched trading data"""
        try:
            # Look for the most recent processed file
            data_dir = Path('data/processed')
            if not data_dir.exists():
                return None

            csv_files = list(data_dir.glob('*_processed.csv'))
            if not csv_files:
                return None

            # Get the most recent file
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)

            # Load last 100 rows for real-time display
            df = pd.read_csv(latest_file).tail(100)

            return {
                'symbol': 'XAUUSD',
                'timeframe': '1M',
                'data': df.to_dict('records'),
                'file': str(latest_file.name)
            }

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None

    async def broadcast_data(self, data):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return

        message = json.dumps({
            'type': 'data_update',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })

        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"❌ Error broadcasting to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self.clients -= disconnected

    async def broadcast_signal(self, signal):
        """Broadcast trading signal to all clients"""
        if not self.clients:
            return

        message = json.dumps({
            'type': 'trading_signal',
            'signal': signal,
            'timestamp': datetime.now().isoformat()
        })

        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
                logger.info(f"📡 Signal sent to client: {signal['type']}")
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"❌ Error sending signal: {e}")
                disconnected.add(client)

        self.clients -= disconnected

    async def handle_client_message(self, websocket, message):
        """Handle incoming messages from clients"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'subscribe':
                # Client wants to subscribe to specific symbol/timeframe
                symbol = data.get('symbol', 'XAUUSD')
                timeframe = data.get('timeframe', '1M')

                response = {
                    'type': 'subscription_confirmed',
                    'symbol': symbol,
                    'timeframe': timeframe
                }
                await websocket.send(json.dumps(response))

            elif msg_type == 'request_data':
                # Client requests current data
                current_data = self.load_latest_data()
                if current_data:
                    await websocket.send(json.dumps({
                        'type': 'data_response',
                        'data': current_data
                    }))

        except Exception as e:
            logger.error(f"❌ Error handling client message: {e}")

    async def client_handler(self, websocket, path):
        """Handle WebSocket client connections"""
        await self.register_client(websocket)

        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"❌ Client handler error: {e}")
        finally:
            await self.unregister_client(websocket)

    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"🌐 Starting WebSocket server on {self.host}:{self.port}")

        # Start data monitoring in background
        asyncio.create_task(self.monitor_data_changes())

        # Start WebSocket server
        await websockets.serve(self.client_handler, self.host, self.port)
        logger.info(f"✅ WebSocket server running on ws://{self.host}:{self.port}")

    async def monitor_data_changes(self):
        """Monitor for data file changes and broadcast updates"""
        last_check = {}

        while True:
            try:
                data_dir = Path('data/processed')
                if data_dir.exists():
                    for csv_file in data_dir.glob('*_processed.csv'):
                        current_mtime = csv_file.stat().st_mtime

                        if csv_file.name not in last_check or last_check[csv_file.name] < current_mtime:
                            # File has been updated
                            last_check[csv_file.name] = current_mtime

                            # Load and broadcast new data
                            data = self.load_latest_data()
                            if data:
                                await self.broadcast_data(data)
                                logger.info(f"📡 Broadcasted data update from {csv_file.name}")

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"❌ Error monitoring data changes: {e}")
                await asyncio.sleep(10)

# Global server instance
websocket_server_instance = None

def get_websocket_server():
    """Get the global WebSocket server instance"""
    global websocket_server_instance
    if websocket_server_instance is None:
        websocket_server_instance = WebSocketServer()
    return websocket_server_instance

async def broadcast_trading_signal(signal):
    """Utility function to broadcast trading signals"""
    server = get_websocket_server()
    await server.broadcast_signal(signal)
