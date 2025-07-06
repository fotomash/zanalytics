#!/usr/bin/env python3
"""
🔗 REST API - Data Access Endpoints
HTTP API for historical data and system configuration
"""

import asyncio
from aiohttp import web, web_request
import json
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RestAPI:
    """REST API server for data access and configuration"""

    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get('/api/health', self.health_check)
        self.app.router.add_get('/api/data/{symbol}/{timeframe}', self.get_data)
        self.app.router.add_get('/api/signals/{symbol}', self.get_signals)
        self.app.router.add_get('/api/analysis/{symbol}', self.get_analysis)
        self.app.router.add_post('/api/config', self.update_config)
        self.app.router.add_get('/api/config', self.get_config)
        self.app.router.add_get('/api/status', self.get_system_status)

        # Enable CORS
        self.app.middlewares.append(self.cors_middleware)

    @web.middleware
    async def cors_middleware(self, request, handler):
        """Enable CORS for all requests"""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })

    async def get_data(self, request):
        """Get historical data for symbol/timeframe"""
        symbol = request.match_info['symbol']
        timeframe = request.match_info['timeframe']

        # Query parameters
        limit = int(request.query.get('limit', 1000))
        start_date = request.query.get('start_date')
        end_date = request.query.get('end_date')

        try:
            # Look for processed data files
            data_dir = Path('data/processed')
            pattern = f"{symbol}*{timeframe}*processed.csv"

            matching_files = list(data_dir.glob(pattern))
            if not matching_files:
                return web.json_response({
                    'error': f'No data found for {symbol} {timeframe}'
                }, status=404)

            # Load the most recent file
            latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)

            # Apply filters
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]

            # Limit results
            df = df.tail(limit)

            return web.json_response({
                'symbol': symbol,
                'timeframe': timeframe,
                'data': df.to_dict('records'),
                'count': len(df),
                'file': str(latest_file.name)
            })

        except Exception as e:
            logger.error(f"❌ Error getting data: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)

    async def get_signals(self, request):
        """Get trading signals for symbol"""
        symbol = request.match_info['symbol']

        try:
            # Look for analysis reports
            data_dir = Path('data/processed/analysis_reports')
            if not data_dir.exists():
                return web.json_response({'signals': []})

            signal_files = list(data_dir.glob(f"{symbol}*signals*.json"))
            if not signal_files:
                return web.json_response({'signals': []})

            # Load latest signals
            latest_file = max(signal_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                signals = json.load(f)

            return web.json_response({
                'symbol': symbol,
                'signals': signals,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"❌ Error getting signals: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)

    async def get_analysis(self, request):
        """Get analysis report for symbol"""
        symbol = request.match_info['symbol']

        try:
            # Look for analysis reports
            analysis_file = Path(f'data/processed/analysis_reports/{symbol}_analysis.json')
            if not analysis_file.exists():
                return web.json_response({
                    'error': f'No analysis found for {symbol}'
                }, status=404)

            with open(analysis_file, 'r') as f:
                analysis = json.load(f)

            return web.json_response({
                'symbol': symbol,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"❌ Error getting analysis: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)

    async def get_config(self, request):
        """Get system configuration"""
        try:
            with open('config/settings.json', 'r') as f:
                config = json.load(f)
            return web.json_response(config)
        except Exception as e:
            return web.json_response({
                'error': str(e)
            }, status=500)

    async def update_config(self, request):
        """Update system configuration"""
        try:
            new_config = await request.json()

            # Save updated config
            with open('config/settings.json', 'w') as f:
                json.dump(new_config, f, indent=2)

            return web.json_response({
                'status': 'success',
                'message': 'Configuration updated'
            })
        except Exception as e:
            return web.json_response({
                'error': str(e)
            }, status=500)

    async def get_system_status(self, request):
        """Get system status"""
        try:
            # Check data pipeline status
            data_dir = Path('data/processed')
            latest_data = None
            if data_dir.exists():
                csv_files = list(data_dir.glob('*_processed.csv'))
                if csv_files:
                    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                    latest_data = {
                        'file': str(latest_file.name),
                        'modified': datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
                    }

            return web.json_response({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'data_pipeline': 'running' if latest_data else 'no_data',
                    'websocket_server': 'running',
                    'rest_api': 'running'
                },
                'latest_data': latest_data
            })

        except Exception as e:
            return web.json_response({
                'error': str(e)
            }, status=500)

    async def start(self):
        """Start the REST API server"""
        logger.info(f"🔗 Starting REST API on {self.host}:{self.port}")

        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"✅ REST API running on http://{self.host}:{self.port}")
