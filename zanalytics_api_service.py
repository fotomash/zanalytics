# zanalytics_api_service.py - Fixed with correct DataFlowManager interface
from flask import Flask, jsonify, request, g
from flask_cors import CORS
import asyncio
import threading
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Import your existing systems
from data_flow_manager import DataFlowManager, DataFlowEvent
from ncos_agent_system import ncOSAgentOrchestrator

app = Flask(__name__)
CORS(app)


# Global state for the API service
class ZAnalyticsAPIState:
    def __init__(self):
        # Data storage
        self.current_data = {}
        self.agent_responses = {}
        self.session_state = {}
        self.analysis_history = []

        # System components
        self.agent_orchestrator = None
        self.data_flow_manager = None

        # Cache for fast access
        self.latest_tick_data = {}
        self.latest_ohlc_data = {}
        self.latest_analysis = {}

    def update_data(self, symbol: str, timeframe: str, data: dict):
        """Update cached data for fast API access"""
        key = f"{symbol}_{timeframe}"
        self.current_data[key] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'processed': True
        }


# Global state instance
api_state = ZAnalyticsAPIState()


# ===============================================================================
# CORE API ENDPOINTS - EVERYTHING YOUR GPT MODELS NEED
# ===============================================================================

@app.route('/status', methods=['GET'])
def get_system_status():
    """Complete system status for LLM awareness"""
    return jsonify({
        'system': 'zanalytics_intelligence_api',
        'version': '1.0.0',
        'status': 'operational',
        'data_feeds': {
            'active_symbols': list(api_state.latest_tick_data.keys()),
            'active_timeframes': list(set([d['timeframe'] for d in api_state.current_data.values()])),
            'last_update': max(
                [d['timestamp'] for d in api_state.current_data.values()]) if api_state.current_data else None
        },
        'agents': {
            'active_agents': ['bozenka', 'stefania', 'lusia', 'zdzisiek', 'rysiek'],
            'last_consensus': api_state.session_state.get('last_consensus', None)
        },
        'data_manager': api_state.data_flow_manager.get_data_status() if api_state.data_flow_manager else {},
        'session_metrics': {
            'events_processed': api_state.session_state.get('event_count', 0),
            'analysis_count': len(api_state.analysis_history)
        }
    })


@app.route('/data/latest/<symbol>', methods=['GET'])
def get_latest_data(symbol):
    """Get latest processed data for a symbol - ALL TIMEFRAMES"""
    symbol = symbol.upper()

    # Get all timeframes for this symbol
    symbol_data = {k: v for k, v in api_state.current_data.items() if k.startswith(symbol)}

    if not symbol_data:
        return jsonify({'error': f'No data available for {symbol}'}), 404

    return jsonify({
        'symbol': symbol,
        'timeframes': symbol_data,
        'tick_data': api_state.latest_tick_data.get(symbol, {}),
        'latest_analysis': api_state.latest_analysis.get(symbol, {}),
        'metadata': {
            'available_timeframes': [k.split('_')[1] for k in symbol_data.keys()],
            'data_freshness': min([v['timestamp'] for v in symbol_data.values()])
        }
    })


@app.route('/data/tick/<symbol>', methods=['GET'])
def get_tick_data(symbol):
    """Raw tick data - perfect for microstructure analysis prompts"""
    symbol = symbol.upper()
    tick_data = api_state.latest_tick_data.get(symbol, {})

    if not tick_data:
        return jsonify({'error': f'No tick data for {symbol}'}), 404

    return jsonify({
        'symbol': symbol,
        'data_type': 'tick',
        'tick_data': tick_data,
        'spread_analysis': tick_data.get('spread_metrics', {}),
        'microstructure': tick_data.get('microstructure_signals', {}),
        'timestamp': tick_data.get('timestamp')
    })


@app.route('/data/ohlc/<symbol>/<timeframe>', methods=['GET'])
def get_ohlc_data(symbol, timeframe):
    """OHLC data with all indicators and patterns"""
    symbol = symbol.upper()
    timeframe = timeframe.upper()
    key = f"{symbol}_{timeframe}"

    data = api_state.current_data.get(key, {})
    if not data:
        return jsonify({'error': f'No OHLC data for {symbol}/{timeframe}'}), 404

    return jsonify({
        'symbol': symbol,
        'timeframe': timeframe,
        'ohlc_data': data['data'],
        'indicators': data['data'].get('indicators', {}),
        'patterns': data['data'].get('patterns', {}),
        'smc_analysis': data['data'].get('smc', {}),
        'wyckoff_analysis': data['data'].get('wyckoff', {}),
        'timestamp': data['timestamp']
    })


@app.route('/agents/decisions', methods=['GET'])
def get_agent_decisions():
    """Latest decisions from all agents - PERFECT FOR GPT PROMPTS"""
    return jsonify({
        'agents': api_state.agent_responses,
        'consensus': api_state.session_state.get('last_consensus', {}),
        'decision_summary': {
            'bozenka_signal': api_state.agent_responses.get('bozenka', {}).get('entry_signal', False),
            'stefania_trust': api_state.agent_responses.get('stefania', {}).get('trust_score', 0.5),
            'lusia_confluence': api_state.agent_responses.get('lusia', {}).get('confluence_score', 0.0),
            'zdzisiek_risk': api_state.agent_responses.get('zdzisiek', {}).get('risk_acceptable', True),
            'rysiek_phase': api_state.agent_responses.get('rysiek', {}).get('phase', 'unknown')
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/analysis/summary/<symbol>', methods=['GET'])
def get_analysis_summary(symbol):
    """Complete analysis summary - ONE CALL FOR EVERYTHING"""
    symbol = symbol.upper()

    # Gather all data for this symbol
    latest_data = {}
    for k, v in api_state.current_data.items():
        if k.startswith(symbol):
            timeframe = k.split('_')[1]
            latest_data[timeframe] = v['data']

    tick_data = api_state.latest_tick_data.get(symbol, {})
    analysis = api_state.latest_analysis.get(symbol, {})

    return jsonify({
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'data_available': list(latest_data.keys()),

        # Microstructure (from tick data)
        'microstructure': {
            'spread': tick_data.get('spread_metrics', {}),
            'liquidity': tick_data.get('liquidity_signals', {}),
            'manipulation': tick_data.get('manipulation_detected', False)
        },

        # Multi-timeframe structure
        'structure_analysis': {tf: data.get('smc', {}) for tf, data in latest_data.items()},

        # Wyckoff phases across timeframes
        'wyckoff_phases': {tf: data.get('wyckoff', {}) for tf, data in latest_data.items()},

        # Technical indicators
        'indicators': {tf: data.get('indicators', {}) for tf, data in latest_data.items()},

        # Agent consensus
        'agent_consensus': api_state.session_state.get('last_consensus', {}),

        # Trade decision support
        'decision_support': {
            'entry_signals': [agent for agent, resp in api_state.agent_responses.items()
                              if resp.get('entry_signal', False)],
            'risk_level': api_state.agent_responses.get('zdzisiek', {}).get('risk_score', 0.5),
            'confluence_score': api_state.agent_responses.get('lusia', {}).get('confluence_score', 0.0),
            'trust_score': api_state.agent_responses.get('stefania', {}).get('trust_score', 0.5)
        }
    })


@app.route('/data/upload', methods=['POST'])
def upload_data():
    """Upload CSV/JSON data for processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Create uploads directory if it doesn't exist
    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)

    # Save file and trigger processing
    file_path = upload_dir / file.filename
    file.save(str(file_path))

    return jsonify({
        'status': 'uploaded',
        'filename': file.filename,
        'processing': 'monitoring will detect automatically'
    })


# ===============================================================================
# EVENT PROCESSING INTEGRATION - FIXED TO USE CORRECT INTERFACE
# ===============================================================================

async def process_data_event(event: DataFlowEvent):
    """Process incoming data events and update API state"""
<<<<<<< HEAD
    try:
        # Update cached data
        api_state.update_data(event.symbol, event.timeframe, event.data)

        # Store tick data separately for microstructure analysis
        if event.source == 'csv' and event.data.get('data_type') == 'tick':
            api_state.latest_tick_data[event.symbol] = event.data

        # Process through agents if orchestrator is available
        if api_state.agent_orchestrator:
            agent_event = {
                'event_type': event.event_type,
                'symbol': event.symbol,
                'timeframe': event.timeframe,
                'data': event.data.get('analysis', event.data),  # Use analysis if available
                'indicators': event.data.get('indicators', {}),
                'risk_metrics': {'spread': 0.2, 'volatility': 0.3},  # Default values
                'wyckoff': {'phase': 'unknown', 'confidence': 0.5}  # Default values
            }

            agent_result = await api_state.agent_orchestrator.process_event(agent_event)

            # Update agent responses
            api_state.agent_responses = agent_result.get('agent_responses', {})
            api_state.session_state['last_consensus'] = agent_result.get('consensus', {})

            # Add to history
            api_state.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': event.symbol,
                'timeframe': event.timeframe,
                'agent_results': agent_result,
                'event_data': event.data
            })

            print(f"✅ Processed {event.symbol} {event.timeframe} through agents")

    except Exception as e:
        print(f"❌ Error processing data event: {e}")
        # Continue rather than crash


def setup_data_integration():
    """Initialize data flow integration - FIXED"""
    config = {
        'watch_directories': ['./data', './exports', './uploads'],
        'agents': {
            'bozenka': {'active': True},
            'stefania': {'active': True},
            'lusia': {'active': True},
            'zdzisiek': {'active': True},
            'rysiek': {'active': True}
        }
    }
<<<<<<< HEAD

    # Create directories if they don't exist
    for directory in config['watch_directories']:
        Path(directory).mkdir(exist_ok=True)

    try:
        # Initialize agent orchestrator
        api_state.agent_orchestrator = ncOSAgentOrchestrator(config)
        print(f"✅ Agent orchestrator initialized")

        # Initialize data flow manager with CORRECT parameter name
        api_state.data_flow_manager = DataFlowManager(
            watch_directories=config['watch_directories'],
            zanalytics_callback=process_data_event  # CORRECT parameter name!
        )

        # Start monitoring
        api_state.data_flow_manager.start_monitoring()
        print(f"✅ Data flow manager started, monitoring: {config['watch_directories']}")

    except Exception as e:
        print(f"❌ Error setting up data integration: {e}")
        print(f"API will run in basic mode without data monitoring")


# ===============================================================================
# STARTUP
# ===============================================================================

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    print("""
╔══════════════════════════════════════════════════════╗
║           ZANALYTICS INTELLIGENCE API v1.0           ║
║              Data Intelligence for GPT Models        ║
╠══════════════════════════════════════════════════════╣
║  🚀 Starting API Service...                          ║
╚══════════════════════════════════════════════════════╝
    """)

    # Initialize integrations
    setup_data_integration()

    print("""
╔══════════════════════════════════════════════════════╗
║  📊 Real-time Data Processing Active                 ║
║  🤖 Agent Decisions Available                        ║
║  ⚡ Ready for LLM Integration                        ║
╚══════════════════════════════════════════════════════╝

API Endpoints for your GPT models:
    GET  /status                    - System status
    GET  /data/latest/{symbol}      - Latest all data
    GET  /data/tick/{symbol}        - Tick data
    GET  /data/ohlc/{symbol}/{tf}   - OHLC with indicators
    GET  /agents/decisions          - Agent decisions
    GET  /analysis/summary/{symbol} - Complete analysis

Running on http://0.0.0.0:5010
    """)

    app.run(host='0.0.0.0', port=5010, debug=False, threaded=True)