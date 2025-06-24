# zanalytics_api_service.py - Complete Data Intelligence API
"""Comprehensive Flask service exposing real-time market data, agent
consensus, and historical analytics for GPT-based trading models.

This module configures routes for fetching cached tick and OHLC data,
retrieving aggregated agent decisions, and managing configuration for the
analytics pipeline. It integrates :class:`DataFlowManager` for
filesystem-driven event processing and an :class:`ncOSAgentOrchestrator`
for generating automated trade analysis. The service maintains a global
:class:`ZAnalyticsAPIState` which stores the most recent data snapshots
and agent outputs to serve responses efficiently to language models or
other clients.
"""
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
            'last_update': max([d['timestamp'] for d in api_state.current_data.values()]) if api_state.current_data else None
        },
        'agents': {
            'active_agents': ['bozenka', 'stefania', 'lusia', 'zdzisiek', 'rysiek'],
            'last_consensus': api_state.session_state.get('last_consensus', None)
        },
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

@app.route('/analysis/historical/<symbol>', methods=['GET'])
def get_historical_analysis(symbol):
    """Historical analysis for pattern recognition and backtesting"""
    symbol = symbol.upper()
    
    # Filter history for this symbol
    symbol_history = [
        entry for entry in api_state.analysis_history 
        if entry.get('symbol') == symbol
    ]
    
    return jsonify({
        'symbol': symbol,
        'historical_entries': len(symbol_history),
        'analysis_history': symbol_history[-50:],  # Last 50 entries
        'patterns': {
            'successful_setups': [entry for entry in symbol_history if entry.get('success', False)],
            'failed_setups': [entry for entry in symbol_history if entry.get('success') == False],
            'win_rate': len([e for e in symbol_history if e.get('success')]) / max(len(symbol_history), 1)
        }
    })

# ===============================================================================
# CONFIGURATION AND MANAGEMENT ENDPOINTS
# ===============================================================================

@app.route('/config/agents', methods=['GET', 'POST'])
def manage_agent_config():
    """Configure agent behavior"""
    if request.method == 'POST':
        new_config = request.json
        # Update agent configuration
        api_state.session_state['agent_config'] = new_config
        return jsonify({'status': 'updated', 'config': new_config})
    else:
        return jsonify({
            'current_config': api_state.session_state.get('agent_config', {}),
            'available_agents': ['bozenka', 'stefania', 'lusia', 'zdzisiek', 'rysiek']
        })

@app.route('/data/upload', methods=['POST'])
def upload_data():
    """Upload CSV/JSON data for processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Save file and trigger processing
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)
    
    # Trigger data flow processing
    # This would integrate with your DataFlowManager
    
    return jsonify({
        'status': 'uploaded',
        'filename': file.filename,
        'processing': 'started'
    })

# ===============================================================================
# WEBSOCKET/STREAMING ENDPOINTS (for real-time updates)
# ===============================================================================

@app.route('/stream/subscribe/<symbol>', methods=['POST'])
def subscribe_to_symbol(symbol):
    """Subscribe to real-time updates for a symbol"""
    symbol = symbol.upper()
    subscribers = api_state.session_state.setdefault('subscribers', set())
    subscribers.add(symbol)
    
    return jsonify({
        'status': 'subscribed',
        'symbol': symbol,
        'active_subscriptions': list(subscribers)
    })

# ===============================================================================
# EVENT PROCESSING INTEGRATION
# ===============================================================================

async def process_data_event(event: DataFlowEvent):
    """Process incoming data events and update API state"""
    # Update cached data
    api_state.update_data(event.symbol, event.timeframe, event.data)
    
    # Store tick data separately
    if event.source == 'csv' and event.data.get('data_type') == 'tick':
        api_state.latest_tick_data[event.symbol] = event.data
    
    # Process through agents if orchestrator is available
    if api_state.agent_orchestrator:
        agent_result = await api_state.agent_orchestrator.process_event({
            'event_type': event.event_type,
            'symbol': event.symbol,
            'timeframe': event.timeframe,
            'data': event.data
        })
        
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

def setup_data_integration():
    """Initialize data flow integration"""
    config = {
        'watch_directories': ['./data', './exports'],
        'agents': {
            'bozenka': {'active': True},
            'stefania': {'active': True},
            'lusia': {'active': True},
            'zdzisiek': {'active': True},
            'rysiek': {'active': True}
        }
    }
    
    # Initialize agent orchestrator
    from ncos_agent_system import ncOSAgentOrchestrator
    api_state.agent_orchestrator = ncOSAgentOrchestrator(config)
    
    # Initialize data flow manager
    api_state.data_flow_manager = DataFlowManager(
        watch_directories=config['watch_directories'],
        event_callback=lambda event: asyncio.create_task(process_data_event(event))
    )
    
    # Start monitoring
    api_state.data_flow_manager.start_monitoring()

# ===============================================================================
# STARTUP
# ===============================================================================

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize integrations
    setup_data_integration()
    
    print("""
╔══════════════════════════════════════════════════════╗
║           ZANALYTICS INTELLIGENCE API v1.0           ║
║              Data Intelligence for GPT Models        ║
╠══════════════════════════════════════════════════════╣
║  🚀 API Endpoints Ready                              ║
║  📊 Real-time Data Processing                        ║
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
    GET  /analysis/historical/{symbol} - Historical patterns
    
Running on http://0.0.0.0:5010
    """)
    
    app.run(host='0.0.0.0', port=5010, debug=False)
