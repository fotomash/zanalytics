#!/usr/bin/env python3
"""
ZANFLOW Trading Data API Server - FIXED VERSION
LLM-Optimized REST API for Custom GPT Integration
Serves enriched trading data in LLM-friendly formats
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uvicorn
from pydantic import BaseModel
import sys
import os

app = FastAPI(
    title="ZANFLOW Trading Data API",
    description="LLM-Optimized REST API for trading data consumption",
    version="1.0.0"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TradingDataAPI:
    """API class for serving trading data to LLMs"""

    def __init__(self, data_directory: str = "."):
        self.data_dir = Path(data_directory)
        self.comprehensive_data = {}
        self.json_reports = {}
        self.load_all_data()

    def load_all_data(self):
        """Load all CSV and JSON data for API serving"""
        print("🔄 Loading data for API...")

        # Load comprehensive CSV files
        csv_pattern = str(self.data_dir / "*COMPREHENSIVE*.csv")
        csv_files = glob.glob(csv_pattern)

        print(f"📁 Looking for CSV files in: {self.data_dir.absolute()}")
        print(f"🔍 CSV pattern: {csv_pattern}")
        print(f"📊 Found {len(csv_files)} comprehensive CSV files")

        for file_path in csv_files:
            try:
                file = Path(file_path)
                df = pd.read_csv(file)
                # Extract timeframe from filename
                key = file.stem.replace("XAUUSD_M1_bars_COMPREHENSIVE_", "")
                self.comprehensive_data[key] = df
                print(f"✅ Loaded {key}: {df.shape[0]} rows, {df.shape[1]} indicators")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

        # Load JSON reports
        json_pattern = str(self.data_dir / "*.json")
        json_files = glob.glob(json_pattern)

        print(f"📋 Found {len(json_files)} JSON files")

        for file_path in json_files:
            try:
                file = Path(file_path)
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.json_reports[file.stem] = data
                print(f"✅ Loaded JSON: {file.name}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

        print(f"📊 TOTAL DATA LOADED:")
        print(f"   📈 Timeframes: {len(self.comprehensive_data)}")
        print(f"   📋 Reports: {len(self.json_reports)}")

    def get_latest_analysis(self, pair: str = "XAUUSD", timeframe: str = "1T") -> Dict[str, Any]:
        """Get latest market analysis optimized for LLM consumption"""

        key = timeframe
        if key not in self.comprehensive_data:
            # Try to find the best available timeframe
            available = list(self.comprehensive_data.keys())
            if available:
                key = available[0]
                print(f"⚠️  Timeframe {timeframe} not found, using {key}")
            else:
                raise HTTPException(status_code=404, detail="No data available")

        df = self.comprehensive_data[key]
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="No data points available")

        latest = df.iloc[-1]

        # Create LLM-optimized response
        response = {
            "pair": pair,
            "timeframe": timeframe,
            "actual_timeframe": key,
            "timestamp": latest.get('timestamp', datetime.now().isoformat()),
            "data_points": len(df),
            "price_action": {
                "current_price": float(latest['close']) if 'close' in latest else None,
                "open": float(latest['open']) if 'open' in latest else None,
                "high": float(latest['high']) if 'high' in latest else None,
                "low": float(latest['low']) if 'low' in latest else None,
                "volume": float(latest['volume']) if 'volume' in latest else None
            },
            "technical_indicators": {},
            "market_structure": {},
            "signals": []
        }

        # Add key technical indicators
        key_indicators = ['rsi_14', 'ADX_14', 'sma_21', 'ema_21', 'MACD_12_26_9', 'STOCHk_14_3_3']
        for indicator in key_indicators:
            if indicator in latest.index and pd.notna(latest[indicator]):
                response["technical_indicators"][indicator] = float(latest[indicator])

        # Add market regime analysis
        if 'rsi_14' in latest.index and pd.notna(latest['rsi_14']):
            rsi = latest['rsi_14']
            if rsi > 70:
                regime = "OVERBOUGHT"
            elif rsi > 60:
                regime = "BULLISH_STRONG"
            elif rsi > 50:
                regime = "BULLISH"
            elif rsi > 40:
                regime = "NEUTRAL"
            elif rsi > 30:
                regime = "BEARISH"
            else:
                regime = "OVERSOLD"

            response["market_structure"]["momentum_regime"] = regime
            response["market_structure"]["rsi_percentile"] = self._calculate_percentile(df, 'rsi_14', rsi)

        # Add trend analysis
        if all(col in latest.index for col in ['sma_21', 'ema_21', 'close']):
            price = latest['close']
            sma21 = latest['sma_21']
            ema21 = latest['ema_21']

            if price > sma21 and price > ema21:
                trend = "BULLISH"
            elif price < sma21 and price < ema21:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            response["market_structure"]["trend"] = trend
            response["market_structure"]["price_vs_sma21"] = float(((price - sma21) / sma21) * 100)

        return response

    def _calculate_percentile(self, df: pd.DataFrame, column: str, value: float) -> float:
        """Calculate percentile of current value in historical data"""
        if column in df.columns:
            series = df[column].dropna()
            if len(series) > 0:
                return float((series <= value).mean() * 100)
        return 50.0

    def get_current_signals(self, pair: str = "XAUUSD") -> Dict[str, Any]:
        """Get current trading signals optimized for LLM"""

        signals = []

        # Try to get signals from JSON reports
        for report_name, report_data in self.json_reports.items():
            if isinstance(report_data, dict) and 'signal_summary' in report_data:
                signal_data = report_data['signal_summary']
                signals.append({
                    "source": report_name,
                    "signals": signal_data
                })

        # Generate signals from latest data
        if self.comprehensive_data:
            key = list(self.comprehensive_data.keys())[0]
            df = self.comprehensive_data[key]
            latest = df.iloc[-1]

            # RSI signals
            if 'rsi_14' in latest.index and pd.notna(latest['rsi_14']):
                rsi = latest['rsi_14']
                if rsi < 30:
                    signals.append({
                        "type": "OVERSOLD",
                        "indicator": "RSI",
                        "value": float(rsi),
                        "confidence": 0.8,
                        "action": "POTENTIAL_BUY"
                    })
                elif rsi > 70:
                    signals.append({
                        "type": "OVERBOUGHT", 
                        "indicator": "RSI",
                        "value": float(rsi),
                        "confidence": 0.8,
                        "action": "POTENTIAL_SELL"
                    })

        return {
            "pair": pair,
            "timestamp": datetime.now().isoformat(),
            "signals": signals,
            "signal_count": len(signals)
        }

    def get_indicator_data(self, pair: str = "XAUUSD", timeframe: str = "1T", 
                          indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get specific indicator data for LLM analysis"""

        key = timeframe
        if key not in self.comprehensive_data:
            available = list(self.comprehensive_data.keys())
            key = available[0] if available else None

        if not key:
            raise HTTPException(status_code=404, detail="No data available")

        df = self.comprehensive_data[key]

        if indicators is None:
            # Return key indicators
            indicators = ['rsi_14', 'ADX_14', 'sma_21', 'ema_21', 'MACD_12_26_9', 'STOCHk_14_3_3']

        # Filter available indicators
        available_indicators = [ind for ind in indicators if ind in df.columns]

        if not available_indicators:
            raise HTTPException(status_code=404, detail="No requested indicators found")

        # Get latest values
        latest = df.iloc[-1]
        indicator_values = {}

        for indicator in available_indicators:
            if pd.notna(latest[indicator]):
                indicator_values[indicator] = {
                    "current": float(latest[indicator]),
                    "percentile": self._calculate_percentile(df, indicator, latest[indicator]),
                    "description": self._get_indicator_description(indicator)
                }

        return {
            "pair": pair,
            "timeframe": timeframe,
            "actual_timeframe": key,
            "timestamp": latest.get('timestamp', datetime.now().isoformat()),
            "indicators": indicator_values,
            "total_indicators_available": len(df.columns)
        }

    def _get_indicator_description(self, indicator: str) -> str:
        """Get human-readable description for indicators"""
        descriptions = {
            'rsi_14': 'Relative Strength Index (14-period) - Momentum oscillator (0-100)',
            'ADX_14': 'Average Directional Index (14-period) - Trend strength indicator',
            'sma_21': 'Simple Moving Average (21-period) - Trend-following indicator',
            'ema_21': 'Exponential Moving Average (21-period) - Trend-following indicator',
            'MACD_12_26_9': 'MACD Line - Trend change and momentum indicator',
            'STOCHk_14_3_3': 'Stochastic %K - Momentum oscillator (0-100)'
        }
        return descriptions.get(indicator, f"Technical indicator: {indicator}")

# Initialize API data handler
print("🚀 Initializing ZANFLOW Trading Data API...")
api_data = TradingDataAPI()

# API Endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """API information and health check"""
    return {
        "name": "ZANFLOW Trading Data API",
        "version": "1.0.0",
        "description": "LLM-Optimized REST API for trading data consumption",
        "status": "active",
        "available_timeframes": list(api_data.comprehensive_data.keys()),
        "available_reports": list(api_data.json_reports.keys()),
        "data_loaded": {
            "csv_timeframes": len(api_data.comprehensive_data),
            "json_reports": len(api_data.json_reports)
        },
        "endpoints": [
            "/api/market/latest/{pair}",
            "/api/indicators/{pair}/{timeframe}",
            "/api/signals/current",
            "/api/reports/list",
            "/docs"
        ]
    }

@app.get("/api/market/latest/{pair}", response_model=Dict[str, Any])
async def get_market_latest(
    pair: str,
    timeframe: str = Query(default="1T", description="Timeframe (1T, 5T, 15T, 30T, 1H)")
):
    """Get latest market analysis for specified pair and timeframe"""
    try:
        return api_data.get_latest_analysis(pair, timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals/current", response_model=Dict[str, Any])
async def get_current_signals(pair: str = Query(default="XAUUSD")):
    """Get current trading signals"""
    try:
        return api_data.get_current_signals(pair)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indicators/{pair}/{timeframe}", response_model=Dict[str, Any])
async def get_indicators(
    pair: str,
    timeframe: str,
    indicators: Optional[str] = Query(default=None, description="Comma-separated list of indicators")
):
    """Get specific technical indicators"""
    try:
        indicator_list = indicators.split(',') if indicators else None
        return api_data.get_indicator_data(pair, timeframe, indicator_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/list", response_model=Dict[str, Any])
async def list_reports():
    """List all available JSON reports"""
    return {
        "reports": list(api_data.json_reports.keys()),
        "count": len(api_data.json_reports)
    }

@app.get("/api/reports/{report_name}", response_model=Dict[str, Any])
async def get_report(report_name: str):
    """Get specific JSON report"""
    if report_name not in api_data.json_reports:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "report_name": report_name,
        "data": api_data.json_reports[report_name]
    }

@app.get("/api/timeframes", response_model=Dict[str, Any])
async def get_available_timeframes():
    """Get all available timeframes and their data info"""
    timeframes = {}

    for tf, df in api_data.comprehensive_data.items():
        timeframes[tf] = {
            "rows": len(df),
            "indicators": len(df.columns),
            "date_range": {
                "start": df['timestamp'].min() if 'timestamp' in df.columns else None,
                "end": df['timestamp'].max() if 'timestamp' in df.columns else None
            }
        }

    return {
        "available_timeframes": timeframes,
        "total_timeframes": len(timeframes)
    }

def main():
    """Main function to run the API server"""
    print("🚀 Starting ZANFLOW Trading Data API Server...")
    print("📊 LLM-Optimized REST API for Custom GPT Integration")
    print("🔗 Access API docs at: http://localhost:8000/docs")
    print("🌐 API root: http://localhost:8000/")
    print()

    # Run the server
    uvicorn.run(
        "zanflow_trading_api_server_fixed:app",  # Import string format
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid the warning
        log_level="info"
    )

if __name__ == "__main__":
    main()
