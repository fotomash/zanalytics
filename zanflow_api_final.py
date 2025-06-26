#!/usr/bin/env python3
"""
ZANFLOW Trading Data API Server - FINAL WORKING VERSION
LLM-Optimized REST API for Custom GPT Integration
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import uvicorn

app = FastAPI(
    title="ZANFLOW Trading Data API",
    description="LLM-Optimized REST API for trading data consumption",
    version="1.0.0"
)

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
        print(f"📁 Data directory: {self.data_dir.absolute()}")

        # Load comprehensive CSV files
        csv_files = list(self.data_dir.glob("*COMPREHENSIVE*.csv"))
        print(f"🎯 Found {len(csv_files)} comprehensive CSV files")

        for file in csv_files:
            try:
                print(f"📊 Loading {file.name}...")
                df = pd.read_csv(file)

                # Extract timeframe from filename
                key = file.stem.replace("XAUUSD_M1_bars_COMPREHENSIVE_", "")
                self.comprehensive_data[key] = df
                print(f"✅ Loaded {key}: {df.shape[0]} rows, {df.shape[1]} indicators")

            except Exception as e:
                print(f"❌ Error loading {file.name}: {e}")

        # Load JSON reports
        json_files = list(self.data_dir.glob("*.json"))
        print(f"📋 Found {len(json_files)} JSON files")

        for file in json_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.json_reports[file.stem] = data
                print(f"✅ Loaded JSON: {file.name}")
            except Exception as e:
                print(f"❌ Error loading JSON {file.name}: {e}")

        print(f"\n📊 FINAL DATA LOADED:")
        print(f"   📈 CSV Timeframes: {len(self.comprehensive_data)}")
        print(f"   📋 JSON Reports: {len(self.json_reports)}")

        if len(self.comprehensive_data) == 0:
            print("⚠️  WARNING: No CSV data loaded!")
        else:
            print(f"✅ Available timeframes: {list(self.comprehensive_data.keys())}")

    def get_latest_analysis(self, pair: str = "XAUUSD", timeframe: str = "1T") -> Dict[str, Any]:
        """Get latest market analysis optimized for LLM consumption"""

        if not self.comprehensive_data:
            raise HTTPException(status_code=404, detail="No comprehensive data available")

        # Find the requested timeframe or use best available
        key = timeframe
        if key not in self.comprehensive_data:
            available = list(self.comprehensive_data.keys())
            key = available[0] if available else None
            print(f"⚠️  Timeframe {timeframe} not found, using {key}")

        if not key:
            raise HTTPException(status_code=404, detail="No data available")

        df = self.comprehensive_data[key]
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="No data points available")

        latest = df.iloc[-1]

        # Create LLM-optimized response
        response = {
            "pair": pair,
            "timeframe": timeframe,
            "actual_timeframe_used": key,
            "timestamp": str(latest.get('timestamp', datetime.now().isoformat())),
            "data_points": len(df),
            "price_action": {},
            "technical_indicators": {},
            "market_structure": {},
            "analysis_summary": ""
        }

        # Add price data safely
        price_fields = ['close', 'open', 'high', 'low', 'volume']
        for field in price_fields:
            if field in df.columns and pd.notna(latest[field]):
                response["price_action"][field] = float(latest[field])

        # Add key technical indicators safely
        key_indicators = ['rsi_14', 'ADX_14', 'sma_21', 'ema_21', 'MACD_12_26_9', 'STOCHk_14_3_3']
        for indicator in key_indicators:
            if indicator in df.columns and pd.notna(latest[indicator]):
                response["technical_indicators"][indicator] = {
                    "value": float(latest[indicator]),
                    "percentile": self._calculate_percentile(df, indicator, latest[indicator])
                }

        # Generate market structure analysis
        response["market_structure"] = self._analyze_market_structure(df, latest)

        # Create analysis summary for LLM
        response["analysis_summary"] = self._generate_analysis_summary(response)

        return response

    def _calculate_percentile(self, df: pd.DataFrame, column: str, value: float) -> float:
        """Calculate percentile of current value in historical data"""
        try:
            if column in df.columns:
                series = df[column].dropna()
                if len(series) > 0:
                    return float((series <= value).mean() * 100)
        except:
            pass
        return 50.0

    def _analyze_market_structure(self, df: pd.DataFrame, latest: pd.Series) -> Dict[str, Any]:
        """Analyze market structure for LLM"""
        structure = {}

        # RSI analysis
        if 'rsi_14' in df.columns and pd.notna(latest['rsi_14']):
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

            structure["momentum_regime"] = regime
            structure["rsi_reading"] = float(rsi)

        # Trend analysis
        if all(col in df.columns for col in ['sma_21', 'close']):
            if pd.notna(latest['close']) and pd.notna(latest['sma_21']):
                price = latest['close']
                sma21 = latest['sma_21']

                if price > sma21:
                    trend = "BULLISH"
                    trend_strength = ((price - sma21) / sma21) * 100
                else:
                    trend = "BEARISH"
                    trend_strength = ((price - sma21) / sma21) * 100

                structure["trend"] = trend
                structure["trend_strength_percent"] = float(trend_strength)

        return structure

    def _generate_analysis_summary(self, response: Dict[str, Any]) -> str:
        """Generate human-readable summary for LLM"""
        summary_parts = []

        # Price summary
        if "close" in response["price_action"]:
            price = response["price_action"]["close"]
            summary_parts.append(f"Current price: ${price:.2f}")

        # Momentum summary
        market_structure = response.get("market_structure", {})
        if "momentum_regime" in market_structure:
            regime = market_structure["momentum_regime"]
            rsi = market_structure.get("rsi_reading", 0)
            summary_parts.append(f"Momentum: {regime} (RSI: {rsi:.1f})")

        # Trend summary
        if "trend" in market_structure:
            trend = market_structure["trend"]
            strength = market_structure.get("trend_strength_percent", 0)
            summary_parts.append(f"Trend: {trend} ({strength:+.2f}% vs SMA21)")

        return " | ".join(summary_parts) if summary_parts else "Market analysis available"

    def get_current_signals(self, pair: str = "XAUUSD") -> Dict[str, Any]:
        """Get current trading signals optimized for LLM"""
        signals = []

        # Generate signals from latest data if available
        if self.comprehensive_data:
            key = list(self.comprehensive_data.keys())[0]
            df = self.comprehensive_data[key]
            latest = df.iloc[-1]

            # RSI signals
            if 'rsi_14' in df.columns and pd.notna(latest['rsi_14']):
                rsi = latest['rsi_14']
                if rsi < 30:
                    signals.append({
                        "type": "OVERSOLD_SIGNAL",
                        "indicator": "RSI_14",
                        "value": float(rsi),
                        "confidence": 0.8,
                        "action": "POTENTIAL_BUY",
                        "description": f"RSI at {rsi:.1f} indicates oversold conditions"
                    })
                elif rsi > 70:
                    signals.append({
                        "type": "OVERBOUGHT_SIGNAL",
                        "indicator": "RSI_14", 
                        "value": float(rsi),
                        "confidence": 0.8,
                        "action": "POTENTIAL_SELL",
                        "description": f"RSI at {rsi:.1f} indicates overbought conditions"
                    })

        # Add signals from JSON reports
        signal_count_from_reports = 0
        for report_name, report_data in self.json_reports.items():
            if isinstance(report_data, dict):
                if 'signals' in report_data:
                    signals.extend(report_data['signals'])
                    signal_count_from_reports += len(report_data['signals'])

        return {
            "pair": pair,
            "timestamp": datetime.now().isoformat(),
            "signals": signals,
            "signal_count": len(signals),
            "sources": {
                "technical_analysis": len(signals) - signal_count_from_reports,
                "json_reports": signal_count_from_reports
            }
        }

# Initialize API data handler
print("🚀 Initializing ZANFLOW Trading Data API...")
api_data = TradingDataAPI()

# API Endpoints
@app.get("/")
async def root():
    """API information and health check"""
    return {
        "name": "ZANFLOW Trading Data API",
        "version": "1.0.0",
        "description": "LLM-Optimized REST API for trading data consumption",
        "status": "active",
        "data_status": {
            "csv_timeframes_loaded": len(api_data.comprehensive_data),
            "json_reports_loaded": len(api_data.json_reports),
            "available_timeframes": list(api_data.comprehensive_data.keys())
        },
        "endpoints": [
            "/api/market/latest/XAUUSD",
            "/api/indicators/XAUUSD/1T", 
            "/api/signals/current",
            "/api/timeframes",
            "/docs"
        ]
    }

@app.get("/api/market/latest/{pair}")
async def get_market_latest(
    pair: str,
    timeframe: str = Query(default="1T", description="Timeframe (1T, 5T, 15T, 30T, 1H)")
):
    """Get latest market analysis for specified pair and timeframe"""
    try:
        return api_data.get_latest_analysis(pair, timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/signals/current")
async def get_current_signals(pair: str = Query(default="XAUUSD")):
    """Get current trading signals"""
    try:
        return api_data.get_current_signals(pair)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/timeframes")
async def get_available_timeframes():
    """Get all available timeframes and their data info"""
    timeframes = {}

    for tf, df in api_data.comprehensive_data.items():
        timeframes[tf] = {
            "rows": len(df),
            "indicators": len(df.columns),
            "latest_timestamp": str(df['timestamp'].max()) if 'timestamp' in df.columns else None,
            "date_range_days": None
        }

        # Calculate date range if possible
        if 'timestamp' in df.columns:
            try:
                start_date = pd.to_datetime(df['timestamp'].min())
                end_date = pd.to_datetime(df['timestamp'].max())
                timeframes[tf]["date_range_days"] = (end_date - start_date).days
            except:
                pass

    return {
        "available_timeframes": timeframes,
        "total_timeframes": len(timeframes),
        "total_data_points": sum(len(df) for df in api_data.comprehensive_data.values())
    }

@app.get("/api/indicators/{pair}/{timeframe}")
async def get_indicators(
    pair: str,
    timeframe: str,
    limit: int = Query(default=50, description="Number of recent data points")
):
    """Get recent indicator data for analysis"""
    try:
        if timeframe not in api_data.comprehensive_data:
            available = list(api_data.comprehensive_data.keys())
            if not available:
                raise HTTPException(status_code=404, detail="No data available")
            timeframe = available[0]

        df = api_data.comprehensive_data[timeframe]
        recent_data = df.tail(limit)

        # Get key indicators
        key_indicators = ['rsi_14', 'ADX_14', 'sma_21', 'ema_21', 'MACD_12_26_9']
        available_indicators = [ind for ind in key_indicators if ind in df.columns]

        indicator_data = {}
        for indicator in available_indicators:
            series = recent_data[indicator].dropna()
            if len(series) > 0:
                indicator_data[indicator] = {
                    "current": float(series.iloc[-1]),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "trend": "rising" if len(series) > 1 and series.iloc[-1] > series.iloc[-2] else "falling"
                }

        return {
            "pair": pair,
            "timeframe": timeframe,
            "data_points": len(recent_data),
            "indicators": indicator_data,
            "total_indicators_available": len(df.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def main():
    """Main function to run the API server"""
    print("\n🚀 Starting ZANFLOW Trading Data API Server...")
    print("📊 LLM-Optimized REST API for Custom GPT Integration")
    print("🔗 API Documentation: http://localhost:8000/docs")
    print("🌐 API Status: http://localhost:8000/")
    print("📈 Market Data: http://localhost:8000/api/market/latest/XAUUSD")
    print()

    # Run the server without reload to avoid warnings
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
