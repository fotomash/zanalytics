#!/usr/bin/env python3
"""
ZANFLOW Multi-Currency Trading Data API Server
LLM-Optimized REST API for Multi-Currency Dataset
Supports: XAUUSD, EURUSD, GBPJPY, GBPUSD, DXY.CASH, US30.CASH
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from pathlib import Path as PathLib
from datetime import datetime
from typing import Optional, List, Dict, Any
import uvicorn
import glob

app = FastAPI(
    title="ZANFLOW Multi-Currency Trading Data API",
    description="LLM-Optimized REST API for 6 currency pairs with 153+ indicators each",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MultiCurrencyTradingAPI:
    """API class for serving multi-currency trading data to LLMs"""

    def __init__(self, data_directory: str = "data"):
        self.data_dir = PathLib(data_directory) if data_directory != "." else PathLib(".")
        self.currency_data = {}  # {pair: {timeframe: dataframe}}
        self.json_reports = {}   # {pair: {report_type: data}}
        self.currency_pairs = ["XAUUSD", "EURUSD", "GBPJPY", "GBPUSD", "DXYCAS", "US30.cash"]
        self.load_all_data()

    def load_all_data(self):
        """Load all currency pairs and their CSV/JSON data"""
        print("🌍 Loading Multi-Currency Trading Data for API...")
        print(f"📁 Base directory: {self.data_dir.absolute()}")

        total_timeframes = 0
        total_reports = 0

        # Map of directory names to clean pair names
        pair_directories = {
            "XAUUSD": "XAUUSD",
            "EURUSD": "EURUSD", 
            "GBPJPY": "GBPJPY",
            "GBPUSD": "GBPUSD",
            "DXYCAS": "DXY.CASH",
            "US30.cash_M1_bars": "US30.CASH"
        }

        for dir_name, clean_pair in pair_directories.items():
            pair_dir = self.data_dir / dir_name

            if not pair_dir.exists():
                print(f"⚠️  Directory not found: {pair_dir}")
                continue

            print(f"\n💰 Loading {clean_pair} data from {dir_name}/")

            # Initialize pair data structure
            if clean_pair not in self.currency_data:
                self.currency_data[clean_pair] = {}
            if clean_pair not in self.json_reports:
                self.json_reports[clean_pair] = {}

            # Load COMPREHENSIVE CSV files
            comprehensive_pattern = str(pair_dir / "*COMPREHENSIVE*.csv")
            csv_files = glob.glob(comprehensive_pattern)

            print(f"   🎯 Found {len(csv_files)} comprehensive CSV files")

            for file_path in csv_files:
                try:
                    file = PathLib(file_path)
                    print(f"   📊 Loading {file.name}...")
                    df = pd.read_csv(file)

                    # Extract timeframe from filename
                    if "COMPREHENSIVE_" in file.name:
                        timeframe = file.name.split("COMPREHENSIVE_")[1].replace(".csv", "")
                    else:
                        timeframe = "unknown"

                    self.currency_data[clean_pair][timeframe] = df
                    total_timeframes += 1
                    print(f"      ✅ {timeframe}: {df.shape[0]} rows, {df.shape[1]} indicators")

                except Exception as e:
                    print(f"   ❌ Error loading {file.name}: {e}")

            # Load JSON reports
            json_pattern = str(pair_dir / "*.json")
            json_files = glob.glob(json_pattern)

            for file_path in json_files:
                try:
                    file = PathLib(file_path)
                    with open(file, 'r') as f:
                        data = json.load(f)
                        report_type = file.stem
                        self.json_reports[clean_pair][report_type] = data
                        total_reports += 1
                    print(f"   📋 Loaded JSON: {file.name}")
                except Exception as e:
                    print(f"   ❌ Error loading JSON {file.name}: {e}")

        print(f"\n🎯 MULTI-CURRENCY DATA LOADED:")
        print(f"   💰 Currency Pairs: {len(self.currency_data)}")
        print(f"   📈 Total Timeframes: {total_timeframes}")
        print(f"   📋 Total Reports: {total_reports}")

        # Show available pairs and timeframes
        for pair, timeframes in self.currency_data.items():
            if timeframes:
                tf_list = list(timeframes.keys())
                print(f"   💰 {pair}: {tf_list}")

    def get_available_pairs(self) -> List[str]:
        """Get list of available currency pairs"""
        return list(self.currency_data.keys())

    def get_latest_analysis(self, pair: str, timeframe: str = "1T") -> Dict[str, Any]:
        """Get latest market analysis for specified pair and timeframe"""

        # Clean up pair name
        pair = pair.upper()
        if pair == "DXY" or pair == "DXY.CASH":
            pair = "DXY.CASH"
        elif pair == "US30" or pair == "US30.CASH":
            pair = "US30.CASH"

        if pair not in self.currency_data:
            available = list(self.currency_data.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Pair {pair} not found. Available: {available}"
            )

        pair_data = self.currency_data[pair]

        # Find the requested timeframe or use best available
        if timeframe not in pair_data:
            available = list(pair_data.keys())
            if not available:
                raise HTTPException(status_code=404, detail=f"No data available for {pair}")
            timeframe = available[0]
            print(f"⚠️  Timeframe {timeframe} not found for {pair}, using {available[0]}")

        df = pair_data[timeframe]
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"No data points available for {pair}")

        latest = df.iloc[-1]

        # Create LLM-optimized response
        response = {
            "pair": pair,
            "timeframe": timeframe,
            "timestamp": str(latest.get('timestamp', datetime.now().isoformat())),
            "data_points": len(df),
            "price_action": {},
            "technical_indicators": {},
            "market_structure": {},
            "analysis_summary": "",
            "available_timeframes": list(pair_data.keys())
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
        response["analysis_summary"] = self._generate_analysis_summary(pair, response)

        return response

    def get_multi_pair_overview(self, pairs: List[str] = None) -> Dict[str, Any]:
        """Get overview of multiple currency pairs"""
        if pairs is None:
            pairs = list(self.currency_data.keys())

        overview = {
            "timestamp": datetime.now().isoformat(),
            "pairs_analyzed": len(pairs),
            "market_overview": {},
            "cross_pair_analysis": {},
            "summary": ""
        }

        pair_summaries = []

        for pair in pairs:
            if pair in self.currency_data:
                try:
                    analysis = self.get_latest_analysis(pair, "1T")
                    price = analysis.get("price_action", {}).get("close", 0)
                    momentum = analysis.get("market_structure", {}).get("momentum_regime", "UNKNOWN")
                    trend = analysis.get("market_structure", {}).get("trend", "UNKNOWN")

                    overview["market_overview"][pair] = {
                        "price": price,
                        "momentum": momentum,
                        "trend": trend,
                        "data_quality": "HIGH" if analysis["data_points"] > 1000 else "MEDIUM"
                    }

                    pair_summaries.append(f"{pair}: {momentum} momentum, {trend} trend")

                except Exception as e:
                    overview["market_overview"][pair] = {"error": str(e)}

        # Generate cross-pair analysis
        overview["cross_pair_analysis"] = self._analyze_cross_pairs(overview["market_overview"])

        # Generate summary
        overview["summary"] = " | ".join(pair_summaries) if pair_summaries else "No data available"

        return overview

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

    def _analyze_cross_pairs(self, market_overview: Dict) -> Dict[str, Any]:
        """Analyze relationships between currency pairs"""
        analysis = {}

        # Count momentum regimes
        momentum_counts = {}
        trend_counts = {}

        for pair, data in market_overview.items():
            if "error" not in data:
                momentum = data.get("momentum", "UNKNOWN")
                trend = data.get("trend", "UNKNOWN")

                momentum_counts[momentum] = momentum_counts.get(momentum, 0) + 1
                trend_counts[trend] = trend_counts.get(trend, 0) + 1

        analysis["momentum_distribution"] = momentum_counts
        analysis["trend_distribution"] = trend_counts

        # Market sentiment
        bullish_count = trend_counts.get("BULLISH", 0)
        bearish_count = trend_counts.get("BEARISH", 0)
        total_pairs = bullish_count + bearish_count

        if total_pairs > 0:
            if bullish_count > bearish_count * 1.5:
                analysis["market_sentiment"] = "RISK_ON"
            elif bearish_count > bullish_count * 1.5:
                analysis["market_sentiment"] = "RISK_OFF"
            else:
                analysis["market_sentiment"] = "MIXED"
        else:
            analysis["market_sentiment"] = "UNCERTAIN"

        return analysis

    def _generate_analysis_summary(self, pair: str, response: Dict[str, Any]) -> str:
        """Generate human-readable summary for LLM"""
        summary_parts = []

        # Price summary
        if "close" in response["price_action"]:
            price = response["price_action"]["close"]
            if pair in ["XAUUSD"]:
                summary_parts.append(f"{pair}: ${price:.2f}")
            elif "USD" in pair:
                summary_parts.append(f"{pair}: {price:.5f}")
            else:
                summary_parts.append(f"{pair}: {price:.2f}")

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
            summary_parts.append(f"Trend: {trend} ({strength:+.2f}%)")

        return " | ".join(summary_parts) if summary_parts else f"{pair} analysis available"

# Initialize API data handler
print("🚀 Initializing ZANFLOW Multi-Currency Trading Data API...")
api_data = MultiCurrencyTradingAPI(data_directory=".")

# API Endpoints
@app.get("/")
async def root():
    """API information and health check"""
    return {
        "name": "ZANFLOW Multi-Currency Trading Data API",
        "version": "2.0.0",
        "description": "LLM-Optimized REST API for 6 currency pairs with 153+ indicators each",
        "status": "active",
        "data_status": {
            "currency_pairs": len(api_data.currency_data),
            "available_pairs": list(api_data.currency_data.keys()),
            "total_timeframes": sum(len(tf) for tf in api_data.currency_data.values()),
            "total_reports": sum(len(reports) for reports in api_data.json_reports.values())
        },
        "supported_pairs": ["XAUUSD", "EURUSD", "GBPJPY", "GBPUSD", "DXY.CASH", "US30.CASH"],
        "endpoints": [
            "/api/pairs",
            "/api/market/{pair}/latest",
            "/api/market/overview",
            "/api/indicators/{pair}/{timeframe}",
            "/docs"
        ]
    }

@app.get("/api/pairs")
async def get_available_pairs():
    """Get all available currency pairs and their data info"""
    pairs_info = {}

    for pair, timeframes in api_data.currency_data.items():
        pairs_info[pair] = {
            "available_timeframes": list(timeframes.keys()),
            "timeframe_count": len(timeframes),
            "total_data_points": sum(len(df) for df in timeframes.values()),
            "max_indicators": max(len(df.columns) for df in timeframes.values()) if timeframes else 0
        }

    return {
        "available_pairs": pairs_info,
        "total_pairs": len(pairs_info),
        "supported_pairs": ["XAUUSD", "EURUSD", "GBPJPY", "GBPUSD", "DXY.CASH", "US30.CASH"]
    }

@app.get("/api/market/{pair}/latest")
async def get_market_latest(
    pair: str = Path(..., description="Currency pair (XAUUSD, EURUSD, GBPJPY, GBPUSD, DXY, US30)"),
    timeframe: str = Query(default="1T", description="Timeframe (1T, 5T, 15T, 30T)")
):
    """Get latest market analysis for specified pair and timeframe"""
    try:
        return api_data.get_latest_analysis(pair, timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/market/overview")
async def get_market_overview(
    pairs: str = Query(default="", description="Comma-separated pairs (empty for all)")
):
    """Get multi-pair market overview"""
    try:
        if pairs:
            pair_list = [p.strip().upper() for p in pairs.split(",")]
        else:
            pair_list = None

        return api_data.get_multi_pair_overview(pair_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/indicators/{pair}/{timeframe}")
async def get_indicators(
    pair: str = Path(..., description="Currency pair"),
    timeframe: str = Path(..., description="Timeframe"),
    limit: int = Query(default=50, description="Number of recent data points")
):
    """Get recent indicator data for analysis"""
    try:
        pair = pair.upper()
        if pair == "DXY":
            pair = "DXY.CASH"
        elif pair == "US30":
            pair = "US30.CASH"

        if pair not in api_data.currency_data:
            available = list(api_data.currency_data.keys())
            raise HTTPException(status_code=404, detail=f"Pair {pair} not found. Available: {available}")

        pair_data = api_data.currency_data[pair]
        if timeframe not in pair_data:
            available = list(pair_data.keys())
            if not available:
                raise HTTPException(status_code=404, detail=f"No data available for {pair}")
            timeframe = available[0]

        df = pair_data[timeframe]
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
    print("\n🚀 Starting ZANFLOW Multi-Currency Trading Data API Server...")
    print("💰 Supporting: XAUUSD, EURUSD, GBPJPY, GBPUSD, DXY.CASH, US30.CASH")
    print("📊 153+ Technical Indicators per Pair")
    print("🔗 API Documentation: http://localhost:8000/docs")
    print("🌐 API Status: http://localhost:8000/")
    print("📈 Market Overview: http://localhost:8000/api/market/overview")
    print("💰 Available Pairs: http://localhost:8000/api/pairs")
    print()

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
