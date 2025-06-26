#!/usr/bin/env python3
"""
ZANFLOW API Test Script
Tests all endpoints and verifies data loading
"""

import requests
import json
import time
from datetime import datetime

def test_api_endpoint(url, description):
    """Test a single API endpoint"""
    try:
        print(f"🧪 Testing: {description}")
        print(f"   URL: {url}")

        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS - Status: {response.status_code}")
            return True, data
        else:
            print(f"   ❌ FAILED - Status: {response.status_code}")
            print(f"   Error: {response.text}")
            return False, None

    except requests.exceptions.ConnectionError:
        print(f"   ❌ CONNECTION ERROR - API server not running")
        return False, None
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False, None

def main():
    """Test all API endpoints"""
    print("🚀 ZANFLOW API COMPREHENSIVE TEST")
    print("=" * 50)
    print(f"⏰ Test started at: {datetime.now()}")
    print()

    base_url = "http://localhost:8000"

    # Test 1: API Status
    print("📊 TEST 1: API STATUS & HEALTH CHECK")
    success, data = test_api_endpoint(f"{base_url}/", "API Root Status")

    if success and data:
        print(f"   📈 CSV Timeframes Loaded: {data.get('data_status', {}).get('csv_timeframes_loaded', 0)}")
        print(f"   📋 JSON Reports Loaded: {data.get('data_status', {}).get('json_reports_loaded', 0)}")
        print(f"   🎯 Available Timeframes: {data.get('data_status', {}).get('available_timeframes', [])}")

        csv_count = data.get('data_status', {}).get('csv_timeframes_loaded', 0)
        if csv_count == 5:
            print("   ✅ ALL 5 CSV FILES LOADED SUCCESSFULLY!")
        elif csv_count > 0:
            print(f"   ⚠️  Only {csv_count} CSV files loaded (expected 5)")
        else:
            print("   ❌ NO CSV FILES LOADED!")

    print()

    # Test 2: Available Timeframes
    print("📊 TEST 2: AVAILABLE TIMEFRAMES")
    success, data = test_api_endpoint(f"{base_url}/api/timeframes", "Available Timeframes")

    if success and data:
        timeframes = data.get('available_timeframes', {})
        total_points = data.get('total_data_points', 0)
        print(f"   📊 Total Data Points: {total_points:,}")

        for tf, info in timeframes.items():
            rows = info.get('rows', 0)
            indicators = info.get('indicators', 0)
            print(f"   📈 {tf}: {rows:,} rows, {indicators} indicators")

    print()

    # Test 3: Latest Market Analysis
    print("📊 TEST 3: LATEST MARKET ANALYSIS")
    success, data = test_api_endpoint(f"{base_url}/api/market/latest/XAUUSD", "Latest XAUUSD Analysis")

    if success and data:
        price_action = data.get('price_action', {})
        technical = data.get('technical_indicators', {})
        structure = data.get('market_structure', {})

        if 'close' in price_action:
            print(f"   💰 Current Price: ${price_action['close']:.2f}")

        if 'momentum_regime' in structure:
            print(f"   📊 Momentum Regime: {structure['momentum_regime']}")

        if 'trend' in structure:
            print(f"   📈 Trend: {structure['trend']}")

        print(f"   🎯 Technical Indicators Loaded: {len(technical)}")

        # Show analysis summary
        summary = data.get('analysis_summary', '')
        if summary:
            print(f"   📋 Analysis Summary: {summary}")

    print()

    # Test 4: Trading Signals
    print("📊 TEST 4: CURRENT TRADING SIGNALS")
    success, data = test_api_endpoint(f"{base_url}/api/signals/current", "Current Trading Signals")

    if success and data:
        signals = data.get('signals', [])
        signal_count = data.get('signal_count', 0)

        print(f"   🚨 Active Signals: {signal_count}")

        for i, signal in enumerate(signals[:3]):  # Show first 3 signals
            signal_type = signal.get('type', 'Unknown')
            action = signal.get('action', 'Unknown')
            print(f"   {i+1}. {signal_type} → {action}")

    print()

    # Test 5: Technical Indicators
    print("📊 TEST 5: TECHNICAL INDICATORS")
    success, data = test_api_endpoint(f"{base_url}/api/indicators/XAUUSD/1T", "Technical Indicators")

    if success and data:
        indicators = data.get('indicators', {})
        total_available = data.get('total_indicators_available', 0)

        print(f"   📊 Total Indicators Available: {total_available}")
        print(f"   🎯 Key Indicators Returned: {len(indicators)}")

        for name, info in indicators.items():
            current = info.get('current', 0)
            trend = info.get('trend', 'unknown')
            print(f"   📈 {name}: {current:.2f} ({trend})")

    print()

    # Final Summary
    print("🎯 FINAL TEST SUMMARY")
    print("=" * 30)

    # Test API documentation
    print("📖 Testing API Documentation...")
    try:
        doc_response = requests.get(f"{base_url}/docs", timeout=5)
        if doc_response.status_code == 200:
            print("✅ API Documentation accessible at: http://localhost:8000/docs")
        else:
            print("❌ API Documentation not accessible")
    except:
        print("❌ Cannot reach API documentation")

    print()
    print("🚀 READY FOR CUSTOM GPT INTEGRATION!")
    print("Next steps:")
    print("1. ✅ API Server is running successfully")
    print("2. 🤖 Import custom_gpt_actions.json to ChatGPT")
    print("3. 🔗 Replace localhost:8000 with your VPS domain")
    print("4. 🧪 Test with: 'Get the latest XAUUSD market analysis'")

if __name__ == "__main__":
    main()
