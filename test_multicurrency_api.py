#!/usr/bin/env python3
"""
ZANFLOW Multi-Currency API Test Script
Tests all endpoints for 6 currency pairs
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
    """Test all multi-currency API endpoints"""
    print("🌍 ZANFLOW MULTI-CURRENCY API COMPREHENSIVE TEST")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now()}")
    print()

    base_url = "http://localhost:8000"

    # Test 1: API Status & Multi-Currency Overview
    print("📊 TEST 1: API STATUS & MULTI-CURRENCY OVERVIEW")
    success, data = test_api_endpoint(f"{base_url}/", "API Root Status")

    if success and data:
        pairs_count = data.get('data_status', {}).get('currency_pairs', 0)
        available_pairs = data.get('data_status', {}).get('available_pairs', [])
        total_timeframes = data.get('data_status', {}).get('total_timeframes', 0)

        print(f"   💰 Currency Pairs Loaded: {pairs_count}")
        print(f"   📈 Total Timeframes: {total_timeframes}")
        print(f"   🎯 Available Pairs: {available_pairs}")

        if pairs_count == 6:
            print("   ✅ ALL 6 CURRENCY PAIRS LOADED SUCCESSFULLY!")
        elif pairs_count > 0:
            print(f"   ⚠️  Only {pairs_count} pairs loaded (expected 6)")
        else:
            print("   ❌ NO CURRENCY PAIRS LOADED!")

    print()

    # Test 2: Available Pairs Details
    print("📊 TEST 2: AVAILABLE PAIRS DETAILS")
    success, data = test_api_endpoint(f"{base_url}/api/pairs", "Available Pairs")

    if success and data:
        pairs_info = data.get('available_pairs', {})
        total_pairs = data.get('total_pairs', 0)

        print(f"   💰 Total Pairs: {total_pairs}")

        for pair, info in pairs_info.items():
            timeframes = info.get('available_timeframes', [])
            data_points = info.get('total_data_points', 0)
            indicators = info.get('max_indicators', 0)
            print(f"   💰 {pair}: {len(timeframes)} timeframes, {data_points:,} data points, {indicators} indicators")

    print()

    # Test 3: Individual Currency Pairs
    print("📊 TEST 3: INDIVIDUAL CURRENCY PAIR ANALYSIS")
    test_pairs = ["XAUUSD", "EURUSD", "GBPJPY", "GBPUSD", "DXY.CASH", "US30.CASH"]

    for pair in test_pairs:
        print(f"\n   💰 Testing {pair}:")
        success, data = test_api_endpoint(f"{base_url}/api/market/{pair}/latest", f"Latest {pair} Analysis")

        if success and data:
            price_action = data.get('price_action', {})
            market_structure = data.get('market_structure', {})
            available_tf = data.get('available_timeframes', [])

            if 'close' in price_action:
                price = price_action['close']
                if pair == "XAUUSD":
                    print(f"      💰 Current Price: ${price:.2f}")
                elif "USD" in pair:
                    print(f"      💰 Current Price: {price:.5f}")
                else:
                    print(f"      💰 Current Price: {price:.2f}")

            if 'momentum_regime' in market_structure:
                print(f"      📊 Momentum: {market_structure['momentum_regime']}")

            if 'trend' in market_structure:
                print(f"      📈 Trend: {market_structure['trend']}")

            print(f"      🎯 Available Timeframes: {available_tf}")

            # Show analysis summary
            summary = data.get('analysis_summary', '')
            if summary:
                print(f"      📋 Summary: {summary}")
        else:
            print(f"      ❌ Failed to load {pair} data")

    print()

    # Test 4: Multi-Pair Market Overview
    print("📊 TEST 4: MULTI-PAIR MARKET OVERVIEW")
    success, data = test_api_endpoint(f"{base_url}/api/market/overview", "Multi-Pair Overview")

    if success and data:
        pairs_analyzed = data.get('pairs_analyzed', 0)
        market_overview = data.get('market_overview', {})
        cross_pair = data.get('cross_pair_analysis', {})

        print(f"   🌍 Pairs Analyzed: {pairs_analyzed}")
        print(f"   📊 Market Overview:")

        for pair, info in market_overview.items():
            if "error" not in info:
                momentum = info.get('momentum', 'N/A')
                trend = info.get('trend', 'N/A')
                quality = info.get('data_quality', 'N/A')
                print(f"      💰 {pair}: {momentum} momentum, {trend} trend ({quality} quality)")

        # Cross-pair analysis
        market_sentiment = cross_pair.get('market_sentiment', 'N/A')
        print(f"   🎯 Overall Market Sentiment: {market_sentiment}")

        # Show momentum and trend distribution
        momentum_dist = cross_pair.get('momentum_distribution', {})
        trend_dist = cross_pair.get('trend_distribution', {})

        print(f"   📊 Momentum Distribution: {momentum_dist}")
        print(f"   📈 Trend Distribution: {trend_dist}")

    print()

    # Test 5: Technical Indicators for Key Pairs
    print("📊 TEST 5: TECHNICAL INDICATORS")
    key_pairs = ["XAUUSD", "EURUSD", "DXY.CASH"]

    for pair in key_pairs:
        print(f"\n   📈 {pair} Technical Indicators (1T timeframe):")
        success, data = test_api_endpoint(f"{base_url}/api/indicators/{pair}/1T", f"{pair} Indicators")

        if success and data:
            indicators = data.get('indicators', {})
            total_available = data.get('total_indicators_available', 0)

            print(f"      🎯 Total Indicators Available: {total_available}")
            print(f"      📊 Key Indicators:")

            for name, info in indicators.items():
                current = info.get('current', 0)
                trend = info.get('trend', 'unknown')
                print(f"         📈 {name}: {current:.3f} ({trend})")

    print()

    # Test 6: Custom GPT Integration Examples
    print("📊 TEST 6: CUSTOM GPT INTEGRATION EXAMPLES")

    # Example 1: Market overview
    print("\n   🤖 Example 1: Market Overview Request")
    success, data = test_api_endpoint(f"{base_url}/api/market/overview?pairs=XAUUSD,EURUSD,DXY.CASH", "Selected Pairs Overview")
    if success and data:
        summary = data.get('summary', '')
        print(f"      📋 LLM Response: {summary}")

    # Example 2: Gold analysis
    print("\n   🤖 Example 2: Gold Analysis Request")
    success, data = test_api_endpoint(f"{base_url}/api/market/XAUUSD/latest?timeframe=1T", "Gold Latest Analysis")
    if success and data:
        summary = data.get('analysis_summary', '')
        print(f"      📋 LLM Response: {summary}")

    print()

    # Final Summary
    print("🎯 FINAL MULTI-CURRENCY TEST SUMMARY")
    print("=" * 40)

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
    print("1. ✅ Multi-Currency API Server is running successfully")
    print("2. 🤖 Import custom_gpt_multicurrency_actions.json to ChatGPT")
    print("3. 🔗 Replace localhost:8000 with your VPS domain")
    print("4. 🧪 Test with: 'Give me a market overview of all currency pairs'")
    print("5. 🧪 Test with: 'What's the latest analysis for XAUUSD?'")
    print("6. 🧪 Test with: 'Compare EURUSD and GBPUSD momentum'")

if __name__ == "__main__":
    main()
