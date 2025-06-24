#!/bin/bash

echo "🧭 Zanzibar Daily Analysis - Starting..."

# 1. Check today's trades and print console summary + save markdown report
echo "📝 Running daily performance review..."
python3 core/performance_monitor.py --today

# 2. Generate updated equity curve + winrate + drawdown dashboard
echo "📈 Generating performance dashboard..."
python3 core/equity_curve_plot.py

echo "✅ Zanzibar Analysis Complete!"