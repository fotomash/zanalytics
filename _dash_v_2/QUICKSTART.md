# ZANFLOW Integrated System - Quick Start Guide

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Add your API keys to `.streamlit/secrets.toml`:
```toml
openai_API = "your-openai-key"
finnhub_api_key = "your-finnhub-key"
telegram_bot_token = "your-telegram-bot-token"  # Optional
telegram_chat_id = "your-telegram-chat-id"      # Optional
```

### 3. Run the Integrated System
```bash
streamlit run zanflow_integrated_system.py
```

## 🎯 Features Implemented

### ✅ LLM Integration (OpenAI GPT-4)
- Real-time pattern analysis
- Multi-strategy confluence scoring
- Actionable trade recommendations
- Risk-adjusted entry/exit levels

### ✅ Telegram Alerts
- Automatic alerts for high-confluence setups
- Formatted trade signals with emojis
- Entry, stop, and target levels
- Market context and invalidation scenarios

### ✅ Market Intelligence
- Finnhub sentiment analysis
- News sentiment integration
- Technical bias calculation
- Inter-market correlation

### ✅ Backtesting Engine
- Strategy performance validation
- Win rate and profit factor
- Sharpe ratio calculation
- Maximum drawdown analysis

### ✅ Custom GPT Router
- Specialized analysis by strategy type
- SMC specialist for order flow
- Wyckoff specialist for phases
- Risk management specialist
- Consensus analysis from multiple GPTs

## 📊 Dashboard Sections

### 1. Live Analysis Tab
- Load your tick data
- Run comprehensive analysis
- View AI-powered insights
- Send Telegram alerts

### 2. Market Intelligence Tab
- Real-time sentiment analysis
- News and social media monitoring
- Technical bias indicators
- Market correlation data

### 3. Backtesting Tab
- Test strategy combinations
- View historical performance
- Optimize parameters
- Risk metrics analysis

### 4. System Monitor Tab
- API usage tracking
- System health metrics
- Cache management
- Performance monitoring

## 🔧 Customization

### Adding New Strategies
1. Add your strategy module to the MODULE_REGISTRY
2. Implement the analysis function
3. The system will automatically integrate it

### Custom GPT Specialists
1. Add new specialist configs to `custom_gpt_router.py`
2. Define specialized system prompts
3. Router will include in consensus analysis

### Alert Customization
1. Modify `format_trade_alert()` in TelegramAlertSystem
2. Add custom fields and formatting
3. Adjust alert triggers based on your criteria

## 🎮 Usage Tips

1. **High Confluence Trades**: Look for 70%+ confluence scores
2. **Multi-Timeframe**: Run analysis on different timeframes
3. **Risk Management**: Always use suggested stop levels
4. **Backtesting**: Validate strategies before live trading
5. **Alerts**: Set minimum confluence for Telegram alerts

## 🐛 Troubleshooting

### OpenAI API Issues
- Check API key in secrets.toml
- Verify API quota and billing
- Test with smaller data samples

### Data Loading Issues
- Verify data paths in secrets.toml
- Check CSV format compatibility
- Ensure timestamp column exists

### Telegram Alerts Not Sending
- Verify bot token and chat ID
- Check internet connectivity
- Test with manual message first

## 📈 Next Steps

1. **Production Deployment**: Deploy on cloud with proper secret management
2. **Database Integration**: Store analysis results for historical tracking
3. **Real-time Data**: Connect to live market data feeds
4. **Advanced ML**: Integrate your trained models for pattern recognition
5. **Portfolio Management**: Add multi-asset portfolio tracking

Happy Trading! 🚀
