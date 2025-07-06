# ZANFLOW Interactive Command Center Documentation

## 🎯 Overview

The Interactive Command Center transforms your Zanalytics dashboard from a read-only monitoring tool into a true command and control interface for your trading strategies. This powerful addition allows you to:

- **Edit Strategies in Real-Time**: Modify strategy configurations directly from the UI
- **Validate Before Deploy**: Ensure configurations are valid before they go live
- **Automatic Backups**: Every change is backed up automatically
- **Visual & Code Editing**: Choose between intuitive visual forms or direct YAML editing
- **Live Reload**: Changes are automatically queued for reload in the live system

## 🏗️ Architecture

The Interactive Command Center follows a secure, three-tier architecture:

```
┌─────────────────────┐
│   Dashboard UI      │  ← Streamlit Interface
│  (Strategy Editor)  │
└──────────┬──────────┘
           │ API Calls
           ▼
┌─────────────────────┐
│    API Service      │  ← Flask REST API
│ (Validation Layer)  │
└──────────┬──────────┘
           │ File I/O
           ▼
┌─────────────────────┐
│  Strategy Configs   │  ← YAML Files
│   & Backups         │
└─────────────────────┘
```

### Security Principles

1. **No Direct File Access**: The dashboard never writes files directly
2. **API Validation**: All changes go through validation before saving
3. **Automatic Backups**: Every change creates a timestamped backup
4. **User Authentication**: Integrates with your existing auth system

## 📦 Components

### 1. Enhanced API Service (`strategy_management_api.py`)

New endpoints added to your existing API:

- `GET /strategies` - List all available strategies
- `GET /strategies/<id>` - Get full configuration for a strategy
- `POST /strategies/<id>` - Update a strategy configuration
- `POST /strategies/<id>/validate` - Validate without saving
- `GET /strategies/<id>/backups` - List available backups
- `POST /strategies/<id>/restore/<backup>` - Restore from backup
- `POST /strategies/create` - Create new strategy from template

### 2. Strategy Editor Dashboard (`🔧_Strategy_Editor.py`)

Features:
- **Strategy Selector**: Browse and select strategies from sidebar
- **Dual Edit Modes**: Visual forms or direct YAML editing
- **Real-time Validation**: Instant feedback on configuration errors
- **Backup Management**: View and restore from previous versions
- **Template System**: Quick-start with pre-configured templates

### 3. Validation Module (`strategy_validator.py`)

Comprehensive validation including:
- Required field checking
- Data type validation
- Business rule enforcement
- Risk parameter limits
- Warning generation for risky configurations

### 4. Component Manager (`component_manager.py`)

Utilities for:
- Template management
- Configuration merging
- Indicator extraction
- Summary generation
- Import/Export functionality

## 🚀 Usage Guide

### Basic Workflow

1. **Select a Strategy**
   ```
   - Open the Strategy Editor page
   - Choose a strategy from the sidebar dropdown
   ```

2. **Edit Configuration**
   - **Visual Mode**: Use forms and inputs for structured editing
   - **YAML Mode**: Direct code editing for advanced users

3. **Validate Changes**
   ```
   - Click "✅ Validate" to check configuration
   - Fix any errors shown in red
   - Address warnings if needed
   ```

4. **Save Changes**
   ```
   - Click "💾 Save Changes" to persist
   - Automatic backup is created
   - Reload command is queued for live system
   ```

### Visual Editor Tabs

#### 🎯 Basic Info
- Strategy name and description
- Status (active/inactive/testing)
- Timeframe selection

#### 📊 Entry Conditions
- Primary entry signals
- Confirmation indicators
- Custom conditions

#### 🚪 Exit Conditions
- Take profit settings (fixed/ATR/percentage)
- Stop loss configuration
- Trailing stop options

#### 💰 Risk Management
- Position sizing
- Maximum concurrent positions
- Daily loss limits
- Maximum drawdown

#### ⚙️ Parameters
- Indicator settings
- Custom parameters
- Optimization ranges

#### 🔧 Advanced
- Additional YAML configuration
- Custom fields
- Extended settings

### Creating New Strategies

1. Click "➕ Create New Strategy" in sidebar
2. Choose a template:
   - **Default**: Basic structure
   - **Scalping**: Fast, small profits
   - **Swing**: Medium-term trades
   - **Breakout**: Momentum-based
3. Customize the configuration
4. Save when ready

### Managing Backups

1. Click "📋 View Backups" when editing
2. See all previous versions with timestamps
3. Select a backup to preview
4. Click "🔄 Restore" to revert changes

## 🔧 Configuration Examples

### Example 1: Simple Trend Following
```yaml
strategy_name: Simple Trend Follower
description: Follows major market trends using moving averages
status: active
timeframes:
  - H4
  - D1
entry_conditions:
  primary:
    - ma_crossover
    - trend_strength
  confirmations:
    - volume_above_average
exit_conditions:
  take_profit:
    type: trailing
    value: 2.0
  stop_loss:
    type: atr
    value: 2.0
risk_management:
  position_size: 0.02
  max_positions: 2
  max_daily_loss: 3.0
parameters:
  fast_ma: 20
  slow_ma: 50
  atr_period: 14
```

### Example 2: London Session Scalper
```yaml
strategy_name: London Scalper Pro
description: Scalps quick moves during London session
status: testing
timeframes:
  - M5
  - M15
entry_conditions:
  primary:
    - london_session_active
    - momentum_spike
  confirmations:
    - spread_acceptable
    - volatility_increasing
exit_conditions:
  take_profit:
    type: fixed
    value: 15
  stop_loss:
    type: fixed
    value: 10
risk_management:
  position_size: 0.01
  max_positions: 3
  max_daily_loss: 2.0
parameters:
  london_start: 8
  london_end: 10
  min_momentum: 0.7
  max_spread: 2
```

## 🛠️ API Integration

### Reading Strategy Configuration
```python
import requests

# Get list of strategies
response = requests.get("http://localhost:5010/strategies")
strategies = response.json()

# Get specific strategy
strategy_id = "london_scalper.yml"
response = requests.get(f"http://localhost:5010/strategies/{strategy_id}")
config = response.json()
```

### Updating Strategy
```python
# Modify configuration
config['parameters']['fast_ma'] = 25

# Validate first
response = requests.post(
    f"http://localhost:5010/strategies/{strategy_id}/validate",
    json=config
)
validation = response.json()

if validation['valid']:
    # Save changes
    response = requests.post(
        f"http://localhost:5010/strategies/{strategy_id}",
        json=config
    )
    print(response.json()['message'])
```

### Creating New Strategy
```python
# Create from template
response = requests.post(
    "http://localhost:5010/strategies/create",
    json={
        'name': 'My New Strategy',
        'template': 'scalping'
    }
)
result = response.json()
print(f"Created: {result['strategy_id']}")
```

## 📊 Validation Rules

### Required Fields
- `strategy_name`: Non-empty, 3-50 characters
- `timeframes`: At least one valid timeframe
- `entry_conditions`: Must have primary conditions
- `exit_conditions`: Must have take_profit and stop_loss
- `risk_management`: Must have position_size and max_positions

### Valid Values
- **Timeframes**: M1, M5, M15, M30, H1, H4, D1, W1
- **Status**: active, inactive, testing, deprecated
- **Exit Types**: fixed, atr, percentage, trailing
- **Position Size**: 0.01 to 1.0 (1% to 100%)
- **Max Positions**: 1 to 100

### Warnings Generated For
- Missing description
- Position size > 10%
- More than 5 concurrent positions
- Risk/reward ratio < 1
- Testing status with large position size

## 🚨 Troubleshooting

### Common Issues

1. **"Could not connect to API"**
   - Ensure API service is running: `python zanalytics_api_service.py`
   - Check port 5010 is not blocked
   - Verify CORS is enabled

2. **"Strategy not found"**
   - Check file exists in `knowledge/strategies/`
   - Ensure `.yml` or `.yaml` extension
   - Verify file permissions

3. **"Invalid YAML format"**
   - Check for proper indentation (2 spaces)
   - Ensure no tabs used
   - Validate special characters are quoted

4. **"Validation failed"**
   - Review error messages carefully
   - Check all required fields present
   - Verify data types match expected

### Debug Mode

Enable debug logging in API:
```python
# In zanalytics_api_service.py
app.run(host='0.0.0.0', port=5010, debug=True)

# Set logging level
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Best Practices

1. **Always Validate Before Saving**
   - Use the Validate button to catch errors early
   - Address all errors before warnings

2. **Use Descriptive Names**
   - Strategy names should indicate their purpose
   - Parameter names should be self-documenting

3. **Start with Templates**
   - Modify existing templates rather than starting blank
   - Templates include proven risk management settings

4. **Regular Backups**
   - System auto-backs up, but export important strategies
   - Use version control for strategy files

5. **Test in Stages**
   - Start with status: testing
   - Use small position sizes initially
   - Graduate to active after validation

## 🔄 Live Reload System

When you save changes:

1. Configuration is validated and saved
2. Backup is created automatically
3. Reload command is queued: `reload_<strategy>_<timestamp>.json`
4. Action Dispatcher picks up command
5. Strategy agent reloads with new config
6. No downtime or manual intervention needed

## 📈 Future Enhancements

Planned features:
- Strategy performance metrics in editor
- A/B testing interface
- Collaborative editing with locks
- Strategy version control with Git
- Machine learning parameter optimization
- Backtesting from editor interface

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review validation error messages
3. Consult API logs for details
4. Contact the Zanalytics team

---

*Interactive Command Center v2.0 - Empowering traders with real-time strategy control*
