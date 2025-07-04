# create_config.py - Run this to create the config file
import yaml
from pathlib import Path

config = {
    'data_management': {
        'scan_directories': [
            "/Users/tom/Documents/_trade/_exports/_tick",
            "/Users/tom/Documents/_trade/_exports/_tick/data"
        ],
        'supported_formats': [
            "*_ticks.csv",
            "*_tick_data.csv",
            "*/tick_*.csv"
        ],
        'pair_categories': {
            'forex_majors': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
            'forex_crosses': ['EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'GBPAUD', 'AUDNZD'],
            'metals': ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'],
            'crypto': ['BTCUSD', 'ETHUSD', 'BNBUSD', 'XRPUSD', 'ADAUSD', 'DOGEUSD'],
            'indices': ['US500', 'US30', 'NAS100', 'UK100', 'GER40', 'JP225']
        },
        'pair_display': {
            'show_categories': True,
            'show_last_update': True,
            'show_data_quality': True,
            'group_by': "category"
        }
    },
    'visualization': {
        'theme': {
            'primary': "#1e3d59",
            'secondary': "#f5f0e1",
            'accent': "#ffc13b",
            'success': "#00d084",
            'danger': "#ff3860",
            'warning': "#ffdd57",
            'info': "#3298dc"
        },
        'chart': {
            'height': 800,
            'template': "plotly_dark",
            'background': "rgba(17, 17, 17, 0.95)",
            'grid_color': "rgba(128, 128, 128, 0.1)"
        },
        'smc_colors': {
            'bullish_ob': "rgba(0, 255, 0, 0.3)",
            'bullish_ob_border': "#00ff00",
            'bearish_ob': "rgba(255, 0, 0, 0.3)",
            'bearish_ob_border': "#ff0000",
            'fvg_bull': "rgba(0, 255, 255, 0.2)",
            'fvg_bear': "rgba(255, 0, 255, 0.2)",
            'liquidity': "rgba(255, 255, 0, 0.3)",
            'inducement': "rgba(255, 165, 0, 0.4)",
            'swept_level': "rgba(128, 128, 128, 0.3)"
        },
        'wyckoff_colors': {
            'accumulation': "#00d084",
            'markup': "#3298dc",
            'distribution': "#ff3860",
            'markdown': "#ffdd57",
            'spring': "#00ff00",
            'test': "#ffff00"
        },
        'markers': {
            'poi_size': 20,
            'structure_size': 15,
            'entry_size': 18
        }
    },
    'analysis': {
        'poi_detection': {
            'min_touches': 3,
            'tolerance_pips': 5,
            'lookback_bars': 100
        },
        'order_blocks': {
            'min_imbalance': 0.7,
            'mitigation_threshold': 0.5
        },
        'fair_value_gaps': {
            'min_gap_size': 10,
            'max_gap_age': 48
        },
        'inducement': {
            'swing_strength': 3,
            'sweep_threshold': 5
        }
    },
    'price_providers': {
        'finnhub': {
            'enabled': True,
            'symbols_mapping': {
                'XAUUSD': "OANDA:XAU_USD",
                'EURUSD': "OANDA:EUR_USD",
                'GBPUSD': "OANDA:GBP_USD",
                'USDJPY': "OANDA:USD_JPY",
                'BTCUSD': "BINANCE:BTCUSDT",
                'ETHUSD': "BINANCE:ETHUSDT"
            }
        },
        'trading_economics': {
            'enabled': True,
            'symbols_mapping': {
                'XAUUSD': "XAUUSD:CUR",
                'EURUSD': "EURUSD:CUR",
                'US500': "INDU:IND"
            }
        },
        'fallback_order': ['finnhub', 'trading_economics', 'historical_data']
    },
    'real_time_settings': {
        'update_interval': 5,
        'cache_duration': 3,
        'show_bid_ask': True,
        'show_change': True,
        'show_volume': True
    }
}

# Save config
config_path = Path("enhanced_smc_dashboard_config.yaml")
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print(f"✅ Created {config_path}")