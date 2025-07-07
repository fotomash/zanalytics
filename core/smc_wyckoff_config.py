"""
SMC and Wyckoff Configuration File
Defines visualization settings for Smart Money Concepts and Wyckoff indicators
"""

# Configuration dictionary
CONFIG = {
    'SMC_FEATURES': {
        'bullish_order_block': {
            'type': 'marker',
            'color': 'lightgreen',
            'label': 'Bullish OB',
            'symbol': 'square',
            'size': 10,
            'y_col': 'low'
        },
        'bearish_order_block': {
            'type': 'marker',
            'color': 'lightcoral',
            'label': 'Bearish OB',
            'symbol': 'square',
            'size': 10,
            'y_col': 'high'
        },
        'structure_break': {
            'type': 'marker',
            'color': 'yellow',
            'label': 'Structure Break',
            'symbol': 'x',
            'size': 15,
            'y_col': 'close'
        },
        'liquidity_high': {
            'type': 'marker',
            'color': 'orange',
            'label': 'Liquidity High',
            'symbol': 'diamond',
            'size': 8,
            'y_col': 'high'
        },
        'liquidity_low': {
            'type': 'marker',
            'color': 'purple',
            'label': 'Liquidity Low',
            'symbol': 'diamond',
            'size': 8,
            'y_col': 'low'
        },
        'fair_value_gap': {
            'type': 'area',
            'color': 'rgba(255, 255, 0, 0.2)',
            'label': 'FVG',
            'border_color': 'yellow',
            'border_width': 1
        },
        'breaker_block': {
            'type': 'marker',
            'color': 'cyan',
            'label': 'Breaker Block',
            'symbol': 'triangle-up',
            'size': 12,
            'y_col': 'close'
        },
        'mitigation_block': {
            'type': 'marker',
            'color': 'magenta',
            'label': 'Mitigation Block',
            'symbol': 'triangle-down',
            'size': 12,
            'y_col': 'close'
        }
    },

    'WYCKOFF_FEATURES': {
        'wyckoff_spring': {
            'color': 'lime',
            'label': 'Spring',
            'size': 15,
            'symbol': 'star',
            'type': 'marker'
        },
        'wyckoff_upthrust': {
            'color': 'red',
            'label': 'Upthrust',
            'size': 15,
            'symbol': 'star',
            'type': 'marker'
        },
        'wyckoff_accumulation': {
            'color': 'green',
            'label': 'Accumulation',
            'size': 10,
            'alpha': 0.3,
            'type': 'area'
        },
        'wyckoff_distribution': {
            'color': 'red',
            'label': 'Distribution',
            'size': 10,
            'alpha': 0.3,
            'type': 'area'
        },
        'wyckoff_markup': {
            'color': 'blue',
            'label': 'Markup',
            'size': 10,
            'alpha': 0.2,
            'type': 'area'
        },
        'wyckoff_markdown': {
            'color': 'orange',
            'label': 'Markdown',
            'size': 10,
            'alpha': 0.2,
            'type': 'area'
        }
    },

    'VOLUME_PROFILE': {
        'poc_line': {
            'color': 'red',
            'width': 2,
            'dash': 'dash',
            'label': 'POC'
        },
        'vah_line': {
            'color': 'green',
            'width': 1,
            'dash': 'dot',
            'label': 'VAH'
        },
        'val_line': {
            'color': 'green',
            'width': 1,
            'dash': 'dot',
            'label': 'VAL'
        },
        'volume_bars': {
            'color': 'rgba(0, 100, 200, 0.3)',
            'border_color': 'rgba(0, 100, 200, 0.8)',
            'border_width': 1
        }
    },

    'MICROSTRUCTURE': {
        'displacement': {
            'color': 'purple',
            'label': 'Displacement',
            'type': 'line',
            'width': 2,
            'dash': 'solid'
        },
        'inducement': {
            'color': 'orange',
            'label': 'Inducement',
            'type': 'marker',
            'symbol': 'circle',
            'size': 8
        },
        'choch': {
            'color': 'red',
            'label': 'CHoCH',
            'type': 'marker',
            'symbol': 'x',
            'size': 12
        },
        'bos': {
            'color': 'green',
            'label': 'BOS',
            'type': 'marker',
            'symbol': 'cross',
            'size': 12
        }
    },

    'CHART_SETTINGS': {
        'theme': 'plotly_dark',
        'height': 800,
        'width': None,  # Full width
        'margin': {
            'l': 50,
            'r': 50,
            't': 50,
            'b': 50
        },
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1
        }
    }
}

# Function to get feature configuration
def get_smc_feature_config(feature_name):
    """Get configuration for a specific SMC feature"""
    return CONFIG['SMC_FEATURES'].get(feature_name, {})

def get_wyckoff_feature_config(feature_name):
    """Get configuration for a specific Wyckoff feature"""
    return CONFIG['WYCKOFF_FEATURES'].get(feature_name, {})

def get_volume_profile_config():
    """Get volume profile configuration"""
    return CONFIG['VOLUME_PROFILE']

def get_microstructure_config():
    """Get microstructure configuration"""
    return CONFIG['MICROSTRUCTURE']

def get_chart_settings():
    """Get general chart settings"""
    return CONFIG['CHART_SETTINGS']

# Test the configuration when run directly
if __name__ == "__main__":
    print("SMC and Wyckoff Configuration Loaded Successfully!")
    print("Available SMC Features:")
    for feature in CONFIG['SMC_FEATURES']:
        print(f"  - {feature}")

    print("Available Wyckoff Features:")
    for feature in CONFIG['WYCKOFF_FEATURES']:
        print(f"  - {feature}")

    print("Volume Profile Settings:")
    for setting in CONFIG['VOLUME_PROFILE']:
        print(f"  - {setting}")

    print("Microstructure Settings:")
    for setting in CONFIG['MICROSTRUCTURE']:
        print(f"  - {setting}")

    print("Configuration file is ready to use!")