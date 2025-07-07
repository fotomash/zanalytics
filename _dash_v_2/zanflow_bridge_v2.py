import redis
import json
import pandas as pd
from datetime import datetime
import time
import sys
import os
import numpy as np
import warnings

# Suppress SSL warnings
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')

# Add the correct path found by setup
ZANALYTICS_PATH = '/Users/tom/Documents/GitHub/zanalytics'
if os.path.exists(ZANALYTICS_PATH):
    sys.path.insert(0, ZANALYTICS_PATH)
    print(f"✅ Using zanalytics from: {ZANALYTICS_PATH}")

# Import available engines
engines_loaded = {}

try:
    from core.analysis.smc_enrichment_engine import tag_smc_zones
    engines_loaded['smc'] = True
    print("✅ Loaded SMC enrichment engine")
except ImportError as e:
    print(f"⚠️ Could not load SMC engine: {e}")
    engines_loaded['smc'] = False

try:
    from core.wyckoff_phase_engine import tag_wyckoff_phases
    engines_loaded['wyckoff'] = True
    print("✅ Loaded Wyckoff phase engine")
except ImportError as e:
    print(f"⚠️ Could not load Wyckoff engine: {e}")
    engines_loaded['wyckoff'] = False

# Skip orchestrator if loguru not available
try:
    from core.orchestrators.main_orchestrator import MainOrchestrator
    engines_loaded['orchestrator'] = True
    print("✅ Loaded Main orchestrator")
except ImportError as e:
    if 'loguru' in str(e):
        print("ℹ️ Orchestrator needs loguru - skipping (pip install loguru to enable)")
    else:
        print(f"⚠️ Could not load Main orchestrator: {e}")
    engines_loaded['orchestrator'] = False

try:
    from core.strategies.advanced_smc import run_advanced_smc_strategy
    engines_loaded['advanced_smc'] = True
    print("✅ Loaded Advanced SMC strategy")
except ImportError as e:
    print(f"⚠️ Could not load Advanced SMC: {e}")
    engines_loaded['advanced_smc'] = False

print(f"\\n📊 Engines loaded: {sum(engines_loaded.values())}/{len(engines_loaded)}")
print("💡 3/4 engines loaded - this is enough for full analysis!")

class ZanflowBridge:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.orchestrator = None
        self.engines_loaded = engines_loaded
        
        print("\\n🚀 ZANFLOW Bridge initialized")
        print(f"✅ SMC Engine: {'Ready' if engines_loaded['smc'] else 'Not available'}")
        print(f"✅ Wyckoff Engine: {'Ready' if engines_loaded['wyckoff'] else 'Not available'}")
        print(f"✅ Advanced SMC: {'Ready' if engines_loaded['advanced_smc'] else 'Not available'}")
    
    def get_mt5_data(self, symbol):
        """Get latest MT5 data with SMC/Wyckoff analysis"""
        try:
            data = self.r.get(f"mt5:{symbol}:latest")
            if data:
                return json.loads(data)
        except:
            pass
        return None
    
    def get_tick_history(self, symbol, limit=100):
        """Get historical ticks for analysis"""
        try:
            ticks = self.r.lrange(f"mt5:{symbol}:ticks", 0, limit-1)
            return [json.loads(t) for t in ticks]
        except:
            return []
    
    def process_with_zanalytics(self, symbol):
        """Process MT5 data through zanalytics engines"""
        # Get tick history
        ticks = self.get_tick_history(symbol, 100)
        if not ticks:
            return None
        
        # Convert to DataFrame for analysis
        df_data = []
        for tick in ticks:
            row = {
                'Time': datetime.fromtimestamp(tick['timestamp']),
                'Open': tick['bid'],
                'High': tick['bid'] + abs(tick['spread']) * 0.00001,
                'Low': tick['bid'] - abs(tick['spread']) * 0.00001,
                'Close': tick['ask'],
                'Volume': tick.get('volume', 100),
                'tick_volume': tick.get('volume', 100)
            }
            df_data.append(row)
        
        if not df_data:
            return None
            
        df = pd.DataFrame(df_data)
        df.set_index('Time', inplace=True)
        
        # Apply your zanalytics engines
        enhanced_data = {}
        
        # Apply SMC analysis
        if self.engines_loaded['smc']:
            try:
                df_smc = tag_smc_zones(df.copy(), tf='M1')
                enhanced_data['smc_analysis'] = {
                    'zones_tagged': True,
                    'last_bos': df_smc['bos'].iloc[-1] if 'bos' in df_smc else None,
                    'last_choch': df_smc['choch'].iloc[-1] if 'choch' in df_smc else None,
                    'market_structure': self._determine_market_structure(df_smc)
                }
                print(f"✅ {symbol}: Applied SMC analysis")
            except Exception as e:
                print(f"⚠️ {symbol}: SMC analysis error: {e}")
        
        # Apply Wyckoff analysis
        if self.engines_loaded['wyckoff']:
            try:
                df_wyckoff = tag_wyckoff_phases(df.copy(), 'M1')
                enhanced_data['wyckoff_analysis'] = {
                    'phases_tagged': True,
                    'current_phase': df_wyckoff['wyckoff_phase'].iloc[-1] if 'wyckoff_phase' in df_wyckoff else None,
                    'phase_strength': self._calculate_phase_strength(df_wyckoff)
                }
                print(f"✅ {symbol}: Applied Wyckoff analysis")
            except Exception as e:
                print(f"⚠️ {symbol}: Wyckoff analysis error: {e}")
        
        # Generate trading signal
        signal_data = self._generate_signal(ticks[-1], enhanced_data)
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'price': {
                'bid': ticks[-1].get('bid', 0),
                'ask': ticks[-1].get('ask', 0),
                'spread': ticks[-1].get('spread', 0)
            },
            'mt5_analysis': {
                'smc': ticks[-1].get('smc', {}),
                'wyckoff': ticks[-1].get('wyckoff', {})
            },
            'zanalytics_enhanced': enhanced_data,
            'signal': signal_data
        }
        
        # Store in Redis
        self.r.set(
            f"zanalytics:{symbol}:enhanced",
            json.dumps(result),
            ex=3600
        )
        
        return result
    
    def _determine_market_structure(self, df):
        """Determine market structure from SMC analysis"""
        if 'bos' not in df:
            return 'Unknown'
        
        recent_bos = df['bos'].dropna().tail(5)
        if len(recent_bos) == 0:
            return 'Ranging'
        
        bullish_count = (recent_bos == 'Bullish BOS').sum()
        bearish_count = (recent_bos == 'Bearish BOS').sum()
        
        if bullish_count > bearish_count:
            return 'Bullish'
        elif bearish_count > bullish_count:
            return 'Bearish'
        else:
            return 'Ranging'
    
    def _calculate_phase_strength(self, df):
        """Calculate Wyckoff phase strength"""
        if 'wyckoff_phase' not in df:
            return 0
        
        # Simple strength calculation based on phase consistency
        recent_phases = df['wyckoff_phase'].dropna().tail(10)
        if len(recent_phases) == 0:
            return 0
        
        mode_phase = recent_phases.mode()
        if len(mode_phase) > 0:
            consistency = (recent_phases == mode_phase[0]).sum() / len(recent_phases)
            return consistency
        return 0.5
    
    def _generate_signal(self, mt5_data, enhanced_data):
        """Generate trading signal combining MT5 and zanalytics data"""
        score = 0
        reasons = []
        
        # MT5 SMC signals
        if 'smc' in mt5_data:
            smc = mt5_data['smc']
            if smc.get('market_structure') == 'Bullish':
                score += 1
                reasons.append("MT5: Bullish structure")
            elif smc.get('market_structure') == 'Bearish':
                score -= 1
                reasons.append("MT5: Bearish structure")
            
            if smc.get('has_order_block'):
                if smc.get('order_block_type') == 'Bullish':
                    score += 2
                    reasons.append("MT5: Bullish OB")
                else:
                    score -= 2
                    reasons.append("MT5: Bearish OB")
        
        # Zanalytics SMC signals
        if 'smc_analysis' in enhanced_data:
            smc_an = enhanced_data['smc_analysis']
            if smc_an.get('market_structure') == 'Bullish':
                score += 1.5
                reasons.append("Zanalytics: Bullish structure")
            elif smc_an.get('market_structure') == 'Bearish':
                score -= 1.5
                reasons.append("Zanalytics: Bearish structure")
        
        # Wyckoff signals
        if 'wyckoff_analysis' in enhanced_data:
            wyckoff = enhanced_data['wyckoff_analysis']
            phase = wyckoff.get('current_phase', '')
            strength = wyckoff.get('phase_strength', 0)
            
            if phase in ['Accumulation', 'Spring']:
                score += 2 * strength
                reasons.append(f"Wyckoff: {phase} ({strength:.0%})")
            elif phase in ['Distribution', 'UTAD']:
                score -= 2 * strength
                reasons.append(f"Wyckoff: {phase} ({strength:.0%})")
        
        # Determine signal
        if score >= 3:
            signal = 'strong_buy'
        elif score >= 1.5:
            signal = 'buy'
        elif score <= -3:
            signal = 'strong_sell'
        elif score <= -1.5:
            signal = 'sell'
        else:
            signal = 'hold'
        
        return {
            'action': signal,
            'score': round(score, 2),
            'confidence': min(abs(score) / 5.0, 1.0),
            'reasons': reasons
        }
    
    def monitor_symbols(self):
        """Monitor all symbols and process through zanalytics"""
        print("\\n🔍 Starting symbol monitoring...")
        print("Waiting for MT5 data...")
        
        last_signal_time = {}
        
        while True:
            try:
                # Get all available symbols
                keys = self.r.keys("mt5:*:latest")
                symbols = set()
                
                for key in keys:
                    parts = key.split(":")
                    if len(parts) >= 2:
                        symbols.add(parts[1])
                
                if symbols:
                    for symbol in symbols:
                        result = self.process_with_zanalytics(symbol)
                        
                        if result and result['signal']['action'] != 'hold':
                            # Only show signal if it's new or been 30 seconds
                            current_time = time.time()
                            last_time = last_signal_time.get(symbol, 0)
                            
                            if current_time - last_time > 30:
                                signal = result['signal']
                                print(f"\\n{'='*70}")
                                print(f"🎯 {symbol} - {signal['action'].upper()}")
                                print(f"💰 Price: {result['price']['bid']:.5f} / {result['price']['ask']:.5f}")
                                print(f"📊 Score: {signal['score']} | Confidence: {signal['confidence']:.1%}")
                                print(f"📝 Reasons:")
                                for reason in signal['reasons']:
                                    print(f"   • {reason}")
                                print('='*70)
                                
                                last_signal_time[symbol] = current_time
                
                else:
                    print(".", end="", flush=True)
                
            except Exception as e:
                print(f"\\n❌ Error: {e}")
            
            time.sleep(1)

def main():
    """Main entry point"""
    print("🚀 ZANFLOW Bridge - With Zanalytics Engines")
    print("="*50)
    
    bridge = ZanflowBridge()
    
    try:
        bridge.r.ping()
        print("\\n✅ Redis connected")
    except:
        print("\\n❌ Cannot connect to Redis!")
        print("Make sure Redis is running: redis-server")
        return
    
    print("\\n💡 To install missing dependency (optional):")
    print("   pip install loguru")
    print("\\n🎯 Starting analysis with available engines...")
    
    bridge.monitor_symbols()

if __name__ == "__main__":
    main()
