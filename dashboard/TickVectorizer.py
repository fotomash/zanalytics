# ncOS Tick Vectorization Module
import numpy as np
from typing import Dict, List, Tuple

class TickVectorizer:
    def __init__(self):
        self.vector_dim = 1536
        self.tick_buffer = []
        self.session_vectors = {}
        
    def vectorize_tick_sequence(self, ticks: List[Dict]) -> np.ndarray:
        """Convert tick sequence to 1536-dim embedding"""
        # Extract microstructure features
        features = []
        
        # 1. Tick density vector (256 dims)
        tick_intervals = np.diff([t['timestamp'] for t in ticks])
        density_vec = np.histogram(tick_intervals, bins=256)[0]
        features.extend(density_vec / (density_vec.sum() + 1e-9))
        
        # 2. Spread dynamics (256 dims)
        spreads = [t['spread'] for t in ticks]
        spread_fft = np.abs(np.fft.fft(spreads, n=256))
        features.extend(spread_fft / (spread_fft.max() + 1e-9))
        
        # 3. Volume profile (512 dims)
        volumes = [t.get('inferred_volume', 100) for t in ticks]
        price_levels = np.linspace(min([t['bid'] for t in ticks]), 
                                  max([t['ask'] for t in ticks]), 512)
        vol_profile = np.zeros(512)
        for i, tick in enumerate(ticks):
            idx = np.argmin(np.abs(price_levels - tick['price_mid']))
            vol_profile[idx] += volumes[i]
        features.extend(vol_profile / (vol_profile.sum() + 1e-9))
        
        # 4. Microstructure signature (512 dims)
        # Includes tick reversal patterns, order flow imbalance, etc
        micro_sig = self._compute_microstructure_signature(ticks)
        features.extend(micro_sig)
        
        return np.array(features[:self.vector_dim])
    
    def _compute_microstructure_signature(self, ticks: List[Dict]) -> np.ndarray:
        """Extract 512-dim microstructure signature"""
        sig = np.zeros(512)
        
        # Tick reversal patterns (128 dims)
        price_changes = np.diff([t['price_mid'] for t in ticks])
        reversal_patterns = np.convolve(price_changes, [-1, 2, -1], mode='same')
        sig[:128] = np.histogram(reversal_patterns, bins=128)[0]
        
        # Order flow imbalance (128 dims)
        bid_pressure = np.diff([t['bid'] for t in ticks])
        ask_pressure = np.diff([t['ask'] for t in ticks])
        imbalance = bid_pressure - ask_pressure
        sig[128:256] = np.histogram(imbalance, bins=128)[0]
        
        # VPIN-like toxicity (128 dims)
        buy_vol = sum([t.get('inferred_volume', 0) for t in ticks if t.get('price_direction', 0) > 0])
        sell_vol = sum([t.get('inferred_volume', 0) for t in ticks if t.get('price_direction', 0) < 0])
        vpin_local = abs(buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)
        sig[256:384] = np.full(128, vpin_local)
        
        # Spread spike patterns (128 dims)
        spread_spikes = [t['spread'] / (np.mean([x['spread'] for x in ticks]) + 1e-9) for t in ticks]
        sig[384:512] = np.histogram(spread_spikes, bins=128)[0]
        
        return sig / (np.linalg.norm(sig) + 1e-9)  # L2 normalize

# Integration with ncOS agents
class LiquidityTrapAgent:
    def __init__(self):
        self.vectorizer = TickVectorizer()
        self.session_memory = {}
        self.maturity_weights = {
            'tick_density': 0.20,
            'spread_stability': 0.15,
            'volume_signature': 0.25,
            'reversal_score': 0.40
        }
    
    def process_tick_window(self, ticks: List[Dict], session_id: str) -> Dict:
        """Process tick window and return signal assessment"""
        # Vectorize current tick window
        tick_vector = self.vectorizer.vectorize_tick_sequence(ticks)
        
        # Store in session memory
        self.session_memory[session_id] = {
            'tick_vector': tick_vector,
            'tick_count': len(ticks),
            'timestamp': ticks[-1]['timestamp']
        }
        
        # Calculate maturity components
        maturity_components = {
            'tick_density': self._calc_density_score(ticks),
            'spread_stability': self._calc_spread_score(ticks),
            'volume_signature': self._calc_volume_score(ticks),
            'reversal_score': self._calc_reversal_score(ticks)
        }
        
        # Weighted maturity score
        maturity_score = sum(self.maturity_weights[k] * v 
                           for k, v in maturity_components.items())
        
        # Vector similarity to known trap patterns
        trap_similarity = self._check_trap_patterns(tick_vector)
        
        return {
            'session_id': session_id,
            'maturity_score': maturity_score,
            'maturity_components': maturity_components,
            'trap_similarity': trap_similarity,
            'tick_vector_norm': np.linalg.norm(tick_vector),
            'action': 'SIGNAL' if maturity_score > 0.75 else 'MONITOR'
        }
    
    def _calc_density_score(self, ticks: List[Dict]) -> float:
        """Calculate tick density score (0-1)"""
        intervals = np.diff([t['timestamp'] for t in ticks])
        if len(intervals) == 0:
            return 0.0
        rate = 1000 / (np.mean(intervals) + 1)  # ticks per second
        return min(rate / 100, 1.0)  # Normalize to 0-1
    
    def _calc_spread_score(self, ticks: List[Dict]) -> float:
        """Calculate spread stability score (0-1)"""
        spreads = [t['spread'] for t in ticks]
        if np.std(spreads) == 0:
            return 1.0
        cv = np.std(spreads) / (np.mean(spreads) + 1e-9)  # Coefficient of variation
        return max(0, 1 - cv)  # Lower CV = higher score
    
    def _calc_volume_score(self, ticks: List[Dict]) -> float:
        """Calculate volume signature score (0-1)"""
        volumes = [t.get('inferred_volume', 100) for t in ticks]
        # Look for volume spikes
        vol_mean = np.mean(volumes)
        vol_max = np.max(volumes)
        spike_ratio = vol_max / (vol_mean + 1e-9)
        return min(spike_ratio / 5, 1.0)  # Normalize spike ratio
    
    def _calc_reversal_score(self, ticks: List[Dict]) -> float:
        """Calculate tick reversal score (0-1)"""
        if len(ticks) < 3:
            return 0.0
        
        # Check for sweep and snapback pattern
        prices = [t['price_mid'] for t in ticks]
        max_idx = np.argmax(prices)
        min_idx = np.argmin(prices)
        
        # Ideal pattern: price extremum followed by sharp reversal
        if max_idx < len(prices) - 1:
            # Upward sweep followed by reversal
            reversal_magnitude = (prices[max_idx] - prices[-1]) / prices[max_idx]
            reversal_speed = (len(prices) - max_idx) / len(prices)
            return min(reversal_magnitude * 100 * reversal_speed, 1.0)
        elif min_idx < len(prices) - 1:
            # Downward sweep followed by reversal
            reversal_magnitude = (prices[-1] - prices[min_idx]) / prices[min_idx]
            reversal_speed = (len(prices) - min_idx) / len(prices)
            return min(reversal_magnitude * 100 * reversal_speed, 1.0)
        
        return 0.0
    
    def _check_trap_patterns(self, tick_vector: np.ndarray) -> float:
        """Compare against known trap pattern embeddings"""
        # In production, load from vector DB
        known_trap_patterns = self._load_trap_patterns()
        
        if not known_trap_patterns:
            return 0.0
        
        # Cosine similarity
        similarities = []
        for pattern in known_trap_patterns:
            sim = np.dot(tick_vector, pattern) / (np.linalg.norm(tick_vector) * np.linalg.norm(pattern) + 1e-9)
            similarities.append(sim)
        
        return max(similarities)
    
    def _load_trap_patterns(self) -> List[np.ndarray]:
        """Load known trap pattern vectors from storage"""
        # Placeholder - would load from vector store
        return []