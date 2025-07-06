"""
London Killzone Agent - Specialist agent for executing the London Kill Zone SMC strategy.
Implements the complete workflow from Asian session analysis to trade idea generation.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import redis
import pandas as pd
import numpy as np
from loguru import logger

class LondonKillzoneAgent:
    """
    Specialist agent that executes the London Kill Zone trading strategy.
    Follows the workflow defined in the strategy manifest step by step.
    """
    
    def __init__(self, redis_client: redis.Redis, manifest: Dict[str, Any]):
        self.redis = redis_client
        self.manifest = manifest
        self.strategy_id = manifest["strategy_id"]
        self.workflow_state = {}
        self.execution_id = None
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete London Killzone strategy workflow.
        
        Args:
            context: Execution context from the scheduler
            
        Returns:
            Execution result with status and any trade ideas found
        """
        self.execution_id = f"{self.strategy_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting London Killzone execution: {self.execution_id}")
        
        # Update agent status
        await self._update_status("running", "Starting workflow execution")
        
        try:
            # Execute workflow steps
            result = await self._execute_workflow()
            
            # Update final status
            if result.get("trade_ideas"):
                await self._update_status("completed", f"Found {len(result['trade_ideas'])} trade setups")
            else:
                await self._update_status("completed", "No valid setups found")
                
            return result
            
        except Exception as e:
            logger.error(f"Error executing London Killzone strategy: {e}")
            await self._update_status("failed", str(e))
            raise
            
    async def _execute_workflow(self) -> Dict[str, Any]:
        """Execute the workflow steps defined in the manifest."""
        workflow = self.manifest.get("workflow", [])
        trade_ideas = []
        
        # Get allowed trading pairs
        allowed_pairs = self.manifest.get("risk_params", {}).get("allowed_pairs", [])
        
        # Execute workflow for each allowed pair
        for symbol in allowed_pairs:
            logger.info(f"Analyzing {symbol} for London Killzone setup...")
            
            # Reset workflow state for each symbol
            self.workflow_state = {"symbol": symbol}
            
            # Execute each step
            for step in workflow:
                if not await self._should_execute_step(step):
                    logger.info(f"Skipping step {step['name']} for {symbol} - precondition not met")
                    break
                    
                success = await self._execute_step(step, symbol)
                if not success:
                    logger.info(f"Step {step['name']} failed for {symbol}, moving to next pair")
                    break
                    
            # Check if we completed all steps successfully
            if self.workflow_state.get("trade_idea_ready"):
                trade_ideas.append(self.workflow_state["trade_idea"])
                
        return {
            "execution_id": self.execution_id,
            "timestamp": datetime.utcnow().isoformat(),
            "trade_ideas": trade_ideas,
            "symbols_analyzed": allowed_pairs
        }
        
    async def _should_execute_step(self, step: Dict[str, Any]) -> bool:
        """Check if a step's preconditions are met."""
        pre_condition = step.get("pre_condition", "")
        
        if "Asian session range is identified" in pre_condition:
            return "asia_high" in self.workflow_state and "asia_low" in self.workflow_state
            
        elif "liquidity sweep has been confirmed" in pre_condition:
            return self.workflow_state.get("sweep_detected", False)
            
        elif "Market structure has shifted" in pre_condition:
            return self.workflow_state.get("structure_confirmed", False)
            
        elif "Valid entry zone identified" in pre_condition:
            return "optimal_entry" in self.workflow_state
            
        elif "All trade parameters calculated" in pre_condition:
            return all(key in self.workflow_state for key in ["entry_price", "stop_loss", "take_profit_1"])
            
        return True  # No specific precondition or default
        
    async def _execute_step(self, step: Dict[str, Any], symbol: str) -> bool:
        """Execute a single workflow step."""
        step_name = step["name"]
        logger.info(f"Executing step: {step_name} for {symbol}")
        
        if step_name == "identify_asian_range":
            return await self._identify_asian_range(symbol, step)
            
        elif step_name == "detect_liquidity_sweep":
            return await self._detect_liquidity_sweep(symbol, step)
            
        elif step_name == "confirm_market_structure":
            return await self._confirm_market_structure(symbol, step)
            
        elif step_name == "refine_entry_zone":
            return await self._refine_entry_zone(symbol, step)
            
        elif step_name == "calculate_trade_parameters":
            return await self._calculate_trade_parameters(symbol, step)
            
        elif step_name == "execute_trade_idea":
            return await self._execute_trade_idea(symbol, step)
            
        return False
        
    async def _identify_asian_range(self, symbol: str, step: Dict[str, Any]) -> bool:
        """Identify the Asian session high and low."""
        params = step.get("params", {})
        lookback_hours = params.get("lookback_hours", 8)
        
        # Get Asian session time range (typically 00:00 - 08:00 GMT)
        now = datetime.now(timezone.utc)
        asian_end = now.replace(hour=6, minute=0, second=0, microsecond=0)
        asian_start = asian_end - timedelta(hours=lookback_hours)
        
        # Fetch price data for Asian session
        price_data = await self._get_price_data(symbol, asian_start, asian_end, "H1")
        
        if price_data.empty:
            logger.warning(f"No price data available for {symbol} Asian session")
            return False
            
        # Calculate high and low
        asia_high = price_data['high'].max()
        asia_low = price_data['low'].min()
        asia_range = asia_high - asia_low
        
        # Store in workflow state
        self.workflow_state.update({
            "asia_high": asia_high,
            "asia_low": asia_low,
            "asia_range_size": asia_range,
            "asia_range_pips": self._price_to_pips(symbol, asia_range)
        })
        
        logger.info(f"{symbol} Asian Range: High={asia_high}, Low={asia_low}, Range={self.workflow_state['asia_range_pips']} pips")
        
        # Check if range is significant enough (>20 pips as per manifest)
        return self.workflow_state["asia_range_pips"] > 20
        
    async def _detect_liquidity_sweep(self, symbol: str, step: Dict[str, Any]) -> bool:
        """Detect if price has swept Asian session liquidity."""
        params = step.get("params", {})
        min_sweep_distance = params.get("min_sweep_distance", 5)
        volume_threshold = params.get("volume_threshold", 1.5)
        
        # Get current London session data
        now = datetime.now(timezone.utc)
        london_start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        
        price_data = await self._get_price_data(symbol, london_start, now, "M15")
        
        if price_data.empty:
            return False
            
        # Check for sweep of Asian highs or lows
        asia_high = self.workflow_state["asia_high"]
        asia_low = self.workflow_state["asia_low"]
        
        # Look for sweep above Asian high
        high_sweep = price_data[price_data['high'] > asia_high + self._pips_to_price(symbol, min_sweep_distance)]
        
        # Look for sweep below Asian low
        low_sweep = price_data[price_data['low'] < asia_low - self._pips_to_price(symbol, min_sweep_distance)]
        
        sweep_detected = False
        sweep_direction = None
        sweep_level = None
        
        if not high_sweep.empty:
            # Check volume on sweep
            avg_volume = price_data['volume'].mean()
            sweep_volume = high_sweep.iloc[0]['volume']
            
            if sweep_volume > avg_volume * volume_threshold:
                sweep_detected = True
                sweep_direction = "bullish_sweep"  # Swept highs, expect reversal down
                sweep_level = asia_high
                logger.info(f"{symbol} detected bullish liquidity sweep above {asia_high}")
                
        elif not low_sweep.empty:
            # Check volume on sweep
            avg_volume = price_data['volume'].mean()
            sweep_volume = low_sweep.iloc[0]['volume']
            
            if sweep_volume > avg_volume * volume_threshold:
                sweep_detected = True
                sweep_direction = "bearish_sweep"  # Swept lows, expect reversal up
                sweep_level = asia_low
                logger.info(f"{symbol} detected bearish liquidity sweep below {asia_low}")
                
        if sweep_detected:
            self.workflow_state.update({
                "sweep_detected": True,
                "sweep_direction": sweep_direction,
                "sweep_level": sweep_level,
                "sweep_time": high_sweep.index[0] if sweep_direction == "bullish_sweep" else low_sweep.index[0]
            })
            
        return sweep_detected
        
    async def _confirm_market_structure(self, symbol: str, step: Dict[str, Any]) -> bool:
        """Confirm market structure shift (BOS/CHoCH) after liquidity sweep."""
        params = step.get("params", {})
        min_structure_break = params.get("min_structure_break", 10)
        
        # Get M5 data after the sweep
        sweep_time = self.workflow_state.get("sweep_time")
        if not sweep_time:
            return False
            
        price_data = await self._get_price_data(symbol, sweep_time, datetime.now(timezone.utc), "M5")
        
        if price_data.empty:
            return False
            
        # Look for structure break based on sweep direction
        sweep_direction = self.workflow_state["sweep_direction"]
        structure_confirmed = False
        bias_direction = None
        structure_break_level = None
        
        if sweep_direction == "bullish_sweep":
            # After sweeping highs, look for bearish structure (lower low)
            # Find swing highs and lows
            swing_highs, swing_lows = self._identify_swings(price_data)
            
            if len(swing_lows) >= 2:
                # Check if we made a lower low
                if swing_lows[-1]['price'] < swing_lows[-2]['price'] - self._pips_to_price(symbol, min_structure_break):
                    structure_confirmed = True
                    bias_direction = "sell"
                    structure_break_level = swing_lows[-2]['price']
                    
        elif sweep_direction == "bearish_sweep":
            # After sweeping lows, look for bullish structure (higher high)
            swing_highs, swing_lows = self._identify_swings(price_data)
            
            if len(swing_highs) >= 2:
                # Check if we made a higher high
                if swing_highs[-1]['price'] > swing_highs[-2]['price'] + self._pips_to_price(symbol, min_structure_break):
                    structure_confirmed = True
                    bias_direction = "buy"
                    structure_break_level = swing_highs[-2]['price']
                    
        if structure_confirmed:
            self.workflow_state.update({
                "structure_confirmed": True,
                "bias_direction": bias_direction,
                "structure_break_level": structure_break_level,
                "structure_type": "BOS"  # Break of Structure
            })
            logger.info(f"{symbol} confirmed {bias_direction} structure shift")
            
        return structure_confirmed
        
    async def _refine_entry_zone(self, symbol: str, step: Dict[str, Any]) -> bool:
        """Refine entry using MIDAS curve, FVGs, and order blocks."""
        params = step.get("params", {})
        
        # Get M1 data for precision entry
        now = datetime.now(timezone.utc)
        lookback = now - timedelta(minutes=60)
        
        price_data = await self._get_price_data(symbol, lookback, now, "M1")
        
        if price_data.empty:
            return False
            
        # Calculate MIDAS curve
        midas_level = self._calculate_midas(price_data, params.get("midas_period", 20))
        
        # Find Fair Value Gaps (FVGs)
        fvgs = self._find_fvgs(price_data, self._pips_to_price(symbol, params.get("fvg_min_size", 3)))
        
        # Find Order Blocks
        order_blocks = self._find_order_blocks(price_data, params.get("order_block_lookback", 50))
        
        # Determine optimal entry zone
        bias = self.workflow_state["bias_direction"]
        current_price = price_data.iloc[-1]['close']
        
        entry_zone_high = None
        entry_zone_low = None
        optimal_entry = None
        
        if bias == "buy":
            # For buys, look for bullish FVGs or order blocks below current price
            valid_fvgs = [fvg for fvg in fvgs if fvg['type'] == 'bullish' and fvg['low'] < current_price]
            valid_obs = [ob for ob in order_blocks if ob['type'] == 'bullish' and ob['low'] < current_price]
            
            if valid_fvgs:
                # Use the nearest FVG
                nearest_fvg = min(valid_fvgs, key=lambda x: current_price - x['high'])
                entry_zone_high = nearest_fvg['high']
                entry_zone_low = nearest_fvg['low']
                optimal_entry = (entry_zone_high + entry_zone_low) / 2
                
        elif bias == "sell":
            # For sells, look for bearish FVGs or order blocks above current price
            valid_fvgs = [fvg for fvg in fvgs if fvg['type'] == 'bearish' and fvg['high'] > current_price]
            valid_obs = [ob for ob in order_blocks if ob['type'] == 'bearish' and ob['high'] > current_price]
            
            if valid_fvgs:
                # Use the nearest FVG
                nearest_fvg = min(valid_fvgs, key=lambda x: x['low'] - current_price)
                entry_zone_high = nearest_fvg['high']
                entry_zone_low = nearest_fvg['low']
                optimal_entry = (entry_zone_high + entry_zone_low) / 2
                
        if optimal_entry:
            self.workflow_state.update({
                "entry_zone_high": entry_zone_high,
                "entry_zone_low": entry_zone_low,
                "midas_level": midas_level,
                "optimal_entry": optimal_entry,
                "entry_quality_score": self._calculate_entry_quality(optimal_entry, midas_level, current_price)
            })
            logger.info(f"{symbol} found entry zone: {entry_zone_low} - {entry_zone_high}")
            return True
            
        return False
        
    async def _calculate_trade_parameters(self, symbol: str, step: Dict[str, Any]) -> bool:
        """Calculate precise entry, stop loss, and take profit levels."""
        params = step.get("params", {})
        rr_ratios = params.get("rr_ratios", [1.5, 2.5, 4.0])
        
        optimal_entry = self.workflow_state["optimal_entry"]
        bias = self.workflow_state["bias_direction"]
        
        # Calculate stop loss
        if bias == "buy":
            # For buys, stop below the sweep low or recent swing low
            stop_loss = self.workflow_state["sweep_level"] - self._pips_to_price(symbol, 10)
        else:
            # For sells, stop above the sweep high or recent swing high
            stop_loss = self.workflow_state["sweep_level"] + self._pips_to_price(symbol, 10)
            
        # Calculate risk in pips
        risk_pips = abs(self._price_to_pips(symbol, optimal_entry - stop_loss))
        
        # Calculate take profit levels based on R:R ratios
        take_profits = []
        for rr in rr_ratios:
            if bias == "buy":
                tp = optimal_entry + (optimal_entry - stop_loss) * rr
            else:
                tp = optimal_entry - (stop_loss - optimal_entry) * rr
            take_profits.append(tp)
            
        # Calculate position size based on risk management
        risk_percent = self.manifest.get("risk_params", {}).get("max_risk_per_trade", 0.01)
        account_balance = 10000  # This would come from account data in production
        risk_amount = account_balance * risk_percent
        
        # Simplified position size calculation (would be more complex in production)
        position_size = risk_amount / (risk_pips * 10)  # $10 per pip for standard lot
        
        self.workflow_state.update({
            "entry_price": optimal_entry,
            "stop_loss": stop_loss,
            "take_profit_1": take_profits[0],
            "take_profit_2": take_profits[1],
            "take_profit_3": take_profits[2],
            "position_size": round(position_size, 2),
            "risk_amount": risk_amount,
            "risk_pips": risk_pips,
            "trade_direction": bias
        })
        
        logger.info(f"{symbol} trade parameters calculated: Entry={optimal_entry}, SL={stop_loss}, TP1={take_profits[0]}")
        return True
        
    async def _execute_trade_idea(self, symbol: str, step: Dict[str, Any]) -> bool:
        """Create and dispatch the trade idea command."""
        # Build the trade idea from workflow state
        trade_idea = {
            "symbol": symbol,
            "direction": self.workflow_state["trade_direction"],
            "entry_price": self.workflow_state["entry_price"],
            "stop_loss": self.workflow_state["stop_loss"],
            "take_profits": [
                self.workflow_state["take_profit_1"],
                self.workflow_state["take_profit_2"],
                self.workflow_state["take_profit_3"]
            ],
            "position_size": self.workflow_state["position_size"],
            "risk_amount": self.workflow_state["risk_amount"],
            "strategy_version": self.strategy_id,
            "analysis_context": {
                "asian_range": f"{self.workflow_state['asia_range_pips']} pips",
                "sweep_type": self.workflow_state["sweep_direction"],
                "structure_shift": self.workflow_state["structure_type"],
                "entry_quality_score": self.workflow_state["entry_quality_score"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.workflow_state["trade_idea"] = trade_idea
        self.workflow_state["trade_idea_ready"] = True
        
        # Create command to log the trade idea
        command = {
            "request_id": f"{self.execution_id}_trade_{symbol}",
            "action_type": "EXECUTE_TRADE_IDEA",
            "payload": {
                "source": f"LondonKillzone_Agent",
                "content": f"High-probability {trade_idea['direction']} trade identified on {symbol} based on London Kill Zone strategy.",
                "trade_setup": trade_idea,
                "charts_to_highlight": [
                    {
                        "timeframe": "H1",
                        "annotations": ["asian_high", "asian_low"]
                    },
                    {
                        "timeframe": "M5", 
                        "annotations": ["structure_break", "entry_zone"]
                    }
                ]
            },
            "metadata": {
                "priority": "high",
                "source": self.strategy_id
            }
        }
        
        # Queue the command for the dispatcher
        self.redis.lpush("zanalytics:command_queue", json.dumps(command))
        
        logger.success(f"Trade idea created for {symbol}: {trade_idea['direction']} @ {trade_idea['entry_price']}")
        return True
        
    # Helper methods
    
    async def _get_price_data(self, symbol: str, start_time: datetime, end_time: datetime, timeframe: str) -> pd.DataFrame:
        """Fetch price data from Redis or data source."""
        # In production, this would fetch from your data manager
        # For now, return mock data for demonstration
        
        # Check Redis for cached data
        cache_key = f"zanalytics:prices:{symbol}:{timeframe}:{start_time.isoformat()}:{end_time.isoformat()}"
        cached_data = self.redis.get(cache_key)
        
        if cached_data:
            return pd.read_json(cached_data)
            
        # Generate mock data for demonstration
        periods = {
            "M1": 1, "M5": 5, "M15": 15, "H1": 60
        }
        
        num_candles = int((end_time - start_time).total_seconds() / 60 / periods[timeframe])
        
        # Create realistic price data
        dates = pd.date_range(start=start_time, end=end_time, freq=f"{periods[timeframe]}min")[:num_candles]
        
        # Base prices for different symbols
        base_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "XAUUSD": 2050.00,
            "USDJPY": 150.50
        }
        
        base_price = base_prices.get(symbol, 1.0)
        
        # Generate OHLCV data
        np.random.seed(int(start_time.timestamp()))  # Consistent data for same request
        
        df = pd.DataFrame({
            'open': base_price + np.random.randn(len(dates)) * 0.001,
            'high': base_price + np.random.randn(len(dates)) * 0.001 + 0.0005,
            'low': base_price + np.random.randn(len(dates)) * 0.001 - 0.0005,
            'close': base_price + np.random.randn(len(dates)) * 0.001,
            'volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        # Cache the data
        self.redis.setex(cache_key, 3600, df.to_json())
        
        return df
        
    def _price_to_pips(self, symbol: str, price_diff: float) -> float:
        """Convert price difference to pips."""
        if "JPY" in symbol:
            return price_diff * 100
        elif "XAU" in symbol:
            return price_diff * 10
        else:
            return price_diff * 10000
            
    def _pips_to_price(self, symbol: str, pips: float) -> float:
        """Convert pips to price difference."""
        if "JPY" in symbol:
            return pips / 100
        elif "XAU" in symbol:
            return pips / 10
        else:
            return pips / 10000
            
    def _identify_swings(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Identify swing highs and lows in price data."""
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df) - 2):
            # Swing high: higher than 2 candles before and after
            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i-2]['high'] and
                df.iloc[i]['high'] > df.iloc[i+1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i+2]['high']):
                swing_highs.append({
                    'index': i,
                    'time': df.index[i],
                    'price': df.iloc[i]['high']
                })
                
            # Swing low: lower than 2 candles before and after
            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i-2]['low'] and
                df.iloc[i]['low'] < df.iloc[i+1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i+2]['low']):
                swing_lows.append({
                    'index': i,
                    'time': df.index[i],
                    'price': df.iloc[i]['low']
                })
                
        return swing_highs, swing_lows
        
    def _calculate_midas(self, df: pd.DataFrame, period: int) -> float:
        """Calculate MIDAS (Market Interpretation Data Analysis System) level."""
        # Simplified MIDAS calculation - VWAP-based
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume = df['volume']
        
        cumulative_tp_volume = (typical_price * volume).rolling(window=period).sum()
        cumulative_volume = volume.rolling(window=period).sum()
        
        midas = cumulative_tp_volume / cumulative_volume
        
        return midas.iloc[-1]
        
    def _find_fvgs(self, df: pd.DataFrame, min_gap_size: float) -> List[Dict]:
        """Find Fair Value Gaps (imbalances) in price data."""
        fvgs = []
        
        for i in range(2, len(df)):
            # Bullish FVG: Gap between candle[i-2] high and candle[i] low
            gap_up = df.iloc[i]['low'] - df.iloc[i-2]['high']
            if gap_up > min_gap_size:
                fvgs.append({
                    'type': 'bullish',
                    'high': df.iloc[i]['low'],
                    'low': df.iloc[i-2]['high'],
                    'time': df.index[i-1],
                    'size': gap_up
                })
                
            # Bearish FVG: Gap between candle[i-2] low and candle[i] high
            gap_down = df.iloc[i-2]['low'] - df.iloc[i]['high']
            if gap_down > min_gap_size:
                fvgs.append({
                    'type': 'bearish',
                    'high': df.iloc[i-2]['low'],
                    'low': df.iloc[i]['high'],
                    'time': df.index[i-1],
                    'size': gap_down
                })
                
        return fvgs
        
    def _find_order_blocks(self, df: pd.DataFrame, lookback: int) -> List[Dict]:
        """Find order blocks (last opposite candle before strong move)."""
        order_blocks = []
        
        # Look for strong moves and identify the last opposite candle
        for i in range(lookback, len(df)):
            # Bullish order block: Last bearish candle before strong up move
            if i >= 3:
                # Check for strong bullish move (3 consecutive up candles)
                if all(df.iloc[j]['close'] > df.iloc[j]['open'] for j in range(i-2, i+1)):
                    # Find last bearish candle
                    for j in range(i-3, max(i-lookback, 0), -1):
                        if df.iloc[j]['close'] < df.iloc[j]['open']:
                            order_blocks.append({
                                'type': 'bullish',
                                'high': df.iloc[j]['high'],
                                'low': df.iloc[j]['low'],
                                'time': df.index[j],
                                'strength': sum(df.iloc[k]['close'] - df.iloc[k]['open'] for k in range(i-2, i+1))
                            })
                            break
                            
        return order_blocks
        
    def _calculate_entry_quality(self, entry: float, midas: float, current: float) -> float:
        """Calculate quality score for the entry (0-100)."""
        score = 100.0
        
        # Distance from MIDAS (closer is better)
        midas_distance = abs(entry - midas) / midas
        score -= midas_distance * 1000  # Penalize distance from MIDAS
        
        # Distance from current price (not too far)
        price_distance = abs(entry - current) / current
        if price_distance > 0.002:  # More than 0.2%
            score -= (price_distance - 0.002) * 1000
            
        return max(0, min(100, score))
        
    async def _update_status(self, status: str, message: str):
        """Update agent execution status in Redis."""
        status_data = {
            "strategy_id": self.strategy_id,
            "execution_id": self.execution_id,
            "status": status,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.redis.setex(
            f"zanalytics:agent_status:{self.strategy_id}",
            300,  # 5 minute TTL
            json.dumps(status_data)
        )


class LondonKillzoneAgentFactory:
    """Factory for creating London Killzone agents."""
    
    @staticmethod
    def create(redis_client: redis.Redis, manifest: Dict[str, Any]) -> LondonKillzoneAgent:
        """Create a new London Killzone agent instance."""
        return LondonKillzoneAgent(redis_client, manifest)
