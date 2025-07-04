from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Optional

# Global alert storage (in production, use Redis or database)
active_alerts = []
alert_history = []

@app.get("/poll/crypto")
def poll_crypto_alert(
    symbol: str = Query(..., description="Crypto symbol like BTC-USD"),
    threshold: float = Query(..., description="Price threshold"),
    direction: str = Query("above", enum=["above", "below"]),
    strategy: str = Query("price", description="Alert strategy type")
):
    """Poll crypto price and check for alert conditions"""
    
    # Get live price
    price_data = get_live_crypto_price(symbol)
    
    if "error" in price_data:
        return {"error": price_data["error"]}, 400
    
    current_price = price_data["price"]
    
    # Check alert condition
    triggered = False
    if direction == "above" and current_price >= threshold:
        triggered = True
    elif direction == "below" and current_price <= threshold:
        triggered = True
    
    alert_result = {
        "symbol": symbol,
        "current_price": current_price,
        "threshold": threshold,
        "direction": direction,
        "strategy": strategy,
        "triggered": triggered,
        "timestamp": datetime.now().isoformat(),
        "message": (
            f"🚨 ALERT: {symbol} {direction} ${threshold} - Current: ${current_price:.2f}"
            if triggered else 
            f"⏱️ Monitoring {symbol} - Current: ${current_price:.2f}, Target: {direction} ${threshold}"
        )
    }
    
    # Store alert if triggered
    if triggered:
        alert_history.append(alert_result)
    
    return alert_result

@app.post("/alerts/multi-symbol")
def multi_symbol_alerts(payload: dict):
    """Monitor multiple symbols with different strategies"""
    
    symbols = payload.get("symbols", [])
    results = []
    
    for alert_config in symbols:
        symbol = alert_config.get("symbol")
        threshold = alert_config.get("threshold")
        direction = alert_config.get("direction", "above")
        strategy = alert_config.get("strategy", "price")
        
        # Get price based on asset type
        if "BTC" in symbol or "ETH" in symbol or "crypto" in strategy.lower():
            price_data = get_live_crypto_price(symbol)
        else:
            price_data = get_live_stock_price(symbol)
        
        if "error" in price_data:
            results.append({
                "symbol": symbol,
                "error": price_data["error"],
                "strategy": strategy
            })
            continue
        
        current_price = price_data["price"]
        
        # Check condition
        triggered = (
            (direction == "above" and current_price >= threshold) or
            (direction == "below" and current_price <= threshold)
        )
        
        # Add strategy-specific logic
        strategy_signal = False
        if triggered and strategy != "price":
            strategy_signal = check_strategy_condition(symbol, strategy, price_data)
        
        result = {
            "symbol": symbol,
            "current_price": current_price,
            "threshold": threshold,
            "direction": direction,
            "strategy": strategy,
            "price_triggered": triggered,
            "strategy_confirmed": strategy_signal if strategy != "price" else triggered,
            "timestamp": datetime.now().isoformat(),
            "message": generate_alert_message(symbol, triggered, strategy_signal, strategy, current_price, threshold)
        }
        
        results.append(result)
        
        # Store significant alerts
        if triggered and (strategy == "price" or strategy_signal):
            alert_history.append(result)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total_symbols": len(symbols),
        "alerts_triggered": len([r for r in results if r.get("strategy_confirmed", False)]),
        "results": results
    }

def check_strategy_condition(symbol: str, strategy: str, price_data: dict) -> bool:
    """Check strategy-specific conditions (placeholder for your logic)"""
    
    # This is where you'd integrate your Wyckoff/ZANFLOW logic
    if strategy.lower() == "wyckoff":
        # Placeholder: Check if price is in accumulation zone
        # You'd replace this with actual Wyckoff analysis
        return True  # Simplified
    
    elif strategy.lower() == "zanflow":
        # Placeholder: Check for ZANFLOW patterns
        # You'd replace this with actual ZANFLOW analysis
        return True  # Simplified
    
    elif strategy.lower() == "breakout":
        # Check for breakout conditions
        return True  # Simplified
    
    return False

def generate_alert_message(symbol: str, price_triggered: bool, strategy_confirmed: bool, 
                         strategy: str, current_price: float, threshold: float) -> str:
    """Generate contextual alert messages"""
    
    if price_triggered and strategy_confirmed:
        if strategy == "wyckoff":
            return f"🎯 WYCKOFF SETUP: {symbol} at ${current_price:.2f} - Accumulation zone triggered!"
        elif strategy == "zanflow":
            return f"🌊 ZANFLOW SIGNAL: {symbol} at ${current_price:.2f} - Pattern confirmed!"
        else:
            return f"🚨 {strategy.upper()} ALERT: {symbol} hit ${threshold} target at ${current_price:.2f}"
    elif price_triggered:
        return f"⚠️ Price target hit for {symbol} at ${current_price:.2f}, awaiting {strategy} confirmation"
    else:
        return f"⏱️ Monitoring {symbol} - Current: ${current_price:.2f}"

@app.get("/alerts/history")
def get_alert_history(limit: int = Query(50, description="Number of alerts to return")):
    """Get recent alert history"""
    
    return {
        "total_alerts": len(alert_history),
        "recent_alerts": alert_history[-limit:],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/alerts/set")
def set_persistent_alert(
    symbol: str = Query(...),
    threshold: float = Query(...),
    direction: str = Query("above", enum=["above", "below"]),
    strategy: str = Query("price"),
    expires_in_hours: int = Query(24)
):
    """Set a persistent alert that stays active"""
    
    alert_id = f"{symbol}_{threshold}_{direction}_{datetime.now().timestamp()}"
    expires_at = datetime.now() + timedelta(hours=expires_in_hours)
    
    alert = {
        "id": alert_id,
        "symbol": symbol,
        "threshold": threshold,
        "direction": direction,
        "strategy": strategy,
        "created_at": datetime.now().isoformat(),
        "expires_at": expires_at.isoformat(),
        "active": True
    }
    
    active_alerts.append(alert)
    
    return {
        "message": f"Alert set for {symbol}",
        "alert_id": alert_id,
        "expires_at": expires_at.isoformat(),
        "total_active_alerts": len(active_alerts)
    }

@app.get("/alerts/active")
def get_active_alerts():
    """Get all active alerts"""
    
    # Remove expired alerts
    current_time = datetime.now()
    global active_alerts
    active_alerts = [
        alert for alert in active_alerts 
        if datetime.fromisoformat(alert["expires_at"]) > current_time
    ]
    
    return {
        "active_alerts": active_alerts,
        "count": len(active_alerts),
        "timestamp": datetime.now().isoformat()
    }