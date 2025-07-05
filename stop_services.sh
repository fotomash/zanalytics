#!/bin/bash

# Trading Analytics Platform Stop Script

echo "🛑 Stopping Trading Analytics Platform..."

# Function to stop a service
stop_service() {
    local name=$1
    if [ -f "logs/${name}.pid" ]; then
        PID=$(cat logs/${name}.pid)
        if ps -p $PID > /dev/null; then
            kill $PID
            echo "✅ Stopped $name (PID: $PID)"
        else
            echo "⚠️  $name was not running"
        fi
        rm -f logs/${name}.pid
    else
        echo "⚠️  No PID file found for $name"
    fi
}

# Stop all services
stop_service "api_server"
stop_service "mt4_bridge"
stop_service "dashboard"

echo ""
echo "✅ All services stopped"
