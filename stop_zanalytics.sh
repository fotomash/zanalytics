#!/bin/bash
# stop_zanalytics.sh - Stop all ZANALYTICS services and ngrok

echo "🛑 STOPPING ZANALYTICS SYSTEM"
echo "=============================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to stop service
stop_service() {
    local name=$1
    local pid_file="pids/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null; then
            echo -e "${BLUE}🛑 Stopping $name (PID: $pid)...${NC}"
            kill $pid
            sleep 2
            if ps -p $pid > /dev/null; then
                echo -e "${YELLOW}⚠️  Force killing $name...${NC}"
                kill -9 $pid
            fi
            echo -e "${GREEN}✅ $name stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  $name was not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}⚠️  No PID file for $name${NC}"
    fi
}

# Stop all services
services=("ngrok" "DATA-PROCESSOR" "DASHBOARD" "ZANALYTICS-API")

for service in "${services[@]}"; do
    stop_service "$service"
done

# Clean up any remaining processes
echo -e "${BLUE}🧹 Cleaning up...${NC}"

# Kill any remaining streamlit processes
pkill -f "streamlit run" 2>/dev/null
pkill -f "zanalytics_api_service" 2>/dev/null
pkill -f "ngrok" 2>/dev/null

echo -e "${GREEN}✅ All services stopped${NC}"
echo -e "${BLUE}📋 Log files preserved in ./logs/${NC}"
