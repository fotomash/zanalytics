#!/bin/bash
# status_zanalytics.sh - Check ZANALYTICS system status

echo "📊 ZANALYTICS SYSTEM STATUS"
echo "============================"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

services=("ZANALYTICS-API" "DASHBOARD" "DATA-PROCESSOR" "ngrok")
ports=(5010 8501 5011 4040)

for i in "${!services[@]}"; do
    service=${services[$i]}
    port=${ports[$i]}

    echo -e "${BLUE}🔍 Checking $service...${NC}"

    if [ -f "pids/${service}.pid" ]; then
        pid=$(cat pids/${service}.pid)
        if ps -p $pid > /dev/null; then
            echo -e "${GREEN}✅ $service: Running (PID: $pid)${NC}"

            # Check if port is responding
            if curl -s "http://localhost:$port" > /dev/null 2>&1; then
                echo -e "${GREEN}   🌐 Port $port: Responding${NC}"
            else
                echo -e "${YELLOW}   ⚠️  Port $port: Not responding${NC}"
            fi
        else
            echo -e "${RED}❌ $service: Process not found${NC}"
        fi
    else
        echo -e "${RED}❌ $service: No PID file${NC}"
    fi
    echo ""
done

# Check logs for errors
echo -e "${BLUE}📋 Recent errors:${NC}"
echo "=================="

if [ -d "logs" ]; then
    find logs -name "*.log" -exec grep -l "ERROR\|Exception\|Failed" {} \; | while read logfile; do
        echo -e "${YELLOW}⚠️  Errors in $logfile:${NC}"
        tail -3 "$logfile" | grep -E "ERROR|Exception|Failed" | head -2
        echo ""
    done
else
    echo -e "${GREEN}No error logs found${NC}"
fi

echo -e "${BLUE}🌐 Access URLs:${NC}"
echo "================"
echo "📊 Dashboard: http://localhost:8501"
echo "🚀 API: http://localhost:5010"
echo "🔧 ngrok Console: http://localhost:4040"
