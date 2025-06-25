#!/bin/bash
# start_real_zanalytics.sh - Startup script for REAL DATA ZANALYTICS system

echo "🚀 ZANALYTICS REAL DATA SYSTEM STARTUP"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}⚠️  Port $1 is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $1 is available${NC}"
        return 0
    fi
}

# Function to start service in background
start_service() {
    local name=$1
    local command=$2
    local port=$3

    echo -e "${BLUE}🔄 Starting $name...${NC}"

    # Check if port is available
    if ! check_port $port; then
        echo -e "${RED}❌ Cannot start $name - port $port in use${NC}"
        return 1
    fi

    # Start the service
    nohup $command > logs/${name}.log 2>&1 &
    local pid=$!
    echo $pid > pids/${name}.pid

    sleep 2

    # Check if service is running
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name started successfully (PID: $pid)${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to start $name${NC}"
        return 1
    fi
}

# Create directories for logs and PIDs
mkdir -p logs pids

echo -e "${BLUE}📋 Checking system requirements...${NC}"

# Check if required files exist
required_files=(
    "real_data_zanalytics_dashboard.py"
    "ngrok_fixed.yml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ Found $file${NC}"
    else
        echo -e "${RED}❌ Missing $file${NC}"
        exit 1
    fi
done

# Check data files
data_files=(
    "XAUUSD_M1_500bars_20250623.csv"
    "XAUUSD_TICK.csv"
    "analysis_20250623_205333.json"
)

echo -e "${BLUE}📊 Checking data files...${NC}"
for file in "${data_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ Found data file: $file${NC}"
    else
        echo -e "${YELLOW}⚠️  Missing data file: $file${NC}"
    fi
done

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo -e "${RED}❌ ngrok is not installed${NC}"
    echo "Please install ngrok from https://ngrok.com/download"
    exit 1
else
    echo -e "${GREEN}✅ ngrok is installed${NC}"
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}❌ streamlit is not installed${NC}"
    echo "Install with: pip install streamlit plotly pandas"
    exit 1
else
    echo -e "${GREEN}✅ streamlit is installed${NC}"
fi

echo -e "${BLUE}🚀 Starting REAL DATA services...${NC}"

# Start Real Data Streamlit Dashboard
start_service "REAL-DASHBOARD" "streamlit run real_data_zanalytics_dashboard.py --server.port 8501 --server.headless true" 8501

# Start ZANALYTICS API Service (if exists)
if [ -f "zanalytics_api_service.py" ]; then
    start_service "ZANALYTICS-API" "python zanalytics_api_service.py" 5010
fi

# Start Data Processor (if exists)
if [ -f "ncOS_ultimate_microstructure_analyzer.py" ]; then
    start_service "DATA-PROCESSOR" "python ncOS_ultimate_microstructure_analyzer.py --api-mode --port 5011" 5011
fi

# Wait for services to fully start
echo -e "${BLUE}⏳ Waiting for services to initialize...${NC}"
sleep 5

echo -e "${BLUE}🌐 Starting ngrok tunnels with FIXED config...${NC}"

# Start ngrok tunnels with fixed config
nohup ngrok start --all --config=ngrok_fixed.yml > logs/ngrok.log 2>&1 &
echo $! > pids/ngrok.pid

echo -e "${GREEN}✅ All services started!${NC}"
echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
echo "==================="

# Check service status
services=("REAL-DASHBOARD" "ZANALYTICS-API" "DATA-PROCESSOR" "ngrok")
ports=(8501 5010 5011 4040)

for i in "${!services[@]}"; do
    service=${services[$i]}
    port=${ports[$i]}

    if [ -f "pids/${service}.pid" ]; then
        pid=$(cat pids/${service}.pid 2>/dev/null)
        if [ -n "$pid" ] && ps -p $pid > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service: Running (PID: $pid, Port: $port)${NC}"
        else
            echo -e "${RED}❌ $service: Not running${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  $service: No PID file${NC}"
    fi
done

echo ""
echo -e "${BLUE}🌍 Access URLs:${NC}"
echo "================"
echo "📊 Real Data Dashboard: http://localhost:8501"
echo "🚀 API (if running): http://localhost:5010"
echo "🔧 ngrok Console: http://localhost:4040"
echo ""
echo -e "${YELLOW}⏳ Getting ngrok URLs...${NC}"
sleep 3

# Get ngrok URLs
if command -v curl &> /dev/null; then
    ngrok_api="http://localhost:4040/api/tunnels"
    if curl -s $ngrok_api > /dev/null 2>&1; then
        echo -e "${GREEN}🌐 Public ngrok URLs:${NC}"
        echo "   Check: http://localhost:4040 for live URLs"

        # Try to extract URLs
        curl -s $ngrok_api 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for tunnel in data.get('tunnels', []):
        name = tunnel.get('name', 'unknown')
        url = tunnel.get('public_url', 'unknown')
        print(f'   {name}: {url}')
except:
    print('   URLs will be available at: http://localhost:4040')
" 2>/dev/null || echo "   URLs available at: http://localhost:4040"
    else
        echo -e "${YELLOW}⚠️  ngrok not ready yet, check http://localhost:4040${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  curl not available, check http://localhost:4040 for URLs${NC}"
fi

echo ""
echo -e "${BLUE}📋 Management Commands:${NC}"
echo "======================="
echo "🛑 Stop all: ./stop_zanalytics.sh"
echo "📊 View logs: tail -f logs/[service].log"
echo "🔄 Restart: ./restart_zanalytics.sh"
echo "📈 Check status: ./status_zanalytics.sh"
echo ""
echo -e "${GREEN}🎉 REAL DATA ZANALYTICS system is ready!${NC}"
echo -e "${BLUE}📊 Your dashboard shows ACTUAL XAUUSD trading data!${NC}"
echo -e "${YELLOW}🔗 Access your real data dashboard at: http://localhost:8501${NC}"
