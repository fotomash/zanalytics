#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    if command_exists lsof; then
        lsof -i:$1 >/dev/null
    elif command_exists netstat; then
        netstat -tuln | grep ":$1 " >/dev/null
    else
        echo "Warning: Cannot check if port $1 is in use. 'lsof' or 'netstat' not found."
        return 1 # Assume not in use
    fi
    return $?
}

# --- Configuration ---
REDIS_PORT=6379
API_PORT=8000

# --- 1. Start Redis Server ---
echo "--- Starting Redis Server ---"
if ! command_exists redis-server; then
    echo "Error: redis-server command not found."
    echo "Please install Redis and ensure 'redis-server' is in your PATH."
    exit 1
fi

if port_in_use $REDIS_PORT; then
    echo "Redis is already running on port $REDIS_PORT."
else
    # Start Redis in the background
    redis-server --port $REDIS_PORT --daemonize yes
    echo "Redis server started in the background on port $REDIS_PORT."
    sleep 2 # Give it a moment to initialize
fi
echo ""

# --- 2. Start the Integrated API Server ---
echo "--- Starting Integrated API Server ---"
if port_in_use $API_PORT; then
    echo "API Server or another process is already running on port $API_PORT."
else
    # Start the API server in the background using nohup
    nohup uvicorn integrated_api_server:app --host 0.0.0.0 --port $API_PORT > api_server.log 2>&1 &
    API_PID=$!
    echo "API Server started in the background with PID $API_PID. Port: $API_PORT."
    echo "Log file: api_server.log"
fi
echo ""

# --- 3. Start the File Ingestor Service ---
echo "--- Starting File Ingestor Service ---"
# Check if the ingestor is already running
if pgrep -f "python3 file_ingestor.py" > /dev/null; then
    echo "File Ingestor service appears to be already running."
else
    # Start the file ingestor in the background using nohup
    nohup python3 file_ingestor.py > file_ingestor.log 2>&1 &
    INGESTOR_PID=$!
    echo "File Ingestor service started in the background with PID $INGESTOR_PID."
    echo "Log file: file_ingestor.log"
fi
echo ""

# --- 4. Start the Streamlit Dashboard ---
# This is run in the foreground as it's typically the main interactive component
echo "--- Starting Streamlit Dashboard ---"
echo "You can access the dashboard at http://localhost:8501"
echo "To stop all services, use the 'stop_services.sh' script."
streamlit run enhanced_dashboard.py
