#!/bin/bash
# deploy.sh - Quick deployment script

echo "🚀 Deploying ZAnalytics Ingestion System..."

# Start Redis
docker run -d --name redis-zanalytics -p 6379:6379 redis:latest

# Install dependencies
pip install fastapi uvicorn redis pandas pyarrow aiohttp

# Start the ingestion API
echo "Starting Ingestion API on port 8000..."
python ingestion_api.py &

echo "✅ ZAnalytics Ingestion System deployed!"
echo "API available at: http://localhost:8000"
echo "Health check: http://localhost:8000/api/v1/health"