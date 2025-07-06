# Zanalytics Quick Start Guide

## Prerequisites
- Docker and Docker Compose
- Python 3.9+
- PostgreSQL (or use Docker)
- Redis (or use Docker)

## Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd zanalytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 3. Start with Docker Compose
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Initialize Database
```bash
# Run migrations
alembic upgrade head

# Load sample data (optional)
python scripts/load_sample_data.py
```

### 5. Access the Application
- API: http://localhost:8000
- Dashboard: http://localhost:8000/dashboard
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health

## Basic Usage

### Running Analysis
```python
import requests

response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "analysis_type": "comprehensive",
        "data_source": "sales_transactions"
    }
)

results = response.json()
```

### Detecting Anomalies
```python
response = requests.post(
    "http://localhost:8000/api/analytics/anomalies",
    json={
        "data_source": "sales_transactions",
        "contamination": 0.05
    }
)

anomalies = response.json()
```

### Getting Insights
```python
response = requests.get(
    "http://localhost:8000/api/insights/latest"
)

insights = response.json()
```

## Development Mode

### Running Locally
```bash
# Start in development mode
python run.py --dev

# Or with hot reload
uvicorn main:app --reload --port 8000
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=zanalytics

# Run specific test
pytest tests/test_data_manager.py
```

## Production Deployment

### Using Docker
```bash
# Build production image
docker build -t zanalytics:prod .

# Run production container
docker run -d \
  --name zanalytics \
  -p 80:8000 \
  -e DATABASE_URL=$DATABASE_URL \
  -e REDIS_URL=$REDIS_URL \
  zanalytics:prod
```

### Using Kubernetes
```bash
# Apply configuration
kubectl apply -f kubernetes.yml

# Check deployment
kubectl get pods -l app=zanalytics

# View logs
kubectl logs -f deployment/zanalytics
```

## Common Commands

### Data Management
```bash
# Import data
python scripts/import_data.py --source csv --file data.csv

# Export results
python scripts/export_results.py --format excel --output results.xlsx

# Clean cache
python scripts/clean_cache.py --older-than 7d
```

### Monitoring
```bash
# View performance metrics
python scripts/show_metrics.py --last 1h

# Generate performance report
python scripts/performance_report.py --output report.html

# Check system health
curl http://localhost:8000/api/health
```

## Troubleshooting

### Service Won't Start
- Check Docker is running: `docker info`
- Check ports are free: `netstat -tulpn | grep 8000`
- Check logs: `docker-compose logs zanalytics-api`

### Database Connection Issues
- Verify connection string: `echo $DATABASE_URL`
- Test connection: `psql $DATABASE_URL -c "SELECT 1"`
- Check firewall rules

### Performance Issues
- Check resource usage: `docker stats`
- Review slow queries: `python scripts/analyze_queries.py`
- Clear cache if needed: `python scripts/clean_cache.py`

## Getting Help
- Documentation: `/documentation`
- API Reference: http://localhost:8000/docs
- Logs: `/logs/zanalytics.log`
- Health Status: http://localhost:8000/api/health

## Next Steps
- Explore the API documentation
- Try different analysis types
- Configure advanced features
- Set up monitoring dashboards
- Customize for your use case
