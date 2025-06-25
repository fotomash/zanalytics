FROM python:3.11-slim

# --- system libs needed for TA-Lib -----------------
RUN apt-get update && apt-get install -y \
        build-essential wget gcc make && \
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib* && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY app/ /app

# Expose:
# 8501 - Streamlit UI
# 5010 - FastAPI backend
EXPOSE 8501 5010

# Start both services with one command
CMD ["bash", "-c", "streamlit run enhanced_dashboard.py --server.port 8501 --server.enableCORS=false & uvicorn zanalytics_api_service:app --host 0.0.0.0 --port 5010"]