FROM python:3.10-slim

WORKDIR /app

# Install system build tools
RUN apt-get update && \
    apt-get install -y gcc make wget curl build-essential && \
    rm -rf /var/lib/apt/lists/
RUN find / -name "libta_lib.so*" || true

# Download, build, and install TA-Lib C library (works on ARM/x86)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
    make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    find / -name "libta_lib.so*" || true && \
    if [ -f /usr/local/lib/libta_lib.so ]; then ln -sf /usr/local/lib/libta_lib.so /usr/lib/libta_lib.so; fi && \
    if [ -f /usr/local/lib/libta_lib.a ]; then ln -sf /usr/local/lib/libta_lib.a /usr/lib/libta_lib.a; fi && \
    ldconfig

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .
