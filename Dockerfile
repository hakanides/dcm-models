# DCM Estimation Framework
# Multi-stage build for smaller final image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy application code
COPY . .

# Create directories for results and data
RUN mkdir -p /app/results /app/data

# Default command: show help
CMD ["python", "-c", "print('DCM Estimation Framework\\n\\nUsage:\\n  python run_all_models.py\\n  python -m pytest tests/\\n\\nModules:\\n  src/models/      - Model specifications\\n  src/estimation/  - Estimation utilities\\n  src/simulation/  - Data generation\\n  src/utils/       - Utilities')"]
