FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching and to guarantee path exists
COPY interface/requirements.txt /tmp/requirements.txt
# Be resilient to slow networks: increase timeout and retries, ensure certs
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --disable-pip-version-check --default-timeout=180 --retries 5 -r /tmp/requirements.txt

# Copy entire repo (simplifies path issues on Windows contexts)
COPY . ./

# (Deps already installed from /tmp/requirements.txt above)

# Expose port for FastAPI
EXPOSE 8080

# Set environment variables (these will be overridden by docker run -e ...)
ENV HOST=0.0.0.0
ENV PORT=8080

# Ensure PYTHONPATH includes ADOBE modules for imports in app.py (explicit)
ENV PYTHONPATH="/app/ADOBE/1A:/app/ADOBE/1B"

# Entrypoint
CMD ["uvicorn", "interface.backend.app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]