FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY interface ./interface

# Expose port for FastAPI
EXPOSE 8080

# Set environment variables (these will be overridden by docker run -e ...)
ENV HOST=0.0.0.0
ENV PORT=8080

# Entrypoint
CMD ["uvicorn", "interface.backend.app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]