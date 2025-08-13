#!/bin/bash

# Docker Run Script for Challenge 1B
# This script builds and runs the Docker container with proper volume mounts

set -e  # Exit on any error

echo "=== Challenge 1B Docker Runner ==="
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile not found. Please run this script from the 1B directory."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p test_output
mkdir -p input
mkdir -p output
mkdir -p PDFs

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t challenge1b .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully!"

# Test the Docker setup
echo "🧪 Testing Docker setup..."
docker run --rm challenge1b python test_docker_setup.py

# Check if test was successful
if [ $? -ne 0 ]; then
    echo "❌ Docker test failed! Check the error messages above."
    exit 1
fi

echo "✅ Docker setup test passed!"

# Run the container
echo "🚀 Running Challenge 1B..."
echo

# Run with volume mounts
docker run --rm \
    -v "$(pwd)/Train:/app/Train" \
    -v "$(pwd)/test_output:/app/test_output" \
    -v "$(pwd)/input:/app/input" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/PDFs:/app/PDFs" \
    -e PYTHONPATH=/app \
    -e PYTHONUNBUFFERED=1 \
    -e CUDA_VISIBLE_DEVICES="" \
    -e PYTHONHASHSEED=42 \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    challenge1b

echo
echo "✅ Processing complete!"
echo "📁 Results saved in: test_output/"
echo
echo "Generated files:"
ls -la test_output/ 