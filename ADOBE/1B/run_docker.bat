@echo off
REM Docker Run Script for Challenge 1B (Windows)
REM This script builds and runs the Docker container with proper volume mounts

echo === Challenge 1B Docker Runner ===
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "Dockerfile" (
    echo ❌ Dockerfile not found. Please run this script from the 1B directory.
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "test_output" mkdir test_output
if not exist "input" mkdir input
if not exist "output" mkdir output
if not exist "PDFs" mkdir PDFs

REM Build the Docker image
echo 🔨 Building Docker image...
docker build -t challenge1b .

REM Check if build was successful
if %errorlevel% neq 0 (
    echo ❌ Docker build failed!
    pause
    exit /b 1
)

echo ✅ Docker image built successfully!

REM Test the Docker setup
echo 🧪 Testing Docker setup...
docker run --rm challenge1b python test_docker_setup.py

REM Check if test was successful
if %errorlevel% neq 0 (
    echo ❌ Docker test failed! Check the error messages above.
    pause
    exit /b 1
)

echo ✅ Docker setup test passed!

REM Run the container
echo 🚀 Running Challenge 1B...
echo.

REM Run with volume mounts (Windows path format)
docker run --rm ^
    -v "%cd%/Train:/app/Train" ^
    -v "%cd%/test_output:/app/test_output" ^
    -v "%cd%/input:/app/input" ^
    -v "%cd%/output:/app/output" ^
    -v "%cd%/PDFs:/app/PDFs" ^
    -e PYTHONPATH=/app ^
    -e PYTHONUNBUFFERED=1 ^
    -e CUDA_VISIBLE_DEVICES="" ^
    -e PYTHONHASHSEED=42 ^
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 ^
    challenge1b

echo.
echo ✅ Processing complete!
echo 📁 Results saved in: test_output/
echo.
echo Generated files:
dir test_output

pause 