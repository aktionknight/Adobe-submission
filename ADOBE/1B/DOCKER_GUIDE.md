# Docker Guide for Challenge 1B

This guide provides step-by-step instructions for running the Challenge 1B project using Docker.

## Prerequisites

1. **Install Docker Desktop**
   - Download from: https://www.docker.com/products/docker-desktop
   - Install and start Docker Desktop
   - Ensure Docker is running (check with `docker --version`)

2. **Project Structure**
   Ensure your project structure looks like this:
   ```
   1B/
   ├── Dockerfile
   ├── docker-compose.yml
   ├── run_docker.sh (Linux/macOS)
   ├── run_docker.bat (Windows)
   ├── test_docker_setup.py
   ├── main.py
   ├── requirements.txt
   ├── Train/
   │   ├── Collection 1/
   │   ├── Collection 2/
   │   └── Collection 3/
   └── test_output/
   ```

## Quick Start

### Option 1: Automated Scripts (Recommended)

**For Linux/macOS:**
```bash
cd 1B
chmod +x run_docker.sh
./run_docker.sh
```

**For Windows:**
```cmd
cd 1B
run_docker.bat
```

The automated scripts will:
1. Build the Docker image
2. Test the setup to ensure all dependencies work
3. Run the main processing
4. Save results to test_output/

### Option 2: Manual Docker Commands

1. **Navigate to the project directory:**
```bash
cd 1B
```

2. **Build the Docker image:**
```bash
docker build -t challenge1b .
```

3. **Test the setup:**
```bash
docker run --rm challenge1b python test_docker_setup.py
```

4. **Run the container:**
```bash
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
```

### Option 3: Docker Compose

```bash
cd 1B
docker-compose up --build
```

## Windows PowerShell

If you're using PowerShell on Windows:

```powershell
docker run --rm `
    -v "${PWD}/Train:/app/Train" `
    -v "${PWD}/test_output:/app/test_output" `
    -v "${PWD}/input:/app/input" `
    -v "${PWD}/output:/app/output" `
    -v "${PWD}/PDFs:/app/PDFs" `
    -e PYTHONPATH=/app `
    -e PYTHONUNBUFFERED=1 `
    -e CUDA_VISIBLE_DEVICES="" `
    -e PYTHONHASHSEED=42 `
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 `
    challenge1b
```

## What the Docker Container Does

1. **Builds the environment** with Python 3.9 and all required dependencies including:
   - PyMuPDF for PDF processing
   - pytesseract for OCR functionality
   - sentence-transformers for semantic analysis
   - scikit-learn for machine learning
   - PyTorch for deep learning
   - Tesseract OCR system dependencies

2. **Mounts volumes** to access your local files:
   - `Train/` - Contains the training collections
   - `test_output/` - Where results are saved
   - `input/` - Additional input files
   - `output/` - Additional output files
   - `PDFs/` - PDF files directory

3. **Tests the setup** to ensure all dependencies work correctly

4. **Runs the main processing** on all collections in the Train directory

5. **Saves results** to the test_output directory

## Expected Output

After running the Docker container, you should see:

```
=== Docker Setup Test ===
Testing imports...
✓ Basic imports successful
✓ PyMuPDF (fitz) import successful
✓ pytesseract import successful
✓ PIL (Pillow) import successful
✓ scikit-learn import successful
✓ sentence-transformers import successful
✓ PyTorch import successful
✓ joblib import successful
✓ 1A text_block_extractor import successful
✓ Tesseract version: 4.1.1
✅ All imports successful! Docker setup is working correctly.
🎉 All tests passed! Your Docker environment is ready.

Processing /app/Train/Collection 1...
Processing /app/Train/Collection 2...
Processing /app/Train/Collection 3...
All collections processed.
```

And find these files in the `test_output/` directory:
- `collection_1_output.json`
- `collection_2_output.json`
- `collection_3_output.json`

## Troubleshooting

### Common Issues

1. **"Docker command not found"**
   - Install Docker Desktop and restart your terminal

2. **"Permission denied" on Linux/macOS**
   - Run: `chmod +x run_docker.sh`

3. **"Volume mount failed"**
   - Ensure you're running from the correct directory (1B/)
   - Check that the directories exist

4. **"Build failed"**
   - Check your internet connection (needed to download dependencies)
   - Ensure Docker has enough disk space

5. **"pytesseract module not found"**
   - The Dockerfile now includes tesseract-ocr system dependencies
   - Rebuild the Docker image: `docker build -t challenge1b .`

6. **"Model download failed"**
   - The first run downloads the sentence transformer model (~90MB)
   - Ensure stable internet connection

### Debugging

1. **Check Docker is running:**
```bash
docker --version
```

2. **List Docker images:**
```bash
docker images
```

3. **Run container interactively:**
```bash
docker run -it --rm challenge1b /bin/bash
```

4. **Test dependencies manually:**
```bash
docker run --rm challenge1b python test_docker_setup.py
```

5. **Check container logs:**
```bash
docker logs <container_id>
```

## Performance Notes

- **First run**: Takes longer due to model download (~90MB) and tesseract installation
- **Subsequent runs**: Faster as model is cached
- **Memory usage**: ~2GB RAM recommended
- **Processing time**: <60 seconds for 3-5 documents
- **CPU only**: No GPU required

## Environment Variables

The Docker container uses these environment variables for deterministic behavior:

- `PYTHONHASHSEED=42` - Ensures consistent hash behavior
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` - Optimizes CPU operations
- `CUDA_VISIBLE_DEVICES=""` - Forces CPU-only processing
- `PYTHONUNBUFFERED=1` - Ensures immediate output display

## Dependencies

The Docker container includes:

### Python Dependencies:
- PyMuPDF==1.23.8 (PDF processing)
- pytesseract==0.3.10 (OCR functionality)
- Pillow==10.0.0 (Image processing)
- scikit-learn==1.3.0 (Machine learning)
- sentence-transformers==2.2.2 (Semantic analysis)
- torch==2.0.1 (Deep learning)
- numpy==1.24.3 (Numerical computing)
- joblib==1.3.2 (Model persistence)

### System Dependencies:
- tesseract-ocr (OCR engine)
- tesseract-ocr-eng (English language pack)
- libtesseract-dev (Development libraries)
- gcc/g++ (Compilation tools)

## Cleanup

To clean up Docker resources:

```bash
# Remove the image
docker rmi challenge1b

# Remove unused containers and images
docker system prune -f
``` 