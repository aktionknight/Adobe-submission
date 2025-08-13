# Challenge 1B - Document Section Analysis

This implementation provides an intelligent document analysis system that extracts and ranks sections from PDF documents based on persona and job requirements.

## Overview

The system processes PDF documents to:
1. Extract document sections using heading detection
2. Rank sections by relevance to persona and job requirements
3. Generate refined subsection analysis
4. Output results in the required JSON format

## Requirements

- Python 3.9+
- CPU-only processing (no GPU required)
- Model size: <1GB
- Processing time: <60 seconds for 3-5 documents

## Installation

### Option 1: Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py <input_json_path> <output_json_path> <pdfs_dir>
```

### Option 2: Docker Installation

#### Prerequisites

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
   ├── main.py
   ├── requirements.txt
   ├── Train/
   │   ├── Collection 1/
   │   ├── Collection 2/
   │   └── Collection 3/
   └── test_output/
   ```

#### Quick Start (Recommended)

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

#### Manual Docker Commands

1. **Navigate to the project directory:**
```bash
cd 1B
```

2. **Build the Docker image:**
```bash
docker build -t challenge1b .
```

3. **Run the container:**
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

#### Using Docker Compose

```bash
cd 1B
docker-compose up --build
```

#### Windows PowerShell

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

#### What the Docker Container Does

1. **Builds the environment** with Python 3.9 and all required dependencies
2. **Mounts volumes** to access your local files:
   - `Train/` - Contains the training collections
   - `test_output/` - Where results are saved
   - `input/` - Additional input files
   - `output/` - Additional output files
   - `PDFs/` - PDF files directory
3. **Runs the main processing** on all collections in the Train directory
4. **Saves results** to the test_output directory

#### Expected Output

After running the Docker container, you should see:

```
Processing /app/Train/Collection 1...
Processing /app/Train/Collection 2...
Processing /app/Train/Collection 3...
All collections processed.
```

And find these files in the `test_output/` directory:
- `collection_1_output.json`
- `collection_2_output.json`
- `collection_3_output.json`

#### Troubleshooting

**Common Issues:**

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

5. **"Model download failed"**
   - The first run downloads the sentence transformer model (~90MB)
   - Ensure stable internet connection

**Debugging:**

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

4. **Check container logs:**
```bash
docker logs <container_id>
```

#### Performance Notes

- **First run**: Takes longer due to model download (~90MB)
- **Subsequent runs**: Faster as model is cached
- **Memory usage**: ~2GB RAM recommended
- **Processing time**: <60 seconds for 3-5 documents
- **CPU only**: No GPU required

#### Environment Variables

The Docker container uses these environment variables for deterministic behavior:

- `PYTHONHASHSEED=42` - Ensures consistent hash behavior
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` - Optimizes CPU operations
- `CUDA_VISIBLE_DEVICES=""` - Forces CPU-only processing
- `PYTHONUNBUFFERED=1` - Ensures immediate output display

#### Cleanup

To clean up Docker resources:

```bash
# Remove the image
docker rmi challenge1b

# Remove unused containers and images
docker system prune -f
```

## Usage

### Command Line Interface

```bash
python main.py <input_json_path> <output_json_path> <pdfs_dir>
```

**Parameters:**
- `input_json_path`: Path to the input JSON file (challenge1b_input.json format)
- `output_json_path`: Path where the output JSON will be saved
- `pdfs_dir`: Directory containing the PDF files referenced in the input

### Example Usage

```bash
# Process Collection 1
python main.py Train/Collection\ 1/challenge1b_input.json output/collection1_output.json Train/Collection\ 1/PDFs/

# Process Collection 2
python main.py Train/Collection\ 2/challenge1b_input.json output/collection2_output.json Train/Collection\ 2/PDFs/

# Process Collection 3
python main.py Train/Collection\ 3/challenge1b_input.json output/collection3_output.json Train/Collection\ 3/PDFs/
```

## Input Format

The input JSON should follow this structure:

```json
{
    "challenge_info": {
        "challenge_id": "round_1b_002",
        "test_case_name": "travel_planner",
        "description": "France Travel"
    },
    "documents": [
        {
            "filename": "document1.pdf",
            "title": "Document 1 Title"
        }
    ],
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a trip of 4 days for a group of 10 college friends."
    }
}
```

## Output Format

The system generates output in the following JSON format:

```json
{
    "metadata": {
        "input_documents": ["document1.pdf", "document2.pdf"],
        "persona": "Travel Planner",
        "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
        "processing_timestamp": "2025-01-27T10:30:00.000000"
    },
    "extracted_sections": [
        {
            "document": "document1.pdf",
            "page_number": 1,
            "section_title": "Introduction",
            "importance_rank": 1
        }
    ],
    "subsection_analysis": [
        {
            "document": "document1.pdf",
            "refined_text": "Extracted and refined content...",
            "page_number": 1
        }
    ]
}
```

## Approach Explained

### Overview
This implementation creates an intelligent document analysis system that extracts and ranks sections from PDF documents based on persona and job requirements. The system uses a hybrid approach combining structural analysis (from Challenge 1A) with semantic similarity ranking to identify the most relevant content.

### Methodology

#### 1. Heading Extraction (Structural Analysis)
The system employs a heuristic-based approach to identify document headings using multiple criteria:
- **Font Size Analysis**: Relative font size compared to page median (>1.5x for H1, >1.2x for H2/H3)
- **Text Formatting**: All-caps text, title case, and numbered lists
- **Positional Cues**: Text ending with colons, alignment patterns
- **Content Length**: Reasonable heading length (3-100 characters)

This approach leverages the structural information from 1A's methodology but adapts it for the specific requirements of 1B, focusing on identifying meaningful section boundaries.

#### 2. Content Extraction and Segmentation
For each identified heading, the system:
- Extracts all content from the heading's page to the next heading's page
- Maintains page number metadata for accurate referencing
- Preserves document structure while creating searchable text chunks

#### 3. Semantic Relevance Ranking
The core innovation lies in the semantic ranking system:
- **Query Formation**: Combines persona and job requirements into a single query
- **Embedding Generation**: Uses the lightweight `all-MiniLM-L6-v2` model (<100MB) to create embeddings
- **Similarity Calculation**: Computes cosine similarity between query and section embeddings
- **Ranking**: Orders sections by relevance score to identify top 5 most important sections

#### 4. Subsection Analysis
For the top-ranked sections, the system performs granular analysis:
- **Paragraph Segmentation**: Splits content into meaningful paragraphs
- **Relevance Scoring**: Uses keyword density analysis for travel-related terms
- **Content Refinement**: Selects and combines the most relevant paragraphs
- **Length Optimization**: Ensures refined text stays within reasonable limits (500 chars)

### Key Advantages

1. **Semantic Understanding**: Goes beyond keyword matching to understand context and intent
2. **Scalable Architecture**: Can handle varying document types and sizes
3. **Resource Efficient**: Meets all technical constraints (CPU-only, <1GB, <60s)
4. **Robust Processing**: Handles different PDF formats and structures
5. **Contextual Ranking**: Considers both persona and job requirements in ranking

## Libraries & Models Used

### Core Libraries

- **PyMuPDF (1.23.8)**: PDF text extraction and document processing
- **scikit-learn (1.3.0)**: Machine learning utilities and cosine similarity calculations
- **numpy (1.24.3)**: Numerical computations and array operations
- **sentence-transformers (2.2.2)**: Semantic text embeddings and similarity
- **torch (2.0.1)**: Deep learning framework for transformer models
- **joblib (1.3.2)**: Model serialization and parallel processing

### Models

#### Primary Model: all-MiniLM-L6-v2
- **Type**: Sentence Transformer
- **Size**: ~90MB (well under 1GB limit)
- **Dimensions**: 384-dimensional embeddings
- **Purpose**: Semantic similarity calculation between queries and document sections
- **Performance**: Optimized for CPU-only processing
- **License**: Apache 2.0

#### Model Selection Rationale
- **Lightweight**: Meets the 1GB model size constraint
- **Fast**: Optimized for <60 second processing time
- **Accurate**: State-of-the-art performance for semantic similarity tasks
- **Compatible**: Works well with CPU-only processing requirements

### Technical Implementation

#### Performance Optimizations
- **Batch Processing**: Processes multiple documents efficiently
- **Content Truncation**: Limits embedding input to 1000 characters per section
- **Early Filtering**: Removes irrelevant content before embedding generation
- **Memory Management**: Processes documents sequentially to minimize memory usage

#### Quality Assurance
- **Duplicate Detection**: Removes similar headings to avoid redundancy
- **Content Validation**: Ensures extracted sections contain meaningful information
- **Error Handling**: Graceful degradation when PDFs are corrupted or inaccessible

## Architecture

### Core Components

1. **Heading Extractor** (`heading_extractor.py`)
   - Identifies document headings using heuristic rules
   - Extracts section content based on heading boundaries

2. **Section Ranker** (`section_ranker.py`)
   - Uses sentence transformers for semantic similarity
   - Ranks sections by relevance to persona and job requirements
   - Extracts refined subsection content

3. **Text Block Extractor** (`text_block_extractor.py`)
   - Extracts text blocks with formatting information
   - Provides features for heading detection

4. **Main Processor** (`main.py`)
   - Orchestrates the entire pipeline
   - Handles input/output and error management

### Model Details

- **Sentence Transformer**: `all-MiniLM-L6-v2`
- **Model Size**: ~90MB (well under 1GB limit)
- **Processing**: CPU-only for compatibility
- **Performance**: Optimized for <60 second processing time
- **Deterministic**: Configured for reproducible results across different devices

### Reproducibility

The system is configured for deterministic behavior to ensure consistent results across different devices:

- **PyTorch Deterministic**: `torch.set_deterministic(True)` enabled
- **Fixed Seeds**: Random seeds set to 42 for NumPy and PyTorch
- **Environment Variables**: `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- **Model Deterministic**: Sentence transformer model configured for deterministic inference
- **Random Forest Models**: Pre-trained and frozen with fixed random_state=42
- **Test Scripts**: 
  - Use `python test_deterministic.py` to verify model consistency
  - Use `python test_reproducibility.py` to verify full pipeline consistency

#### Deterministic Components:

1. **Random Forest Models (1A)**:
   - Stage 1: Heading vs Non-Heading classification
   - Stage 2: H1/H2/H3/H4 level classification
   - Both models trained with `random_state=42`
   - Models are pre-trained and frozen (no retraining)

2. **Sentence Transformer Model (1B)**:
   - Model: `all-MiniLM-L6-v2`
   - Deterministic encoding with `torch.no_grad()`
   - Fixed precision (float32) to avoid floating point variations
   - CPU-only processing to ensure consistency

3. **Text Processing**:
   - Deterministic text extraction from PDFs
   - Consistent feature extraction for heading detection
   - Fixed sorting and ranking algorithms

#### Cross-Device Compatibility:

The system will produce **identical results** across different devices because:
- All random operations use fixed seeds
- Models are pre-trained and frozen
- No GPU-specific optimizations that could vary
- Floating point operations are deterministic
- Text processing is deterministic

## Testing

### Test with Sample Data

1. Navigate to a collection directory:
```bash
cd Train/Collection\ 1/
```

2. Run the processor:
```bash
python ../../main.py challenge1b_input.json ../../test_output/collection1_result.json PDFs/
```

3. Verify the output:
```bash
cat ../../test_output/collection1_result.json
```

### Validation

The output should contain:
- 5 ranked sections in `extracted_sections`
- Multiple refined text chunks in `subsection_analysis`
- Proper metadata with timestamp
- Valid JSON structure

## Performance Metrics

- **Section Relevance**: 60 points - How well selected sections match persona + job requirements
- **Sub-Section Relevance**: 40 points - Quality of granular subsection extraction and ranking

## Troubleshooting

### Common Issues

1. **PDF not found**: Ensure PDF files are in the correct directory
2. **Memory issues**: The system is optimized for CPU-only processing
3. **Model download**: First run may download the sentence transformer model (~90MB)

### Error Handling

The system includes robust error handling for:
- Missing PDF files
- Corrupted documents
- Invalid JSON input
- Memory constraints

## License

This implementation is designed for the Adobe Challenge 1B requirements. 