# Challenge 1B Approach Explanation

## Overview
This implementation addresses Challenge 1B by creating an intelligent document analysis system that extracts and ranks sections from PDF documents based on persona and job requirements. The system uses a combination of heuristic-based heading extraction and semantic similarity ranking to identify the most relevant content.

## Methodology

### 1. Heading Extraction (Structural Analysis)
The system employs a heuristic-based approach to identify document headings using multiple criteria:
- **Font Size Analysis**: Relative font size compared to page median (>1.5x for H1, >1.2x for H2/H3)
- **Text Formatting**: All-caps text, title case, and numbered lists
- **Positional Cues**: Text ending with colons, alignment patterns
- **Content Length**: Reasonable heading length (3-100 characters)

This approach leverages the structural information from 1A's methodology but adapts it for the specific requirements of 1B, focusing on identifying meaningful section boundaries.

### 2. Content Extraction and Segmentation
For each identified heading, the system:
- Extracts all content from the heading's page to the next heading's page
- Maintains page number metadata for accurate referencing
- Preserves document structure while creating searchable text chunks

### 3. Semantic Relevance Ranking
The core innovation lies in the semantic ranking system:
- **Query Formation**: Combines persona and job requirements into a single query
- **Embedding Generation**: Uses the lightweight `all-MiniLM-L6-v2` model (<100MB) to create embeddings
- **Similarity Calculation**: Computes cosine similarity between query and section embeddings
- **Ranking**: Orders sections by relevance score to identify top 5 most important sections

### 4. Subsection Analysis
For the top-ranked sections, the system performs granular analysis:
- **Paragraph Segmentation**: Splits content into meaningful paragraphs
- **Relevance Scoring**: Uses keyword density analysis for travel-related terms
- **Content Refinement**: Selects and combines the most relevant paragraphs
- **Length Optimization**: Ensures refined text stays within reasonable limits (500 chars)

## Technical Implementation

### Model Selection
- **Sentence Transformer**: `all-MiniLM-L6-v2` (384 dimensions, ~90MB)
- **CPU-Only Processing**: Ensures compatibility with deployment constraints
- **Lightweight Architecture**: Meets the 1GB model size requirement

### Performance Optimizations
- **Batch Processing**: Processes multiple documents efficiently
- **Content Truncation**: Limits embedding input to 1000 characters per section
- **Early Filtering**: Removes irrelevant content before embedding generation
- **Memory Management**: Processes documents sequentially to minimize memory usage

### Quality Assurance
- **Duplicate Detection**: Removes similar headings to avoid redundancy
- **Content Validation**: Ensures extracted sections contain meaningful information
- **Error Handling**: Graceful degradation when PDFs are corrupted or inaccessible

## Key Advantages

1. **Semantic Understanding**: Goes beyond keyword matching to understand context and intent
2. **Scalable Architecture**: Can handle varying document types and sizes
3. **Resource Efficient**: Meets all technical constraints (CPU-only, <1GB, <60s)
4. **Robust Processing**: Handles different PDF formats and structures
5. **Contextual Ranking**: Considers both persona and job requirements in ranking

## Expected Performance
- **Processing Time**: <60 seconds for 3-5 documents
- **Model Size**: ~90MB (well under 1GB limit)
- **Accuracy**: High relevance scores for travel planning scenarios
- **Scalability**: Linear scaling with document count

This approach successfully bridges the gap between structural document analysis (1A) and semantic content understanding (1B), providing a comprehensive solution for intelligent document processing. 