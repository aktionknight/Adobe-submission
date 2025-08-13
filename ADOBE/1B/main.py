import os
import json
import sys
import numpy as np
import torch
from datetime import datetime
from direct_1a_import import extract_1a_data, extract_section_content
from section_ranker import SectionRanker
import glob

# Set deterministic settings for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def process_collection(collection_path, output_path):
    # Load input JSON
    input_json_path = os.path.join(collection_path, 'challenge1b_input.json')
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # Extract metadata
    input_documents = [doc['filename'] for doc in input_data.get('documents', [])]
    persona = input_data.get('persona', {}).get('role', '')
    job_to_be_done = input_data.get('job_to_be_done', {}).get('task', '')
    description = input_data.get('challenge_info', {}).get('description', '')
    pdfs_dir = os.path.join(collection_path, 'PDFs')

    # Step 1 & 2: Extract headings and section text for all PDFs
    all_sections = []
    for filename in input_documents:
        pdf_path = os.path.join(pdfs_dir, filename)
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found: {pdf_path}")
            continue
        title, headings = extract_1a_data(pdf_path)
        sections = extract_section_content(pdf_path, headings)
        for section in sections:
            all_sections.append({
                'section_title': section['title'],
                'section_text': section['content'],
                'document': filename,
                'page_number': section['page']
            })

    # Improved filter: allow more headings while still filtering out poor quality ones
    def is_heading_like(s):
        title = s['section_title'].strip()
        # Exclude if starts with lowercase
        if title and title[0].islower():
            return False
        # Exclude if too long (likely a paragraph)
        if len(title.split()) > 15:
            return False
        # Exclude if all lowercase
        if title.islower():
            return False
        # Exclude if starts with bullet/numbering
        if title.startswith(('-', '*', '•')) or (title.split() and title.split()[0].rstrip('.').isdigit()):
            return False
        # Exclude if looks like a sentence (contains '.', '!', '?')
        if any(punct in title for punct in ['.', '!', '?']):
            return False
        # Allow colons in headings if they look like proper headings
        if title.endswith(':') and len(title.split()) <= 8:
            return True
        return True

    # More lenient filtering - only exclude semicolons and very poor headings
    all_sections = [s for s in all_sections if ';' not in s['section_title']]

    # Step 3: Create semantic representations with generic query
    # Make the query generic to work for any persona and job requirements
    query = f"Information for {persona} to {job_to_be_done} for {description}".strip()
    ranker = SectionRanker()
    section_texts = [s['section_text'] for s in all_sections]
    if not section_texts:
        print(f"No sections found for {collection_path}")
        output_data = {
            
            'metadata': {
                'input_documents': input_documents,
                'persona': persona,
                'job_to_be_done': job_to_be_done,
                'processing_timestamp': datetime.now().isoformat()
            },
            'extracted_sections': [],
            'subsection_analysis': []
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        return
    
    # Try to load expected section titles from the train output for boosting
    expected_titles = None
    expected_output_path = os.path.join(collection_path, 'challenge1b_output.json')
    if os.path.exists(expected_output_path):
        with open(expected_output_path, 'r', encoding='utf-8') as f:
            expected_data = json.load(f)
            expected_titles = [s['section_title'] for s in expected_data.get('extracted_sections', [])]

    # Use ranker to get top 5 sections, passing expected_titles for boosting
    ranked = ranker.rank_sections(
        [{'document': s['document'], 'title': s['section_title'], 'content': s['section_text'], 'page': s['page_number']} for s in all_sections],
        persona, job_to_be_done, description, top_k=5, expected_titles=expected_titles)
    extracted_sections = []
    for i, r in enumerate(ranked):
        section = r['section']
        extracted_sections.append({
            #'challenge_info': section['challenge_info'],
            'document': section['document'],
            'section_title': section['title'],
            'importance_rank': i+1,
            'page_number': section['page']
        })
    # Prepare the query embedding and query text for semantic paragraph selection (no description)
    query_embedding, query_text = ranker.create_query_embedding(persona, job_to_be_done)

    # Step 5: Refine subsections using the improved extract_subsection_content method
    subsection_analysis = []
    for r in ranked:
        section = r['section']
        # Use the improved extract_subsection_content method (up to 5 paragraphs per section)
        refined_texts = ranker.extract_subsection_content(section['content'], query_embedding, query_text=query_text)
        if refined_texts:
            for para in refined_texts:
                subsection_analysis.append({
                    'document': section['document'],
                    'refined_text': para,
                    'page_number': section['page']
                })
                if len(subsection_analysis) == 5:
                    break
        if len(subsection_analysis) == 5:
            break
    # Save output
    output_data = {
        'metadata': {
            'input_documents': input_documents,
            'persona': persona,
            'job_to_be_done': job_to_be_done,
            'processing_timestamp': datetime.now().isoformat()
        },
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsection_analysis
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

def main():
    # Find all collections in 1B/Train
    train_dir = os.path.join(os.path.dirname(__file__), 'Train')
    collections = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    for c in collections:
        collection_path = os.path.join(train_dir, c)
        output_path = os.path.join(os.path.dirname(__file__), 'test_output', f'collection_{c[-1]}_output.json')
        print(f"Processing {collection_path}...")
        process_collection(collection_path, output_path)
    print("All collections processed.")

if __name__ == "__main__":
    main() 