import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../1A')))
from direct_1a_import import extract_1a_data, extract_section_content

# Test on Collection 1
collection_path = "Train/Collection 1"
input_json_path = os.path.join(collection_path, 'challenge1b_input.json')

import json
with open(input_json_path, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

input_documents = [doc['filename'] for doc in input_data.get('documents', [])]
pdfs_dir = os.path.join(collection_path, 'PDFs')

print("All sections found across all PDFs:")
print("=" * 60)

all_sections = []
for filename in input_documents:
    pdf_path = os.path.join(pdfs_dir, filename)
    if not os.path.exists(pdf_path):
        continue
    title, headings = extract_1a_data(pdf_path)
    sections = extract_section_content(pdf_path, headings)
    print(f"\n{filename}:")
    for i, s in enumerate(sections):
        print(f"  {i+1}. '{s['title']}' (Page: {s['page']}, Content length: {len(s['content'])} chars)")
        all_sections.append({
            'section_title': s['title'],
            'section_text': s['content'],
            'document': filename,
            'page_number': s['page']
        })

print(f"\nTotal sections found: {len(all_sections)}") 