import os
import json
import joblib
import subprocess
import sys
from predict_title import predict_title_for_pdf
from predict_headings import predict_headings_for_pdf
from text_block_extractor import extract_text_blocks

# Always resolve directories relative to this file so it works from any CWD
BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_heading_model.joblib')


def main():
    # Ensure required directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Run text block extraction for all PDFs
    print('Extracting text blocks for all PDFs...')
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(INPUT_DIR, fname)
        extract_text_blocks(pdf_path, save_to_output=False)

    # 2. Run train_model.py to retrain the model (via absolute path and current interpreter)
    print('Training model...')
    train_script = os.path.join(BASE_DIR, 'train_model.py')
    subprocess.run([sys.executable, train_script], check=True)

    # 3. Run heading and title prediction for all PDFs
    print('Running predictions...')
    clf1 = None
    clf2 = None
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(INPUT_DIR, fname)
        base_name = fname[:-4]
        # Predict title
        title = predict_title_for_pdf(pdf_path)
        # Predict headings
        outline = predict_headings_for_pdf(pdf_path, clf1, clf2)
        # Save in train-style format
        out_path = os.path.join(OUTPUT_DIR, f'{base_name}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'title': title, 'outline': outline}, f, ensure_ascii=False, indent=2)
        print(f"{fname}: title='{title}', headings={len(outline)}")

if __name__ == '__main__':
    main() 
    