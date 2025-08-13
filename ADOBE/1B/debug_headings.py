import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../1A')))
from predict_headings import predict_headings_for_pdf
import joblib

# Load models
STAGE1_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'output', 'rf_stage1_heading_model.joblib')
STAGE2_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'output', 'rf_stage2_level_model.joblib')

clf1 = joblib.load(STAGE1_MODEL_PATH)
clf2 = joblib.load(STAGE2_MODEL_PATH)

# Test on one PDF
pdf_path = "Train/Collection 1/PDFs/South of France - Things to Do.pdf"
print(f"Testing headings extraction for: {pdf_path}")
print("=" * 50)

headings = predict_headings_for_pdf(pdf_path, clf1, clf2)
print(f"Found {len(headings)} headings:")
for i, h in enumerate(headings):
    print(f"{i+1}. '{h['text']}' (Level: {h['level']}, Page: {h['page']})") 