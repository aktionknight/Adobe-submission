import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../1A')))
from direct_1a_import import extract_1a_data, extract_section_content

# Test on one PDF
pdf_path = "Train/Collection 1/PDFs/South of France - Things to Do.pdf"
print(f"Testing section extraction for: {pdf_path}")
print("=" * 50)

title, headings = extract_1a_data(pdf_path)
print(f"Title: {title}")
print(f"Found {len(headings)} headings:")
for i, h in enumerate(headings):
    print(f"{i+1}. '{h['text']}' (Level: {h['level']}, Page: {h['page']})")

print("\n" + "=" * 50)
print("Extracting sections...")

sections = extract_section_content(pdf_path, headings)
print(f"Found {len(sections)} sections:")
for i, s in enumerate(sections):
    print(f"{i+1}. '{s['title']}' (Page: {s['page']}, Content length: {len(s['content'])} chars)")
    print(f"   Content preview: {s['content'][:100]}...")
    print() 