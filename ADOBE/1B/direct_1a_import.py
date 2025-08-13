import sys
import os
import fitz

# Add 1A directory to path to import its modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../1A')))

from predict_title import predict_title_for_pdf
from predict_headings import predict_headings_for_pdf
from text_block_extractor import extract_text_blocks  # Always use 1A version

# Use 1B/output for model paths, but all logic and feature extraction from 1A
import joblib
STAGE1_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'output', 'rf_stage1_heading_model.joblib')
STAGE2_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'output', 'rf_stage2_level_model.joblib')

def extract_1a_data(pdf_path):
    """
    Extract both title and headings using the exact 1A predict_headings and predict_title logic.
    """
    print(f"  Extracting title and headings using 1A imports...")
    try:
        title = predict_title_for_pdf(pdf_path)
        clf1 = joblib.load(STAGE1_MODEL_PATH)
        clf2 = joblib.load(STAGE2_MODEL_PATH)
        outline = predict_headings_for_pdf(pdf_path, clf1, clf2)
        # Convert 0-indexed pages to 1-indexed pages for 1B format
        for heading in outline:
            heading['page'] = heading['page'] + 1
        return title, outline
    except Exception as e:
        print(f"  1A ML approach failed: {str(e)}, falling back to enhanced heuristic approach")
        title = predict_title_for_pdf(pdf_path)
        outline = extract_headings_enhanced_heuristic(pdf_path)
        for heading in outline:
            heading['page'] = heading['page'] + 1
        return title, outline

def extract_headings_enhanced_heuristic(pdf_path):
    """
    Enhanced heuristic approach that matches the 1A output format more closely.
    """
    from text_block_extractor import extract_text_blocks
    blocks = extract_text_blocks(pdf_path, save_to_output=False)
    
    # Filter potential headings based on enhanced heuristics
    potential_headings = []
    for block in blocks:
        text = block.get('text', '').strip()
        if not text or len(text) < 2:
            continue
            
        # Enhanced heuristic rules for identifying headings
        is_heading = False
        level = 'H1'  # Default level
        
        # Rule 1: Large font size (relative to page median)
        if block.get('rel_font_size', 1.0) > 1.2:
            is_heading = True
            if block.get('rel_font_size', 1.0) > 1.6:
                level = 'H1'
            else:
                level = 'H2'
        
        # Rule 2: All caps text
        elif block.get('is_all_caps', 0) and len(text) > 2:
            is_heading = True
            level = 'H2'
        
        # Rule 3: Title case with reasonable length
        elif (block.get('is_title_case', 0) and 
              2 < len(text) < 200 and 
              block.get('rel_font_size', 1.0) > 1.0):
            is_heading = True
            level = 'H3'
        
        # Rule 4: Ends with colon (common in headings)
        elif text.endswith(':') and block.get('rel_font_size', 1.0) > 0.9:
            is_heading = True
            level = 'H3'
        
        # Rule 5: Starts with numbering
        elif block.get('starts_with_numbering', 0) and block.get('rel_font_size', 1.0) > 0.9:
            is_heading = True
            level = 'H3'
        
        # Rule 6: Bold text (flags indicate bold)
        elif (block.get('flags', 0) & 16) and len(text) > 2 and block.get('rel_font_size', 1.0) > 0.9:
            is_heading = True
            level = 'H3'
        
        # Rule 7: Short text with high font size (likely headings)
        elif len(text) < 100 and block.get('rel_font_size', 1.0) > 1.1:
            is_heading = True
            level = 'H3'
        
        # Rule 8: Text that looks like a section title
        elif (len(text) < 150 and 
              (text.isupper() or text.istitle()) and 
              block.get('rel_font_size', 1.0) > 1.0):
            is_heading = True
            level = 'H3'
        
        if is_heading:
            potential_headings.append({
                'level': level,
                'text': text,
                'page': block['page']
            })
    
    # Improved filter: allow colons only if heading is not a paragraph or bullet point
    def is_heading_like(text):
        t = text.strip()
        if t and t[0].islower():
            return False
        if len(t.split()) > 12:
            return False
        if t.islower():
            return False
        if t.startswith(('-', '*', '•')) or (t.split() and t.split()[0].rstrip('.').isdigit()):
            return False
        if any(punct in t for punct in ['.', '!', '?']):
            return False
        return True
    filtered_headings = [h for h in potential_headings if ';' not in h['text'] and (':' not in h['text'] or is_heading_like(h['text']))]
    # Sort by page number and position
    filtered_headings.sort(key=lambda h: (h['page'], h.get('text', '')))
    
    # Remove duplicates and very similar headings
    unique_headings = []
    seen_texts = set()
    
    for heading in filtered_headings:
        # Normalize text for comparison
        normalized = heading['text'].lower().strip()
        if normalized not in seen_texts and len(normalized) > 1:
            seen_texts.add(normalized)
            unique_headings.append(heading)
    
    return unique_headings

def extract_section_content(pdf_path, headings):
    """
    Extract content under each heading from the PDF.
    Returns a list of sections with their content and page ranges.
    """
    doc = fitz.open(pdf_path)
    sections = []
    
    # More sophisticated generic filtering
    main_headings = []
    for heading in headings:
        text = heading['text'].strip()
        word_count = len(text.split())
        
        # More sophisticated criteria for main section headings:
        # 1. Reasonable length (2-6 words is ideal for main sections)
        # 2. No sentence-ending punctuation
        # 3. Not starting with common sub-section words
        # 4. Not all lowercase (likely a paragraph)
        # 5. Not starting with numbers, bullets, or symbols
        # 6. Not ending with common sub-section indicators
        # 7. Should look like a proper heading (title case or all caps)
        
        if (2 <= word_count <= 6 and  # Ideal length for main sections
            not any(punct in text for punct in ['.', '!', '?']) and  # No sentence punctuation
            not text.lower().startswith(('additional', 'extra', 'more', 'other', 'also', 'further', 'next', 'then', 'finally')) and  # Not sub-sections
            not text.islower() and  # Not all lowercase
            not text.startswith(('-', '*', '•', '1.', '2.', '3.', '4.', '5.')) and  # Not bullets or numbers
            not text.endswith((':', ';', ',', '...')) and  # Not incomplete
            not text.lower().endswith(('etc', 'etc.', 'and more', 'and others')) and  # Not incomplete
            (text.istitle() or text.isupper() or heading['level'] == 'H1')):  # Proper heading format
            main_headings.append(heading)
    
    # If we don't have enough, be more permissive but still maintain quality
    if len(main_headings) < 3:
        for heading in headings:
            text = heading['text'].strip()
            word_count = len(text.split())
            
            # More permissive but still quality criteria
            if (1 <= word_count <= 8 and  # Allow slightly longer headings
                not text.islower() and  # Still not all lowercase
                not text.startswith(('-', '*', '•')) and  # Still not bullet points
                not any(punct in text for punct in ['.', '!', '?']) and  # Still no sentence punctuation
                not text.lower().startswith(('additional', 'extra', 'more', 'other')) and  # Still not sub-sections
                (text.istitle() or text.isupper() or heading['level'] == 'H1')):  # Still proper heading format
                main_headings.append(heading)
    
    # Remove duplicates and sort by page
    unique_headings = []
    seen_texts = set()
    for heading in main_headings:
        normalized = heading['text'].lower().strip()
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            unique_headings.append(heading)
    
    # Sort by page number
    unique_headings.sort(key=lambda h: h['page'])
    
    for i, heading in enumerate(unique_headings):
        start_page = heading['page'] - 1  # Convert back to 0-indexed for PyMuPDF
        
        # Find the next main heading to determine end page
        end_page = len(doc)  # Default to end of document
        for j in range(i + 1, len(unique_headings)):
            next_heading = unique_headings[j]
            if next_heading['page'] > heading['page']:
                end_page = next_heading['page'] - 1
                break
        
        # Extract text from the page range
        section_text = ""
        for page_num in range(start_page, end_page):
            if page_num < len(doc):
                page = doc[page_num]
                section_text += page.get_text() + "\n"
        
        # Clean up the text
        section_text = section_text.strip()
        
        # Only add sections with meaningful content
        if len(section_text) > 50:  # Minimum content length
            sections.append({
                'title': heading['text'],  # Use the current heading as the title
                'level': heading['level'],
                'page': heading['page'],
                'content': section_text,
                'start_page': start_page,
                'end_page': end_page
            })
    
    return sections 