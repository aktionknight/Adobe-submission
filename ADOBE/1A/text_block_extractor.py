import fitz
import os
import json
import numpy as np

def extract_text_blocks(pdf_path, save_to_output=False):
    """
    Extracts text blocks and their features from all pages of a PDF using PyMuPDF.
    Returns a list of dicts: {page, text, font_size, bbox, font, flags, origin, dist_from_top, font_size_rank, rel_font_size, is_all_caps, is_title_case, line_gap_before, ends_with_colon, y_pct, word_count, starts_with_numbering, alignment, font_is_unique}
    Text block JSON output is disabled to prevent generation of text block files.
    """
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        page_blocks = []
        prev_y = None
        font_counter = {}
        for b in page.get_text("dict")['blocks']:
            if 'lines' not in b:
                continue
            for line in b['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    y = span['bbox'][1]
                    line_gap_before = 0.0
                    if prev_y is not None:
                        line_gap_before = round(y - prev_y, 2)
                    prev_y = y
                    font = span['font']
                    font_counter[font] = font_counter.get(font, 0) + 1
                    block = {
                        'page': page_num,
                        'text': text,
                        'font_size': round(span['size'], 1),
                        'bbox': span['bbox'],
                        'font': span['font'],
                        'flags': span['flags'],
                        'origin': (span['bbox'][0], span['bbox'][1]),
                        'dist_from_top': span['bbox'][1] / page_height if page_height else 0.0,
                        'line_gap_before': line_gap_before,
                        'is_all_caps': int(text.isupper()),
                        'is_title_case': int(text.istitle()),
                        'ends_with_colon': int(text.endswith(':')),
                        'y_pct': span['bbox'][1] / page_height if page_height else 0.0,
                        'word_count': len(text.split()),
                        'starts_with_numbering': int(bool(text.split() and (text.split()[0].rstrip('.').isdigit() or text.split()[0][:-1].isdigit()))),
                    }
                    # Add is_bold feature
                    font_name = span.get('font', '')
                    flags = span.get('flags', 0)
                    is_bold = False
                    if isinstance(font_name, str) and ('Bold' in font_name or 'bold' in font_name):
                        is_bold = True
                    # PDFMiner/fitz: flag 2 means bold (see docs)
                    if isinstance(flags, int) and (flags & 2):
                        is_bold = True
                    block['is_bold'] = is_bold
                    
                    # Add line_gap_before feature
                    if page_blocks:
                        prev_block = page_blocks[-1]
                        prev_y = prev_block['bbox'][3]  # bottom of previous block
                        curr_y = span['bbox'][1]  # top of current block
                        line_gap = curr_y - prev_y
                        block['line_gap_before'] = max(0, line_gap)
                    else:
                        block['line_gap_before'] = 0.0
                    
                    # Add is_title_case feature
                    text_words = text.split()
                    is_title_case = False
                    if text_words:
                        # Check if first word starts with uppercase and others are lowercase
                        if (text_words[0][0].isupper() if text_words[0] else False):
                            is_title_case = all(word[0].islower() for word in text_words[1:] if word)
                    block['is_title_case'] = is_title_case
                    
                    # Add alignment feature (0=left, 1=center, 2=right)
                    x0, x1 = span['bbox'][0], span['bbox'][2]
                    page_width = page_width if page_width else 612  # default A4 width
                    center_x = page_width / 2
                    text_width = x1 - x0
                    text_center = x0 + text_width / 2
                    
                    if abs(text_center - center_x) < 50:  # within 50 points of center
                        alignment = 1  # center
                    elif x0 < page_width * 0.3:  # left third
                        alignment = 0  # left
                    else:
                        alignment = 2  # right
                    block['alignment'] = alignment
                    
                    page_blocks.append(block)
        # font_size_rank and rel_font_size for this page
        font_sizes = [b['font_size'] for b in page_blocks]
        median_font_size = float(np.median(font_sizes)) if font_sizes else 1.0
        sorted_blocks = sorted(page_blocks, key=lambda b: -b['font_size'])
        for i, b in enumerate(sorted_blocks):
            b['font_size_rank'] = (i+1) / len(page_blocks) if page_blocks else 1.0
            b['rel_font_size'] = b['font_size'] / median_font_size if median_font_size else 1.0
        # font_is_unique for this page
        for b in page_blocks:
            b['font_is_unique'] = int(font_counter.get(b['font'], 0) == 1)
        blocks.extend(page_blocks)
    
    # Add document-level features after all blocks are extracted
    total_blocks = len(blocks)
    page_heading_counts = {}
    doc_heading_count = 0
    
    for i, block in enumerate(blocks):
        # Document position rank (0-1, where 0 is first block in document)
        block['doc_position_rank'] = i / total_blocks if total_blocks > 0 else 0
        
        # Page heading count (how many potential headings on this page)
        page = block.get('page', 0)
        if page not in page_heading_counts:
            page_heading_counts[page] = 0
        page_heading_counts[page] += 1
        block['page_heading_count'] = page_heading_counts[page]
        
        # Document heading count (total potential headings so far)
        doc_heading_count += 1
        block['doc_heading_count'] = doc_heading_count
    
    # Second pass to add contextual features
    for i, block in enumerate(blocks):
        page = block.get('page', 0)
        
        # Is this the first potential heading on this page?
        is_first_on_page = True
        for j in range(i):
            if blocks[j].get('page', 0) == page:
                is_first_on_page = False
                break
        block['is_first_heading_on_page'] = is_first_on_page
        
        # Heading density (how many potential headings in nearby blocks)
        nearby_heading_count = 0
        for j in range(max(0, i-5), min(len(blocks), i+6)):
            if j != i and blocks[j].get('page', 0) == page:
                nearby_heading_count += 1
        block['heading_density'] = nearby_heading_count
    
    # Remove per-PDF normalization. Compute font_size_rank and rel_font_size globally as before.
    # (No per-PDF normalization block)
    
    if save_to_output:
        test_dir = os.path.join(os.path.dirname(__file__), 'test')
        os.makedirs(test_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(test_dir, f'{base}_blocks.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)
    
    return blocks

if __name__ == '__main__':
    import glob
    input_dir = 'input'
    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
    for pdf_path in pdf_files:
        print(f"Extracting blocks from {pdf_path}...")
        extract_text_blocks(pdf_path, save_to_output=False)
    print(f"Extraction complete. Text block JSON output disabled.") 