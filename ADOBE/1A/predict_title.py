import fitz
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import pytesseract
from PIL import Image
import numpy as np
from text_block_extractor import extract_text_blocks

def predict_title_for_pdf(pdf_path):
    blocks = extract_text_blocks(pdf_path)
    page0_blocks = [b for b in blocks if b.get('page', 0) == 0]
    if not page0_blocks:
        return ""
    
    font_sizes = [b.get('font_size', 0) for b in page0_blocks]
    if not font_sizes:
        return ""
    
    max_font = max(font_sizes)
    candidates = [b for b in page0_blocks if abs(b.get('font_size', 0) - max_font) <= 2.0]
    
    # If no candidates found with strict criteria, relax the criteria
    if not candidates:
        candidates = [b for b in page0_blocks if abs(b.get('font_size', 0) - max_font) <= 5.0]
    
    # If still no candidates, take top 5 blocks by font size
    if not candidates:
        candidates = sorted(page0_blocks, key=lambda b: b.get('font_size', 0), reverse=True)[:5]
    
    # Check for overlap among candidates
    ocr_triggered = False
    if len(candidates) >= 2:
        for i, b1 in enumerate(candidates):
            y1 = b1.get('origin', [0,0])[1]
            x1_0, x1_1 = b1.get('bbox', [0,0,0,0])[0], b1.get('bbox', [0,0,0,0])[2]
            for j, b2 in enumerate(candidates):
                if i == j:
                    continue
                y2 = b2.get('origin', [0,0])[1]
                x2_0, x2_1 = b2.get('bbox', [0,0,0,0])[0], b2.get('bbox', [0,0,0,0])[2]
                if abs(y1 - y2) <= 1.0:
                    x_overlap = min(x1_1, x2_1) - max(x1_0, x2_0)
                    width1 = x1_1 - x1_0
                    width2 = x2_1 - x2_0
                    min_width = min(width1, width2)
                    if x_overlap > 0.5 * min_width:
                        ocr_triggered = True
                        break
            if ocr_triggered:
                break
    if ocr_triggered and candidates:
        # OCR on union of candidate bboxes
        x0 = min(b['bbox'][0] for b in candidates)
        y0 = min(b['bbox'][1] for b in candidates)
        x1 = max(b['bbox'][2] for b in candidates)
        y1 = max(b['bbox'][3] for b in candidates)
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, config='--psm 6').strip()
            if ocr_text:
                # Apply the same grouping/word logic to OCR lines
                lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
                words = []
                for line in lines:
                    for w in line.split():
                        if len(words) < 25:
                            words.append(w)
                        else:
                            break
                    if len(words) >= 25:
                        break
                return ' '.join(words)
        except Exception as e:
            print(f"[DEBUG] OCR error for {pdf_path}: {e}")
            pass
    # Non-OCR logic (provided)
    if candidates:
        # Sort candidates by vertical position (top to bottom)
        candidates = sorted(candidates, key=lambda b: b.get('origin', [0,0])[1])
        
        # Group consecutive blocks that are part of the same title
        title_blocks = []
        for i, candidate in enumerate(candidates):
            if i == 0:
                title_blocks.append(candidate)
                continue
            
            prev_block = candidates[i-1]
            
            # Check if this block is part of the same title
            same_font_size = abs(candidate.get('font_size', 0) - prev_block.get('font_size', 0)) <= 1.0
            same_bold = candidate.get('is_bold', False) == prev_block.get('is_bold', False)
            same_alignment = candidate.get('alignment', 0) == prev_block.get('alignment', 0)
            
            # Check if blocks are close vertically (within reasonable gap)
            y_gap = candidate.get('origin', [0,0])[1] - prev_block.get('origin', [0,0])[1]
            close_vertically = y_gap <= 50  # Within 50 points
            
            # Check if both blocks are in top portion of page
            in_top_portion = (candidate.get('dist_from_top', 1.0) < 0.5 and 
                             prev_block.get('dist_from_top', 1.0) < 0.5)
            
            if (same_font_size and same_bold and same_alignment and 
                close_vertically and in_top_portion):
                title_blocks.append(candidate)
            else:
                break  # Stop at first non-matching block
        
        # Fallback: if we only have one block, look for other blocks with same properties nearby
        if len(title_blocks) == 1 and len(candidates) > 1:
            first_block = title_blocks[0]
            for candidate in candidates[1:]:
                same_font_size = abs(candidate.get('font_size', 0) - first_block.get('font_size', 0)) <= 1.0
                same_bold = candidate.get('is_bold', False) == first_block.get('is_bold', False)
                same_alignment = candidate.get('alignment', 0) == first_block.get('alignment', 0)
                y_gap = candidate.get('origin', [0,0])[1] - first_block.get('origin', [0,0])[1]
                close_vertically = y_gap <= 100  # More permissive for fallback
                
                if same_font_size and same_bold and same_alignment and close_vertically:
                    title_blocks.append(candidate)
                    break
        
        # Concatenate all title blocks
        title_texts = []
        for block in title_blocks:
            block_text = block.get('text', '').strip()
            if block_text:
                # Split by newlines and add all non-empty lines
                lines = [line.strip() for line in block_text.split('\n') if line.strip()]
                title_texts.extend(lines)
        
        return ' '.join(title_texts)
    
    return "" 