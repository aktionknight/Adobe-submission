#!/usr/bin/env python3
"""
Test script to verify Docker setup and dependencies
"""

def test_imports():
    """Test that all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import os
        import json
        import sys
        import numpy as np
        print("✓ Basic imports successful")
        
        # Test PyMuPDF
        import fitz
        print("✓ PyMuPDF (fitz) import successful")
        
        # Test pytesseract
        import pytesseract
        print("✓ pytesseract import successful")
        
        # Test PIL
        from PIL import Image
        print("✓ PIL (Pillow) import successful")
        
        # Test scikit-learn
        import sklearn
        print("✓ scikit-learn import successful")
        
        # Test sentence-transformers
        import sentence_transformers
        print("✓ sentence-transformers import successful")
        
        # Test torch
        import torch
        print("✓ PyTorch import successful")
        
        # Test joblib
        import joblib
        print("✓ joblib import successful")
        
        # Test 1A imports
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../1A')))
        from text_block_extractor import extract_text_blocks
        print("✓ 1A text_block_extractor import successful")
        
        print("\n✅ All imports successful! Docker setup is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_tesseract():
    """Test that tesseract is properly installed"""
    try:
        import pytesseract
        # Test tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
        return True
    except Exception as e:
        print(f"❌ Tesseract error: {e}")
        return False

if __name__ == "__main__":
    print("=== Docker Setup Test ===\n")
    
    imports_ok = test_imports()
    tesseract_ok = test_tesseract()
    
    if imports_ok and tesseract_ok:
        print("\n🎉 All tests passed! Your Docker environment is ready.")
    else:
        print("\n❌ Some tests failed. Please check the Docker setup.")
        sys.exit(1) 