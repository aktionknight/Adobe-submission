#!/usr/bin/env python3
"""
Comprehensive reproducibility test for Challenge 1B.
This test verifies that the entire pipeline produces identical results across multiple runs.
"""

import os
import json
import numpy as np
import torch
import tempfile
import shutil
from datetime import datetime

# Set deterministic settings
torch.set_deterministic(True)
torch.manual_seed(42)
np.random.seed(42)

def test_full_pipeline_reproducibility():
    """Test that the full pipeline produces identical results across multiple runs."""
    
    print("=== Challenge 1B Reproducibility Test ===")
    print("Testing full pipeline consistency...")
    print("=" * 60)
    
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    test_output_dir = os.path.join(temp_dir, "test_output")
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        # Import the main processing function
        from main import process_collection
        
        # Test data - use a simple collection
        test_collection_path = "Train/Collection 1"
        if not os.path.exists(test_collection_path):
            print(f"❌ Test collection not found: {test_collection_path}")
            return False
        
        # Run the pipeline multiple times
        results = []
        for run in range(3):
            print(f"\n--- Run {run + 1} ---")
            
            # Create unique output path for this run
            output_path = os.path.join(test_output_dir, f"run_{run}_output.json")
            
            # Process the collection
            process_collection(test_collection_path, output_path)
            
            # Load and store results
            with open(output_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # Extract key information for comparison
            run_result = {
                'extracted_sections': [
                    {
                        'document': section['document'],
                        'section_title': section['section_title'],
                        'importance_rank': section['importance_rank'],
                        'page_number': section['page_number']
                    }
                    for section in result_data.get('extracted_sections', [])
                ],
                'subsection_analysis': [
                    {
                        'document': subsection['document'],
                        'refined_text': subsection['refined_text'][:100],  # First 100 chars for comparison
                        'page_number': subsection['page_number']
                    }
                    for subsection in result_data.get('subsection_analysis', [])
                ]
            }
            
            results.append(run_result)
            
            print(f"  ✅ Run {run + 1} completed")
            print(f"  📊 Sections found: {len(run_result['extracted_sections'])}")
            print(f"  📊 Subsections found: {len(run_result['subsection_analysis'])}")
        
        # Compare results
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS:")
        print("=" * 60)
        
        all_consistent = True
        
        # Compare extracted sections
        print("\n📋 Extracted Sections Comparison:")
        for i in range(len(results[0]['extracted_sections'])):
            section_consistent = True
            base_section = results[0]['extracted_sections'][i]
            
            for run in range(1, len(results)):
                if run < len(results) and i < len(results[run]['extracted_sections']):
                    current_section = results[run]['extracted_sections'][i]
                    if (current_section['section_title'] != base_section['section_title'] or
                        current_section['importance_rank'] != base_section['importance_rank'] or
                        current_section['page_number'] != base_section['page_number']):
                        section_consistent = False
                        print(f"  ❌ Section {i+1} inconsistent across runs")
                        break
                else:
                    section_consistent = False
                    print(f"  ❌ Section {i+1} missing in run {run+1}")
                    break
            
            if section_consistent:
                print(f"  ✅ Section {i+1}: {base_section['section_title']}")
            else:
                all_consistent = False
        
        # Compare subsection analysis
        print("\n📝 Subsection Analysis Comparison:")
        for i in range(len(results[0]['subsection_analysis'])):
            subsection_consistent = True
            base_subsection = results[0]['subsection_analysis'][i]
            
            for run in range(1, len(results)):
                if run < len(results) and i < len(results[run]['subsection_analysis']):
                    current_subsection = results[run]['subsection_analysis'][i]
                    if (current_subsection['refined_text'] != base_subsection['refined_text'] or
                        current_subsection['page_number'] != base_subsection['page_number']):
                        subsection_consistent = False
                        print(f"  ❌ Subsection {i+1} inconsistent across runs")
                        break
                else:
                    subsection_consistent = False
                    print(f"  ❌ Subsection {i+1} missing in run {run+1}")
                    break
            
            if subsection_consistent:
                print(f"  ✅ Subsection {i+1}: {base_subsection['refined_text'][:50]}...")
            else:
                all_consistent = False
        
        # Final result
        print("\n" + "=" * 60)
        if all_consistent:
            print("🎉 SUCCESS: All runs produced identical results!")
            print("✅ The pipeline is deterministic and reproducible.")
        else:
            print("❌ FAILURE: Results are not consistent across runs!")
            print("⚠️  The pipeline may have non-deterministic elements.")
        
        return all_consistent
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_model_consistency():
    """Test that the sentence transformer model produces consistent embeddings."""
    
    print("\n=== Model Consistency Test ===")
    
    from section_ranker import SectionRanker
    
    # Test texts
    test_texts = [
        "Introduction to machine learning",
        "Advanced neural networks",
        "Natural language processing techniques"
    ]
    
    # Create multiple ranker instances
    rankers = []
    embeddings_sets = []
    
    for run in range(3):
        print(f"\n--- Model Run {run + 1} ---")
        
        # Create fresh ranker
        ranker = SectionRanker()
        rankers.append(ranker)
        
        # Get embeddings
        embeddings = []
        for text in test_texts:
            with torch.no_grad():
                embedding = ranker.model.encode([text], convert_to_tensor=True, convert_to_numpy=True)
            embeddings.append(embedding[0])
        
        embeddings_sets.append(embeddings)
        print(f"  ✅ Generated {len(embeddings)} embeddings")
    
    # Compare embeddings
    print("\n--- Embedding Comparison ---")
    embedding_consistent = True
    
    for i, text in enumerate(test_texts):
        base_embedding = embeddings_sets[0][i]
        
        for run in range(1, len(embeddings_sets)):
            current_embedding = embeddings_sets[run][i]
            
            # Check if embeddings are identical (within floating point precision)
            if not np.allclose(base_embedding, current_embedding, rtol=1e-6, atol=1e-6):
                embedding_consistent = False
                print(f"  ❌ Embedding {i+1} for '{text}' inconsistent")
                break
        else:
            print(f"  ✅ Embedding {i+1} for '{text}' consistent")
    
    if embedding_consistent:
        print("\n🎉 SUCCESS: Model embeddings are consistent!")
    else:
        print("\n❌ FAILURE: Model embeddings are inconsistent!")
    
    return embedding_consistent

if __name__ == "__main__":
    print("Starting comprehensive reproducibility tests...")
    print("=" * 60)
    
    # Test model consistency
    model_consistent = test_model_consistency()
    
    # Test full pipeline
    pipeline_consistent = test_full_pipeline_reproducibility()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY:")
    print("=" * 60)
    print(f"Model Consistency: {'✅ PASS' if model_consistent else '❌ FAIL'}")
    print(f"Pipeline Consistency: {'✅ PASS' if pipeline_consistent else '❌ FAIL'}")
    
    if model_consistent and pipeline_consistent:
        print("\n🎉 ALL TESTS PASSED!")
        print("The system is fully deterministic and will produce identical results across all devices.")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("The system may produce different results on different devices.")
        exit(1) 