#!/usr/bin/env python3
"""
Test script to verify deterministic behavior of the section ranker.
This script should produce identical results across multiple runs.
"""

import os
import json
import numpy as np
import torch
from section_ranker import SectionRanker

# Set deterministic settings
torch.set_deterministic(True)
torch.manual_seed(42)
np.random.seed(42)

def test_deterministic_ranking():
    """Test that ranking produces consistent results across multiple runs."""
    
    # Sample sections for testing
    test_sections = [
        {
            'document': 'test1.pdf',
            'title': 'Introduction',
            'content': 'This is an introduction to the topic.',
            'page': 1
        },
        {
            'document': 'test2.pdf', 
            'title': 'Methods',
            'content': 'Here are the methods used in this study.',
            'page': 2
        },
        {
            'document': 'test3.pdf',
            'title': 'Results',
            'content': 'The results show interesting findings.',
            'page': 3
        }
    ]
    
    # Test persona and job
    persona = "Researcher"
    job_to_be_done = "analyze research findings"
    description = "scientific study"
    
    print("Testing deterministic behavior...")
    print("=" * 50)
    
    # Run multiple times and compare results
    results = []
    for run in range(3):
        print(f"\nRun {run + 1}:")
        
        # Create fresh ranker instance
        ranker = SectionRanker()
        
        # Get rankings
        ranked = ranker.rank_sections(
            test_sections, 
            persona, 
            job_to_be_done, 
            description, 
            top_k=2
        )
        
        # Store results
        run_results = []
        for item in ranked:
            run_results.append({
                'title': item['section']['title'],
                'score': round(item['similarity_score'], 6)
            })
        
        results.append(run_results)
        
        # Print results
        for i, item in enumerate(run_results):
            print(f"  {i+1}. {item['title']} (score: {item['score']})")
    
    # Verify consistency
    print("\n" + "=" * 50)
    print("Consistency Check:")
    
    all_consistent = True
    for i in range(len(results[0])):
        for run in range(1, len(results)):
            if (results[run][i]['title'] != results[0][i]['title'] or 
                abs(results[run][i]['score'] - results[0][i]['score']) > 1e-6):
                all_consistent = False
                print(f"  ❌ Inconsistency found in position {i+1}")
                break
    
    if all_consistent:
        print("  ✅ All runs produced identical results")
    else:
        print("  ❌ Results are not deterministic")
    
    return all_consistent

if __name__ == "__main__":
    success = test_deterministic_ranking()
    exit(0 if success else 1) 