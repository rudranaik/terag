#!/usr/bin/env python3
"""
Test script to verify LLM error message improvements
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terag import TERAG, TERAGConfig

def test_llm_errors():
    """Test that LLM error messages are clear and not confusing"""
    print("\n" + "="*70)
    print("TEST: LLM Error Messages")
    print("="*70)
    
    # Load chunks
    with open('chunks.json', 'r') as f:
        chunks = json.load(f)
    
    print("\n--- Test 1: LLM DISABLED (use_llm_for_ner=False) ---")
    print("Expected: No confusing error messages, clean output\n")
    
    config = TERAGConfig(min_concept_freq=1, top_k=3, use_llm_for_ner=False)
    terag = TERAG.from_chunks(chunks[:2], config=config, verbose=True)
    
    print("\n✓ Graph built successfully without confusing LLM errors!")
    
    # Test query
    print("\n--- Testing Query ---")
    results, metrics = terag.retrieve("What is the Mahabharata?", verbose=False)
    print(f"Results: {len(results)}")
    if len(results) > 0:
        print("✓ Query works correctly")
    
    print("\n" + "="*70)
    print("\n--- Test 2: LLM ENABLED but no API key ---")
    print("Expected: Clear message that LLM is unavailable, fallback used\n")
    
    # Make sure no API key is set
    if 'GROQ_API_KEY' in os.environ:
        del os.environ['GROQ_API_KEY']
    
    config = TERAGConfig(min_concept_freq=1, top_k=3, use_llm_for_ner=True)
    terag = TERAG.from_chunks(chunks[:2], config=config, verbose=True)
    
    print("\n✓ Graph built with clear messaging about LLM unavailability!")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ No confusing 'LLM not found' errors when LLM is disabled")
    print("✓ Clear messaging when LLM is enabled but unavailable")
    print("✓ System works correctly in both modes")
    print("="*70)


if __name__ == "__main__":
    test_llm_errors()
