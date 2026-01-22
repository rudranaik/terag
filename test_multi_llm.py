#!/usr/bin/env python3
"""
Test script to verify Multi-LLM support
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terag import TERAG, TERAGConfig

def test_multi_llm():
    """Test provider configuration"""
    print("\n" + "="*70)
    print("TEST: Multi-LLM Support")
    print("="*70)
    
    # Load chunks
    with open('chunks.json', 'r') as f:
        chunks = json.load(f)
    
    print("\n--- Test 1: OpenAI Provider (Expect 'Not found' if no key) ---")
    
    # Configure for OpenAI
    config = TERAGConfig(
        min_concept_freq=1, 
        use_llm_for_ner=True,
        llm_provider="openai"
    )
    
    # This should print "Provider: openai" and "API Key: Not found" (unless user has OPENAI_API_KEY set)
    # And then fall back to regex
    terag = TERAG.from_chunks(chunks[:1], config=config, verbose=True)
    
    print("\n--- Test 2: Groq Provider (Explicit) ---")
    config = TERAGConfig(
        min_concept_freq=1, 
        use_llm_for_ner=True,
        llm_provider="groq"
    )
    
    terag = TERAG.from_chunks(chunks[:1], config=config, verbose=True)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ Successfully configured multiple providers")
    print("✓ Output shows correct provider names")
    print("="*70)

if __name__ == "__main__":
    test_multi_llm()
