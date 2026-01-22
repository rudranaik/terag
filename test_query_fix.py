#!/usr/bin/env python3
"""
Test script to verify the blank query results fix
"""
import json
import sys
import os
import dotenv

dotenv.load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terag import TERAG, TERAGConfig

def test_query_fix():
    """Test that queries now return results"""
    print("\n" + "="*70)
    print("TEST: Query Results Fix")
    print("="*70)
    
    # Load chunks
    with open('chunks.json', 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Build TERAG (without LLM for faster testing)
    print("\nBuilding graph...")
    config = TERAGConfig(min_concept_freq=1, top_k=5, use_llm_for_ner=False)
    terag = TERAG.from_chunks(chunks[:3], config=config, verbose=False)
    
    # Test queries
    test_queries = [
        "What is the Mahabharata about?",
        "Who is Vyasa?",
        "Tell me about the translation",
        "Kisari Mohan Ganguli",
        "Bharata"
    ]
    
    print("\n" + "="*70)
    print("Testing Queries")
    print("="*70)
    
    total_results = 0
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        
        try:
            results, metrics = terag.retrieve(query, verbose=False)
            print(f"Results: {len(results)}")
            
            if len(results) > 0:
                print("✓ SUCCESS - Got results!")
                total_results += len(results)
                for i, r in enumerate(results[:2], 1):  # Show top 2
                    print(f"\n  {i}. Score: {r.score:.4f}")
                    print(f"     Content: {r.content[:100]}...")
                    print(f"     Matched concepts: {r.matched_concepts[:5]}")
            else:
                print("✗ FAILED - No results (still blank)")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    print("\n" + "="*70)
    print(f"SUMMARY: Got {total_results} total results across {len(test_queries)} queries")
    if total_results > 0:
        print("✓ FIX SUCCESSFUL - Queries are returning results!")
    else:
        print("✗ FIX FAILED - Still getting blank results")
    print("="*70)


if __name__ == "__main__":
    test_query_fix()
