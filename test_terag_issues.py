#!/usr/bin/env python3
"""
Test script to reproduce TERAG usability issues
"""
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terag import TERAG, TERAGConfig

def test_issue_1_graph_persistence():
    """Test Issue 1: Where does the graph get saved?"""
    print("\n" + "="*70)
    print("TEST 1: Graph Persistence")
    print("="*70)
    
    # Load chunks
    with open('chunks.json', 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Build TERAG
    config = TERAGConfig(min_concept_freq=1, top_k=5)
    terag = TERAG.from_chunks(chunks[:2], config=config, verbose=True)
    
    # Try to save graph
    print("\n--- Attempting to save graph ---")
    terag.save_graph("test_graph.json")
    
    # Check if file exists
    if os.path.exists("test_graph.json"):
        print(f"✓ Graph saved successfully to test_graph.json")
        print(f"  File size: {os.path.getsize('test_graph.json')} bytes")
    else:
        print("✗ Graph file not found!")
    
    return terag


def test_issue_2_llm_not_found():
    """Test Issue 2: LLM not found error"""
    print("\n" + "="*70)
    print("TEST 2: LLM Configuration")
    print("="*70)
    
    # Load chunks
    with open('chunks.json', 'r') as f:
        chunks = json.load(f)
    
    # Test with LLM enabled
    print("\n--- Testing with use_llm_for_ner=True ---")
    config = TERAGConfig(min_concept_freq=1, top_k=5, use_llm_for_ner=True)
    
    try:
        terag = TERAG.from_chunks(chunks[:2], config=config, verbose=True)
        print("✓ Graph built successfully with LLM enabled")
    except Exception as e:
        print(f"✗ Error with LLM enabled: {e}")
    
    # Test without LLM
    print("\n--- Testing with use_llm_for_ner=False ---")
    config = TERAGConfig(min_concept_freq=1, top_k=5, use_llm_for_ner=False)
    
    try:
        terag = TERAG.from_chunks(chunks[:2], config=config, verbose=True)
        print("✓ Graph built successfully without LLM")
        return terag
    except Exception as e:
        print(f"✗ Error without LLM: {e}")
        return None


def test_issue_3_blank_results(terag):
    """Test Issue 3: Query returns blank results"""
    print("\n" + "="*70)
    print("TEST 3: Query Results")
    print("="*70)
    
    if terag is None:
        print("✗ Cannot test - TERAG object not available")
        return
    
    queries = [
        "What is the Mahabharata about?",
        "Who is Vyasa?",
        "Tell me about the translation",
    ]
    
    for query in queries:
        print(f"\n--- Query: {query} ---")
        try:
            results, metrics = terag.retrieve(query, verbose=True)
            print(f"\nResults: {len(results)}")
            
            if len(results) == 0:
                print("✗ No results returned (BLANK RESULTS ISSUE)")
            else:
                print(f"✓ Got {len(results)} results")
                for i, r in enumerate(results, 1):
                    print(f"\n{i}. Score: {r.score:.4f}")
                    print(f"   Content: {r.content[:150]}...")
                    print(f"   Matched concepts: {r.matched_concepts}")
        except Exception as e:
            print(f"✗ Error during retrieval: {e}")


def test_issue_4_llm_documentation():
    """Test Issue 4: How to use other LLMs"""
    print("\n" + "="*70)
    print("TEST 4: LLM Documentation & Configuration")
    print("="*70)
    
    print("\nChecking available LLM configuration options...")
    print(f"TERAGConfig.use_llm_for_ner: {TERAGConfig.use_llm_for_ner}")
    
    # Check if there's documentation about LLM providers
    print("\nLooking for LLM provider configuration...")
    print("✗ No clear documentation on how to use different LLM providers")
    print("✗ No examples showing OpenAI, Anthropic, or other providers")


def main():
    print("\n" + "="*70)
    print("TERAG USABILITY TESTING")
    print("Testing all reported issues")
    print("="*70)
    
    # Test Issue 1: Graph persistence
    terag = test_issue_1_graph_persistence()
    
    # Test Issue 2: LLM configuration
    terag = test_issue_2_llm_not_found()
    
    # Test Issue 3: Blank results
    test_issue_3_blank_results(terag)
    
    # Test Issue 4: LLM documentation
    test_issue_4_llm_documentation()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    
    # Cleanup
    if os.path.exists("test_graph.json"):
        os.remove("test_graph.json")
        print("\nCleaned up test files")


if __name__ == "__main__":
    main()
