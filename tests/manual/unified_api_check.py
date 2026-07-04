#!/usr/bin/env python3
"""Test script for unified retrieval API"""
import os
import time
from terag import TERAG, TERAGConfig

def main():
    print('=== Testing Unified Retrieval API ===\n')
    
    # Create test chunks
    chunks = [
        {'content': 'Machine learning is a subset of artificial intelligence.', 'metadata': {'source': 'doc1'}},
        {'content': 'Deep learning uses neural networks with multiple layers.', 'metadata': {'source': 'doc2'}}
    ]
    
    print(f'Building graph from {len(chunks)} chunks...')
    config = TERAGConfig(use_llm_for_ner=False)
    terag = TERAG.from_chunks(chunks, config=config)
    print(f'✓ TERAG instance created with {len(terag.graph.passages)} passages\n')
    
    query = 'What is machine learning?'
    
    # Test PPR
    print('Testing PPR...')
    try:
        start = time.time()
        results, metrics = terag.retrieve(query, method='ppr', top_k=5)
        elapsed = time.time() - start
        print(f'  ✓ PPR: {len(results)} results in {elapsed:.3f}s')
        print(f'    Metrics: {metrics.num_results} results, {metrics.retrieval_time:.3f}s')
    except Exception as e:
        print(f'  ✗ PPR failed: {e}')
        return False
    
    # Test Semantic
    print('\nTesting Semantic...')
    try:
        start = time.time()
        results, metrics = terag.retrieve(query, method='semantic', top_k=5)
        elapsed = time.time() - start
        print(f'  ✓ Semantic: {len(results)} results in {elapsed:.3f}s')
        print(f'    Metrics: {metrics.num_results} results, {metrics.retrieval_time:.3f}s')
    except Exception as e:
        print(f'  ✗ Semantic failed: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    # Test Hybrid
    print('\nTesting Hybrid...')
    try:
        start = time.time()
        results, metrics = terag.retrieve(query, method='hybrid', top_k=5)
        elapsed = time.time() - start
        print(f'  ✓ Hybrid: {len(results)} results in {elapsed:.3f}s')
        print(f'    Metrics: {metrics.num_results} results, {metrics.retrieval_time:.3f}s')
    except Exception as e:
        print(f'  ✗ Hybrid failed: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    print('\n✓ All retrieval methods working!')
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
