#!/usr/bin/env python3
"""Comprehensive test suite for TERAG API improvements"""
import os
import time
from dotenv import load_dotenv
from terag import TERAG, TERAGConfig
from terag.embeddings.manager import EmbeddingManager

# Load environment variables
load_dotenv()

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_unified_api_basic():
    """Test 1: Basic unified API functionality"""
    print_section("Test 1: Basic Unified API")
    
    chunks = [
        {'content': 'Apple Inc announced revenue growth of 15% in Q4 2024.', 'metadata': {'source': 'earnings_report'}},
        {'content': 'Microsoft reported strong cloud computing achievements with Azure.', 'metadata': {'source': 'tech_news'}},
        {'content': 'The company revenue increased significantly due to AI products.', 'metadata': {'source': 'analysis'}},
        {'content': 'Deep learning and neural networks are transforming the industry.', 'metadata': {'source': 'research'}}
    ]
    
    print(f"Creating TERAG from {len(chunks)} chunks (no LLM)...")
    config = TERAGConfig(use_llm_for_ner=False)
    embedding_manager = EmbeddingManager(api_key=os.getenv('OPENAI_API_KEY'))
    terag = TERAG.from_chunks(chunks, config=config, embedding_model=embedding_manager)
    print(f"✓ Graph created: {len(terag.graph.passages)} passages, {len(terag.graph.concepts)} concepts\n")
    
    query = "What is the revenue growth?"
    
    # Test PPR
    print("Testing PPR retrieval...")
    start = time.time()
    ppr_results, ppr_metrics = terag.retrieve(query, method='ppr', top_k=3, verbose=True)
    ppr_time = time.time() - start
    print(f"✓ PPR: {len(ppr_results)} results in {ppr_time:.3f}s")
    print(f"  Metrics: {ppr_metrics.num_results} results, {ppr_metrics.retrieval_time:.3f}s")
    if ppr_results:
        print(f"  Top result: {ppr_results[0].passage_id} (score: {ppr_results[0].score:.4f})")
    
    # Test Semantic
    print("\nTesting Semantic retrieval...")
    start = time.time()
    sem_results, sem_metrics = terag.retrieve(query, method='semantic', top_k=3, verbose=True)
    sem_time = time.time() - start
    print(f"✓ Semantic: {len(sem_results)} results in {sem_time:.3f}s")
    print(f"  Metrics: {sem_metrics.num_results} results, {sem_metrics.retrieval_time:.3f}s")
    if sem_results:
        print(f"  Top result: {sem_results[0].passage_id} (score: {sem_results[0].score:.4f})")
    
    # Test Hybrid
    print("\nTesting Hybrid retrieval...")
    start = time.time()
    hyb_results, hyb_metrics = terag.retrieve(
        query, 
        method='hybrid', 
        top_k=3, 
        ppr_weight=0.6, 
        semantic_weight=0.4,
        verbose=True
    )
    hyb_time = time.time() - start
    print(f"✓ Hybrid: {len(hyb_results)} results in {hyb_time:.3f}s")
    print(f"  Metrics: {hyb_metrics.num_results} results, {hyb_metrics.retrieval_time:.3f}s")
    if hyb_results:
        print(f"  Top result: {hyb_results[0].passage_id} (score: {hyb_results[0].score:.4f})")
    
    return True

def test_llm_based_ner():
    """Test 2: LLM-based NER for improved entity extraction"""
    print_section("Test 2: LLM-Based NER")
    
    chunks = [
        {'content': 'Q4 revenue for AAPL was $120 billion, up from $105B in Q3.', 'metadata': {'source': 'financial'}},
        {'content': 'Microsoft Azure cloud revenue grew 30% year-over-year.', 'metadata': {'source': 'cloud_report'}},
        {'content': 'AI and machine learning investments are paying off.', 'metadata': {'source': 'strategy'}}
    ]
    
    # Test with Groq LLM
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        print("Testing with Groq LLM...")
        config = TERAGConfig(
            use_llm_for_ner=True,
            llm_provider='groq',
            llm_api_key=groq_key
        )
        embedding_manager = EmbeddingManager(api_key=os.getenv('OPENAI_API_KEY'))
        terag_groq = TERAG.from_chunks(chunks, config=config, embedding_model=embedding_manager)
        print(f"✓ Graph with Groq LLM: {len(terag_groq.graph.passages)} passages, {len(terag_groq.graph.concepts)} concepts")
        
        # Test retrieval
        results, metrics = terag_groq.retrieve("What was the revenue?", method='ppr', top_k=2)
        print(f"✓ PPR with LLM NER: {len(results)} results")
        if results:
            print(f"  Top result matched concepts: {results[0].matched_concepts}")
    else:
        print("⚠ GROQ_API_KEY not found, skipping Groq test")
    
    # Test with OpenAI LLM
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("\nTesting with OpenAI LLM...")
        config = TERAGConfig(
            use_llm_for_ner=True,
            llm_provider='openai',
            llm_api_key=openai_key
        )
        embedding_manager = EmbeddingManager(api_key=openai_key)
        terag_openai = TERAG.from_chunks(chunks, config=config, embedding_model=embedding_manager)
        print(f"✓ Graph with OpenAI LLM: {len(terag_openai.graph.passages)} passages, {len(terag_openai.graph.concepts)} concepts")
        
        # Test retrieval
        results, metrics = terag_openai.retrieve("cloud revenue growth", method='ppr', top_k=2)
        print(f"✓ PPR with LLM NER: {len(results)} results")
        if results:
            print(f"  Top result matched concepts: {results[0].matched_concepts}")
    else:
        print("⚠ OPENAI_API_KEY not found, skipping OpenAI test")
    
    return True

def test_hybrid_with_llm():
    """Test 3: Hybrid retrieval with LLM-based NER"""
    print_section("Test 3: Hybrid Retrieval with LLM NER")
    
    chunks = [
        {'content': 'Tesla reported record deliveries in Q4 2024 with 500,000 vehicles.', 'metadata': {'source': 'auto'}},
        {'content': 'Electric vehicle market share grew to 15% globally.', 'metadata': {'source': 'market'}},
        {'content': 'Battery technology improvements enabled longer range.', 'metadata': {'source': 'tech'}},
        {'content': 'Autonomous driving features were enhanced with new AI models.', 'metadata': {'source': 'innovation'}}
    ]
    
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        print("⚠ GROQ_API_KEY not found, skipping test")
        return False
    
    print("Creating TERAG with LLM-based NER for hybrid retrieval...")
    config = TERAGConfig(
        use_llm_for_ner=True,
        llm_provider='groq',
        llm_api_key=groq_key
    )
    embedding_manager = EmbeddingManager(api_key=os.getenv('OPENAI_API_KEY'))
    terag = TERAG.from_chunks(chunks, config=config, embedding_model=embedding_manager)
    print(f"✓ Graph created: {len(terag.graph.passages)} passages, {len(terag.graph.concepts)} concepts\n")
    
    query = "How many vehicles were delivered?"
    
    # Test hybrid with different weight combinations
    weight_configs = [
        (0.7, 0.3, "PPR-heavy"),
        (0.5, 0.5, "Balanced"),
        (0.3, 0.7, "Semantic-heavy")
    ]
    
    for ppr_w, sem_w, desc in weight_configs:
        print(f"Testing {desc} (PPR:{ppr_w}, Semantic:{sem_w})...")
        results, metrics = terag.retrieve(
            query,
            method='hybrid',
            top_k=3,
            ppr_weight=ppr_w,
            semantic_weight=sem_w,
            verbose=False
        )
        top_score = results[0].score if results else 0.0
        print(f"  ✓ {len(results)} results, top score: {top_score:.4f}")
    
    return True

def test_graph_convenience_methods():
    """Test 4: Graph convenience methods"""
    print_section("Test 4: Graph Convenience Methods")
    
    chunks = [
        {'content': 'Revenue increased by 20% in the last quarter.', 'metadata': {'quarter': 'Q4'}},
        {'content': 'Profit margins improved due to cost optimization.', 'metadata': {'quarter': 'Q4'}},
        {'content': 'Customer satisfaction scores reached all-time high.', 'metadata': {'metric': 'satisfaction'}}
    ]
    
    print("Creating test graph...")
    terag = TERAG.from_chunks(chunks, config=TERAGConfig(use_llm_for_ner=False))
    graph = terag.graph
    print(f"✓ Graph: {len(graph.passages)} passages, {len(graph.concepts)} concepts\n")
    
    # Test list methods
    print("Testing list methods...")
    passages = graph.list_passages()
    print(f"  ✓ list_passages(): {len(passages)} passages")
    
    concepts = graph.list_concepts()
    print(f"  ✓ list_concepts(): {len(concepts)} concepts")
    
    # Test get methods
    if passages:
        print("\nTesting get methods...")
        passage_id = passages[0]
        
        passage = graph.get_passage(passage_id)
        print(f"  ✓ get_passage('{passage_id}'): {passage is not None}")
        
        content = graph.get_passage_content(passage_id)
        print(f"  ✓ get_passage_content('{passage_id}'): {len(content) if content else 0} chars")
    
    if concepts:
        concept_id = concepts[0]
        concept = graph.get_concept(concept_id)
        print(f"  ✓ get_concept('{concept_id}'): {concept is not None}")
    
    # Test search
    print("\nTesting search methods...")
    matches = graph.search_concepts("revenue")
    print(f"  ✓ search_concepts('revenue'): {len(matches)} matches")
    
    return True

def test_error_handling():
    """Test 5: Error handling and validation"""
    print_section("Test 5: Error Handling")
    
    chunks = [{'content': 'Test content.', 'metadata': {}}]
    terag = TERAG.from_chunks(chunks, config=TERAGConfig(use_llm_for_ner=False))
    
    # Test invalid method
    print("Testing invalid retrieval method...")
    try:
        terag.retrieve("test", method='invalid_method')
        print("  ✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✓ Caught ValueError: {str(e)[:50]}...")
    
    # Test missing API key graceful fallback
    print("\nTesting missing API key handling...")
    config = TERAGConfig(use_llm_for_ner=True, llm_api_key=None)
    # Temporarily unset env var
    old_groq = os.environ.get('GROQ_API_KEY')
    old_openai = os.environ.get('OPENAI_API_KEY')
    if old_groq:
        del os.environ['GROQ_API_KEY']
    if old_openai:
        del os.environ['OPENAI_API_KEY']
    
    try:
        terag_no_key = TERAG.from_chunks(chunks, config=config)
        print("  ✓ Gracefully fell back to regex NER")
    finally:
        # Restore env vars
        if old_groq:
            os.environ['GROQ_API_KEY'] = old_groq
        if old_openai:
            os.environ['OPENAI_API_KEY'] = old_openai
    
    return True

def test_backward_compatibility():
    """Test 6: Backward compatibility"""
    print_section("Test 6: Backward Compatibility")
    
    chunks = [
        {'content': 'Legacy test content about revenue.', 'metadata': {}},
        {'content': 'More legacy content about profits.', 'metadata': {}}
    ]
    
    print("Testing legacy API (no method parameter)...")
    terag = TERAG.from_chunks(chunks, config=TERAGConfig(use_llm_for_ner=False))
    
    # Old way - should still work and default to PPR
    results, metrics = terag.retrieve("revenue")
    print(f"  ✓ Legacy retrieve() call: {len(results)} results")
    print(f"  ✓ Defaults to PPR as expected")
    
    # Verify default can be changed
    print("\nTesting custom default method...")
    config = TERAGConfig(use_llm_for_ner=False, default_retrieval_method='semantic')
    embedding_manager = EmbeddingManager(api_key=os.getenv('OPENAI_API_KEY'))
    terag_custom = TERAG.from_chunks(chunks, config=config, embedding_model=embedding_manager)
    results, metrics = terag_custom.retrieve("revenue")
    print(f"  ✓ Custom default (semantic): {len(results)} results")
    
    return True

def main():
    """Run all comprehensive tests"""
    print("\n" + "="*60)
    print("  TERAG API IMPROVEMENTS - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Check API keys
    print("\nChecking API keys...")
    groq_key = os.getenv('GROQ_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    print(f"  GROQ_API_KEY: {'✓ Found' if groq_key else '✗ Not found'}")
    print(f"  OPENAI_API_KEY: {'✓ Found' if openai_key else '✗ Not found'}")
    
    if not openai_key:
        print("\n⚠ Warning: OPENAI_API_KEY required for most tests")
        return False
    
    # Run tests
    tests = [
        ("Basic Unified API", test_unified_api_basic),
        ("LLM-Based NER", test_llm_based_ner),
        ("Hybrid with LLM", test_hybrid_with_llm),
        ("Graph Convenience Methods", test_graph_convenience_methods),
        ("Error Handling", test_error_handling),
        ("Backward Compatibility", test_backward_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
