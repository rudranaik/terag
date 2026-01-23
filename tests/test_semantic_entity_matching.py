"""
Tests for semantic entity matching in TERAG

Tests the three matching strategies:
1. Exact text match
2. Partial text match (substring)
3. Semantic similarity match
"""

import pytest
import numpy as np
from terag import TERAG, TERAGConfig
from terag.graph.builder import TERAGGraph, PassageNode, ConceptNode
from terag.retrieval.ppr import TERAGRetriever


class MockEmbeddingModel:
    """Mock embedding model for testing semantic matching"""
    
    def __init__(self, embeddings_map):
        """
        Args:
            embeddings_map: Dict mapping text -> embedding vector
        """
        self.embeddings_map = embeddings_map
    
    def encode(self, texts, show_progress_bar=False):
        """Return embeddings for given texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Normalize text same way as TERAG
            normalized = " ".join(text.lower().strip().split())
            if normalized in self.embeddings_map:
                embeddings.append(self.embeddings_map[normalized])
            else:
                # Return random embedding for unknown text
                embeddings.append(np.random.randn(384))
        
        return np.array(embeddings)


def create_test_graph():
    """Create a simple test graph with known concepts"""
    graph = TERAGGraph()
    
    # Add passages
    passages = [
        PassageNode("p1", "The company's cash flow improved significantly in Q4 2024.", 0, {"source": "report"}),
        PassageNode("p2", "Artificial intelligence investments drove revenue growth.", 1, {"source": "report"}),
        PassageNode("p3", "The CEO announced a new strategic initiative.", 2, {"source": "news"}),
    ]
    
    for p in passages:
        graph.add_passage(p)
    
    # Add concepts (normalized)
    concepts = [
        ConceptNode("cash flow", "cash flow", "document_concept", frequency=1),
        ConceptNode("artificial intelligence", "artificial intelligence", "document_concept", frequency=1),
        ConceptNode("ceo", "CEO", "named_entity", frequency=1),
        ConceptNode("revenue", "revenue", "document_concept", frequency=1),
        ConceptNode("q4 2024", "Q4 2024", "date", frequency=1),
    ]
    
    for c in concepts:
        graph.add_concept(c)
    
    # Add edges
    graph.add_edge("p1", "cash flow")
    graph.add_edge("p1", "q4 2024")
    graph.add_edge("p2", "artificial intelligence")
    graph.add_edge("p2", "revenue")
    graph.add_edge("p3", "ceo")
    
    return graph


def test_exact_match():
    """Test exact text matching"""
    graph = create_test_graph()
    
    # No embedding model - text matching only
    retriever = TERAGRetriever(
        graph=graph,
        embedding_model=None,
        use_semantic_matching=False
    )
    
    # Test exact match
    query_entities = ["cash flow", "revenue"]
    matched = retriever._match_entities_to_concepts(query_entities)
    
    assert "cash flow" in matched
    assert "revenue" in matched
    assert len(matched) == 2


def test_partial_match():
    """Test partial (substring) matching"""
    graph = create_test_graph()
    
    retriever = TERAGRetriever(
        graph=graph,
        embedding_model=None,
        use_semantic_matching=False
    )
    
    # Test partial match - "cash" should match "cash flow"
    query_entities = ["cash"]
    matched = retriever._match_entities_to_concepts(query_entities)
    
    assert "cash flow" in matched


def test_spelling_variations():
    """Test semantic matching handles spelling variations (e.g., 'cashflow' vs 'cash flow')"""
    graph = create_test_graph()
    
    # Create embeddings where "cashflow" and "cash flow" are very similar
    embeddings_map = {
        "cashflow": np.array([1.0, 0.0, 0.0] + [0.0] * 381),  # 384-dim vector
        "cash flow": np.array([0.95, 0.1, 0.0] + [0.0] * 381),  # Very similar
        "artificial intelligence": np.array([0.0, 1.0, 0.0] + [0.0] * 381),
        "ai": np.array([0.05, 0.95, 0.0] + [0.0] * 381),  # Similar to "artificial intelligence"
        "ceo": np.array([0.0, 0.0, 1.0] + [0.0] * 381),
        "revenue": np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 380),
        "q4 2024": np.array([0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 379),
    }
    
    mock_model = MockEmbeddingModel(embeddings_map)
    
    retriever = TERAGRetriever(
        graph=graph,
        embedding_model=mock_model,
        use_semantic_matching=True,
        semantic_threshold=0.7
    )
    
    # Test: "cashflow" (one word) should match "cash flow" (two words) via semantic similarity
    query_entities = ["cashflow"]
    matched = retriever._match_entities_to_concepts(query_entities)
    
    assert "cash flow" in matched, "Semantic matching should match 'cashflow' to 'cash flow'"


def test_abbreviations():
    """Test semantic matching handles abbreviations (e.g., 'AI' vs 'artificial intelligence')"""
    graph = create_test_graph()
    
    # Create embeddings where "ai" and "artificial intelligence" are similar
    embeddings_map = {
        "ai": np.array([0.05, 0.95, 0.0] + [0.0] * 381),
        "artificial intelligence": np.array([0.0, 1.0, 0.0] + [0.0] * 381),
        "cash flow": np.array([1.0, 0.0, 0.0] + [0.0] * 381),
        "ceo": np.array([0.0, 0.0, 1.0] + [0.0] * 381),
        "revenue": np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 380),
        "q4 2024": np.array([0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 379),
    }
    
    mock_model = MockEmbeddingModel(embeddings_map)
    
    retriever = TERAGRetriever(
        graph=graph,
        embedding_model=mock_model,
        use_semantic_matching=True,
        semantic_threshold=0.7
    )
    
    # Test: "AI" should match "artificial intelligence"
    query_entities = ["AI"]
    matched = retriever._match_entities_to_concepts(query_entities)
    
    assert "artificial intelligence" in matched, "Semantic matching should match 'AI' to 'artificial intelligence'"


def test_threshold_sensitivity():
    """Test that different thresholds produce different results"""
    graph = create_test_graph()
    
    # Create embeddings with varying similarities
    embeddings_map = {
        "income": np.array([0.8, 0.0, 0.0] + [0.0] * 381),  # Similarity to "revenue" = 0.8
        "revenue": np.array([1.0, 0.0, 0.0] + [0.0] * 381),
        "cash flow": np.array([0.0, 1.0, 0.0] + [0.0] * 381),
        "artificial intelligence": np.array([0.0, 0.0, 1.0] + [0.0] * 381),
        "ceo": np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 380),
        "q4 2024": np.array([0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 379),
    }
    
    mock_model = MockEmbeddingModel(embeddings_map)
    
    # Test with high threshold (0.85) - should NOT match
    retriever_high = TERAGRetriever(
        graph=graph,
        embedding_model=mock_model,
        use_semantic_matching=True,
        semantic_threshold=0.85
    )
    
    query_entities = ["income"]
    matched_high = retriever_high._match_entities_to_concepts(query_entities)
    
    # Test with lower threshold (0.7) - should match
    retriever_low = TERAGRetriever(
        graph=graph,
        embedding_model=mock_model,
        use_semantic_matching=True,
        semantic_threshold=0.7
    )
    
    matched_low = retriever_low._match_entities_to_concepts(query_entities)
    
    # With threshold 0.85, similarity of 0.8 should NOT match
    assert "revenue" not in matched_high, "High threshold (0.85) should not match similarity of 0.8"
    
    # With threshold 0.7, similarity of 0.8 should match
    assert "revenue" in matched_low, "Low threshold (0.7) should match similarity of 0.8"


def test_fallback_without_embeddings():
    """Test graceful fallback to text-only matching when no embedding model provided"""
    graph = create_test_graph()
    
    # Semantic matching enabled but no embedding model
    retriever = TERAGRetriever(
        graph=graph,
        embedding_model=None,
        use_semantic_matching=True,  # Enabled but will fall back
        semantic_threshold=0.7
    )
    
    # Should still work with text matching
    query_entities = ["cash flow", "revenue"]
    matched = retriever._match_entities_to_concepts(query_entities)
    
    assert "cash flow" in matched
    assert "revenue" in matched


def test_parallel_matching_strategies():
    """Test that all three strategies run in parallel and combine results"""
    graph = create_test_graph()
    
    # Create embeddings
    embeddings_map = {
        "cashflow": np.array([0.95, 0.1, 0.0] + [0.0] * 381),  # Similar to "cash flow"
        "cash flow": np.array([1.0, 0.0, 0.0] + [0.0] * 381),
        "cash": np.array([0.5, 0.5, 0.0] + [0.0] * 381),  # Different from "cash flow"
        "revenue": np.array([0.0, 1.0, 0.0] + [0.0] * 381),
        "artificial intelligence": np.array([0.0, 0.0, 1.0] + [0.0] * 381),
        "ceo": np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 380),
        "q4 2024": np.array([0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 379),
    }
    
    mock_model = MockEmbeddingModel(embeddings_map)
    
    retriever = TERAGRetriever(
        graph=graph,
        embedding_model=mock_model,
        use_semantic_matching=True,
        semantic_threshold=0.7
    )
    
    # Query with entities that match via different strategies:
    # - "revenue" -> exact match
    # - "cash" -> partial match (substring of "cash flow")
    # - "cashflow" -> semantic match (similar to "cash flow")
    query_entities = ["revenue", "cash", "cashflow"]
    matched = retriever._match_entities_to_concepts(query_entities)
    
    # All should be found via their respective strategies
    assert "revenue" in matched, "Should match via exact match"
    assert "cash flow" in matched, "Should match via partial AND semantic match"
    
    # Should have at least 2 unique concepts matched
    assert len(matched) >= 2


def test_integration_with_terag():
    """Integration test: Full TERAG workflow with semantic matching"""
    # Create chunks
    chunks = [
        {"content": "The company's cash flow improved significantly in Q4 2024.", "metadata": {"source": "report"}},
        {"content": "Artificial intelligence investments drove revenue growth.", "metadata": {"source": "report"}},
        {"content": "The CEO announced a new strategic initiative.", "metadata": {"source": "news"}},
    ]
    
    # Create embeddings for integration test
    embeddings_map = {
        "cashflow": np.array([0.95, 0.1, 0.0] + [0.0] * 381),
        "cash flow": np.array([1.0, 0.0, 0.0] + [0.0] * 381),
        "ai": np.array([0.05, 0.95, 0.0] + [0.0] * 381),
        "artificial intelligence": np.array([0.0, 1.0, 0.0] + [0.0] * 381),
        "ceo": np.array([0.0, 0.0, 1.0] + [0.0] * 381),
        "chief executive": np.array([0.1, 0.0, 0.9] + [0.0] * 381),
        "revenue": np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 380),
        "q4 2024": np.array([0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 379),
        "strategy": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 378),
        "strategic": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9] + [0.0] * 378),
        # Add query embeddings
        "what is their cashflow strategy?": np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5] + [0.0] * 378),
    }
    
    mock_model = MockEmbeddingModel(embeddings_map)
    
    # Configure with semantic matching enabled
    config = TERAGConfig(
        use_semantic_entity_matching=True,
        semantic_match_threshold=0.7,
        min_concept_freq=1,
        top_k=3
    )
    
    # Build TERAG
    terag = TERAG.from_chunks(
        chunks,
        config=config,
        embedding_model=mock_model,
        verbose=False
    )
    
    # Query with spelling variation: "cashflow" instead of "cash flow"
    results, metrics = terag.retrieve("What is their cashflow strategy?", verbose=False)
    
    # Should retrieve results even though query uses "cashflow" and graph has "cash flow"
    assert len(results) > 0, "Should retrieve results with semantic matching"
    assert metrics.num_matched_concepts > 0, "Should match concepts via semantic similarity"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
