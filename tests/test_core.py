import pytest
from terag import TERAG, TERAGConfig

def test_terag_initialization_and_retrieval():
    # Sample data
    chunks = [
        {"content": "Apple Inc announced strong revenue growth in Q4 2024.", "metadata": {"source": "news"}},
        {"content": "Microsoft Corporation reported significant cloud achievements.", "metadata": {"source": "news"}},
        {"content": "The technology sector saw increased competition.", "metadata": {"source": "analysis"}}
    ]

    # Initialize TERAG
    config = TERAGConfig(top_k=2, min_concept_freq=1)
    terag = TERAG.from_chunks(chunks, config=config, verbose=False)

    # Check graph construction
    stats = terag.get_graph_statistics()
    assert stats['num_passages'] == 3
    assert stats['num_concepts'] > 0
    assert stats['num_edges'] > 0

    # Test retrieval
    query = "What about Apple's revenue?"
    results, metrics = terag.retrieve(query)

    # Verify results
    assert len(results) > 0
    assert len(results) <= 2
    assert metrics.num_results == len(results)
    
    # Check if relevant passage is retrieved
    top_passage = results[0]
    assert "Apple" in top_passage.content

def test_terag_config():
    config = TERAGConfig(top_k=5, ppr_alpha=0.2)
    assert config.top_k == 5
    assert config.ppr_alpha == 0.2
