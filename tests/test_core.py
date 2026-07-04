import pytest
from terag import TERAG, TERAGConfig
from terag import RetrievalResult, RetrievalMetrics

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

def test_retrieval_result_public_shape_and_compatibility():
    result = RetrievalResult(
        passage_id="passage_1",
        content="Example content",
        score=0.75,
        matched_concepts=["example"],
        metadata={"source": "unit-test"}
    )

    assert result.id == "passage_1"
    assert result.passage_id == "passage_1"
    assert result.text == "Example content"
    assert result.content == "Example content"

    as_dict = result.to_dict()
    assert as_dict["id"] == "passage_1"
    assert as_dict["passage_id"] == "passage_1"
    assert as_dict["content"] == "Example content"
    assert as_dict["text"] == "Example content"
    assert as_dict["score"] == 0.75
    assert as_dict["metadata"] == {"source": "unit-test"}
    assert as_dict["matched_concepts"] == ["example"]

    document = result.to_document()
    assert document["page_content"] == "Example content"
    assert document["metadata"]["id"] == "passage_1"
    assert document["metadata"]["passage_id"] == "passage_1"
    assert document["metadata"]["score"] == 0.75
    assert document["metadata"]["matched_concepts"] == ["example"]

def test_public_api_exports_result_types():
    assert RetrievalResult.__name__ == "RetrievalResult"
    assert RetrievalMetrics.__name__ == "RetrievalMetrics"

def test_terag_config():
    config = TERAGConfig(top_k=5, ppr_alpha=0.2)
    assert config.top_k == 5
    assert config.ppr_alpha == 0.2
