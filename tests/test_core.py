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

def test_query_returns_results_without_metrics_by_default():
    chunks = [
        {"content": "Apple revenue increased in Q4 2024.", "metadata": {"source": "news"}},
        {"content": "Microsoft expanded Azure cloud services.", "metadata": {"source": "news"}},
    ]
    terag = TERAG.from_chunks(
        chunks,
        config=TERAGConfig(
            top_k=1,
            min_concept_freq=1,
            max_concept_freq_ratio=1.0,
            use_semantic_entity_matching=False
        ),
        verbose=False
    )

    results = terag.query("Apple revenue")
    assert isinstance(results, list)
    assert len(results) == 1
    assert "Apple" in results[0].content

def test_query_can_return_metrics_for_compatibility():
    chunks = [
        {"content": "Apple revenue increased in Q4 2024.", "metadata": {"source": "news"}},
    ]
    terag = TERAG.from_chunks(
        chunks,
        config=TERAGConfig(
            top_k=1,
            min_concept_freq=1,
            max_concept_freq_ratio=1.0,
            use_semantic_entity_matching=False
        ),
        verbose=False
    )

    results, metrics = terag.query("Apple revenue", return_metrics=True)
    assert len(results) == 1
    assert metrics.num_results == 1

def test_empty_insert_and_query_with_string_documents():
    terag = TERAG.empty(
        config=TERAGConfig(
            top_k=1,
            min_concept_freq=1,
            max_concept_freq_ratio=1.0,
            use_semantic_entity_matching=False
        )
    )
    terag.insert([
        "Apple revenue increased in Q4 2024.",
        "Microsoft expanded Azure cloud services.",
    ], verbose=False)

    assert terag.get_graph_statistics()["num_passages"] == 2
    results = terag.query("Apple revenue")
    assert len(results) == 1
    assert "Apple" in results[0].content

def test_add_documents_supports_dict_text_and_document_like_objects():
    class Document:
        page_content = "Tesla deliveries increased."
        metadata = {"source": "doc-like"}

    terag = TERAG.empty(
        config=TERAGConfig(
            top_k=2,
            min_concept_freq=1,
            max_concept_freq_ratio=1.0,
            use_semantic_entity_matching=False
        )
    )
    terag.add_documents([
        {"text": "Apple revenue increased.", "metadata": {"source": "dict-text"}},
        Document(),
    ], verbose=False)

    results = terag.query("Tesla deliveries", top_k=2)
    assert any("Tesla" in result.content for result in results)
    assert any(result.metadata.get("source") == "doc-like" for result in results)

def test_add_chunks_rebuilds_existing_index_and_rejects_incremental_flag():
    terag = TERAG.from_chunks(
        [{"content": "Apple revenue increased.", "metadata": {"source": "initial"}}],
        config=TERAGConfig(
            top_k=2,
            min_concept_freq=1,
            max_concept_freq_ratio=1.0,
            use_semantic_entity_matching=False
        ),
        verbose=False
    )
    terag.add_chunks(
        [{"content": "Microsoft cloud revenue increased.", "metadata": {"source": "added"}}],
        verbose=False
    )

    assert terag.get_graph_statistics()["num_passages"] == 2
    results = terag.query("Microsoft revenue", top_k=2)
    assert any("Microsoft" in result.content for result in results)

    with pytest.raises(NotImplementedError):
        terag.add_chunks(
            [{"content": "A third document.", "metadata": {}}],
            rebuild=False,
            verbose=False
        )

def test_document_conversion_rejects_unknown_shape():
    terag = TERAG.empty(
        config=TERAGConfig(
            top_k=1,
            min_concept_freq=1,
            max_concept_freq_ratio=1.0,
            use_semantic_entity_matching=False
        )
    )
    with pytest.raises(TypeError):
        terag.add_documents([object()], verbose=False)

    with pytest.raises(ValueError):
        terag.add_documents([{"metadata": {"source": "missing-content"}}], verbose=False)
