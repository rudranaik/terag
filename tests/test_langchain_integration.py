import pytest

from terag import EmbeddingConfig, GraphConfig, RetrievalConfig, TERAG, TERAGConfig
from terag.retrieval.ppr import RetrievalResult

pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from terag.integrations.langchain import TERAGRetriever, results_to_documents


def _build_rag():
    rag = TERAG.empty(
        config=TERAGConfig(
            graph_config=GraphConfig(min_concept_freq=1, max_concept_freq_ratio=1.0),
            retrieval_config=RetrievalConfig(top_k=2),
            embedding_config=EmbeddingConfig(use_semantic_entity_matching=False),
        )
    )
    rag.insert(
        [
            "Apple revenue increased in Q4 2024.",
            "Microsoft expanded Azure cloud services.",
        ],
        verbose=False,
    )
    return rag


def test_results_to_langchain_documents_preserves_result_metadata():
    result = RetrievalResult(
        passage_id="passage_1",
        content="Example content",
        score=0.42,
        matched_concepts=["example"],
        metadata={"source": "unit-test"},
    )

    documents = results_to_documents([result])

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "Example content"
    assert documents[0].metadata["source"] == "unit-test"
    assert documents[0].metadata["id"] == "passage_1"
    assert documents[0].metadata["passage_id"] == "passage_1"
    assert documents[0].metadata["score"] == 0.42
    assert documents[0].metadata["matched_concepts"] == ["example"]


def test_as_langchain_retriever_invokes_terag_query():
    rag = _build_rag()

    retriever = rag.as_langchain_retriever(top_k=1)
    documents = retriever.invoke("Apple revenue")

    assert isinstance(retriever, TERAGRetriever)
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert "Apple" in documents[0].page_content
    assert documents[0].metadata["score"] >= 0
    assert documents[0].metadata["passage_id"]
