"""
TERAG: Token-Efficient Graph-Based Retrieval-Augmented Generation

A lightweight graph-based RAG framework achieving 80%+ of GraphRAG's accuracy
while consuming only 3-11% of the output tokens.

Based on: arXiv:2509.18667 (September 2025)

Main Components:
- graph_builder: Construct directed graphs from document chunks
- ner_extractor: Extract named entities and concepts
- ppr_retriever: Personalized PageRank retrieval algorithm
- terag_retriever: Main TERAG interface

Quick Start:
    from terag import TERAG, TERAGConfig

    # Build from chunks
    terag = TERAG.from_chunks_file("chunks.json")

    # Retrieve
    results, metrics = terag.retrieve("What was the revenue?")
"""

from .terag_retriever import (
    TERAG,
    TERAGConfig,
    create_terag_from_existing_chunks,
    build_terag_from_chunks,
    load_terag_from_graph
)

from .graph_builder import (
    TERAGGraph,
    GraphBuilder,
    PassageNode,
    ConceptNode
)

from .ner_extractor import (
    NERExtractor,
    QueryNER,
    extract_concepts_from_text
)

from .ppr_retriever import (
    PersonalizedPageRank,
    RetrievalResult,
    RetrievalMetrics
)

__version__ = "0.5.6"
__author__ = "Rudra Naik"

__all__ = [
    # Main interface
    "TERAG",
    "TERAGConfig",
    "create_terag_from_existing_chunks",
    "build_terag_from_chunks",
    "load_terag_from_graph",

    # Graph components
    "TERAGGraph",
    "GraphBuilder",
    "PassageNode",
    "ConceptNode",

    # NER components
    "NERExtractor",
    "QueryNER",
    "extract_concepts_from_text",

    # Retrieval components
    "PersonalizedPageRank",
    "RetrievalResult",
    "RetrievalMetrics",
]
