"""
TERAG Main Retriever Interface

Complete TERAG retrieval system integrating:
1. Graph construction from chunks
2. Named Entity Recognition
3. Personalized PageRank retrieval
4. Result ranking and formatting
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

from terag.graph.builder import TERAGGraph, GraphBuilder
from terag.ingestion.ner_extractor import NERExtractor, QueryNER, extract_concepts_from_text
from terag.retrieval.ppr import TERAGRetriever as PPRRetriever, RetrievalResult, RetrievalMetrics


@dataclass
class TERAGConfig:
    """Configuration for TERAG system"""
    # Graph construction
    min_concept_freq: int = 2
    max_concept_freq_ratio: float = 0.5
    enable_concept_clustering: bool = False

    # PPR parameters
    ppr_alpha: float = 0.15
    ppr_max_iterations: int = 100

    # Weighting
    semantic_weight: float = 0.5
    frequency_weight: float = 0.5

    # Retrieval
    top_k: int = 10

    # Optional LLM for enhanced NER
    use_llm_for_ner: bool = False


class TERAG:
    """
    Complete TERAG system

    Usage:
        # Build from chunks
        terag = TERAG.from_chunks(chunks, config)

        # Retrieve
        results = terag.retrieve(query, top_k=10)
    """

    def __init__(
        self,
        graph: TERAGGraph,
        config: TERAGConfig,
        embedding_model: Optional[object] = None
    ):
        self.graph = graph
        self.config = config
        self.embedding_model = embedding_model

        # Initialize components
        self.ner_extractor = NERExtractor(use_llm=config.use_llm_for_ner)
        self.query_ner = QueryNER()
        self.retriever = PPRRetriever(
            graph=graph,
            embedding_model=embedding_model,
            alpha=config.ppr_alpha,
            ppr_max_iterations=config.ppr_max_iterations
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: List[Dict],
        config: Optional[TERAGConfig] = None,
        embedding_model: Optional[object] = None,
        verbose: bool = True
    ) -> 'TERAG':
        """
        Build TERAG system from document chunks

        Args:
            chunks: List of chunk dicts with 'content' and 'metadata'
            config: TERAG configuration
            embedding_model: Optional embedding model (e.g., SentenceTransformer)
            verbose: Print progress

        Returns:
            TERAG system ready for retrieval
        """
        if config is None:
            config = TERAGConfig()

        if verbose:
            print("Building TERAG system from chunks...")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Config: {config}")

        start_time = time.time()

        # Build graph
        builder = GraphBuilder(
            min_concept_freq=config.min_concept_freq,
            max_concept_freq_ratio=config.max_concept_freq_ratio,
            enable_concept_clustering=config.enable_concept_clustering
        )

        graph = builder.build_graph_from_chunks(
            chunks=chunks,
            extract_concepts_fn=extract_concepts_from_text
        )

        build_time = time.time() - start_time

        if verbose:
            print(f"\nGraph built in {build_time:.2f}s")
            stats = graph.get_statistics()
            print(f"  Passages: {stats['num_passages']}")
            print(f"  Concepts: {stats['num_concepts']}")
            print(f"  Edges: {stats['num_edges']}")

        # Create TERAG system
        terag = cls(graph=graph, config=config, embedding_model=embedding_model)

        return terag

    @classmethod
    def from_chunks_file(
        cls,
        chunks_file: str,
        config: Optional[TERAGConfig] = None,
        embedding_model: Optional[object] = None,
        verbose: bool = True
    ) -> 'TERAG':
        """
        Build TERAG system from chunks JSON file

        Args:
            chunks_file: Path to chunks JSON file
            config: TERAG configuration
            embedding_model: Optional embedding model
            verbose: Print progress

        Returns:
            TERAG system
        """
        if verbose:
            print(f"Loading chunks from {chunks_file}...")

        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        return cls.from_chunks(chunks, config, embedding_model, verbose)

    @classmethod
    def from_graph_file(
        cls,
        graph_file: str,
        config: Optional[TERAGConfig] = None,
        embedding_model: Optional[object] = None,
        verbose: bool = True
    ) -> 'TERAG':
        """
        Load TERAG system from pre-built graph file

        Args:
            graph_file: Path to graph JSON file
            config: TERAG configuration
            embedding_model: Optional embedding model
            verbose: Print progress

        Returns:
            TERAG system
        """
        if verbose:
            print(f"Loading graph from {graph_file}...")

        graph = TERAGGraph.load_from_file(graph_file)

        if config is None:
            config = TERAGConfig()

        if verbose:
            stats = graph.get_statistics()
            print(f"  Passages: {stats['num_passages']}")
            print(f"  Concepts: {stats['num_concepts']}")
            print(f"  Edges: {stats['num_edges']}")

        return cls(graph=graph, config=config, embedding_model=embedding_model)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Retrieve relevant passages for a query

        Args:
            query: User query
            top_k: Number of results (default: from config)
            verbose: Print progress

        Returns:
            (results, metrics)
        """
        if top_k is None:
            top_k = self.config.top_k

        # Extract query entities
        query_entities = self.query_ner.extract_query_entities(query)

        # Retrieve using PPR
        results, metrics = self.retriever.retrieve(
            query=query,
            query_entities=query_entities,
            top_k=top_k,
            semantic_weight=self.config.semantic_weight,
            frequency_weight=self.config.frequency_weight,
            verbose=verbose
        )

        return results, metrics

    def save_graph(self, filepath: str):
        """Save graph to file for reuse"""
        self.graph.save_to_file(filepath)
        print(f"Graph saved to {filepath}")

    def get_graph_statistics(self) -> Dict:
        """Get graph statistics"""
        return self.graph.get_statistics()


def create_terag_from_existing_chunks(
    chunks_file: str = "chunks_full_metadata.json",
    output_graph_file: Optional[str] = "terag_graph.json",
    config: Optional[TERAGConfig] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> TERAG:
    """
    Convenience function to create TERAG from existing chunks

    Args:
        chunks_file: Path to chunks file
        output_graph_file: Where to save graph (None = don't save)
        config: TERAG config
        embedding_model_name: Name of embedding model to use

    Returns:
        TERAG system ready for use
    """
    # Load embedding model if available
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {embedding_model_name}")
        embedding_model = SentenceTransformer(embedding_model_name)
    except ImportError:
        print("Warning: sentence-transformers not available, semantic weighting disabled")

    # Build TERAG
    terag = TERAG.from_chunks_file(
        chunks_file=chunks_file,
        config=config,
        embedding_model=embedding_model,
        verbose=True
    )

    # Save graph if requested
    if output_graph_file:
        terag.save_graph(output_graph_file)

    return terag


# Convenience functions for integration
def build_terag_from_chunks(chunks: List[Dict], **kwargs) -> TERAG:
    """Quick function to build TERAG from chunks"""
    return TERAG.from_chunks(chunks, **kwargs)


def load_terag_from_graph(graph_file: str, **kwargs) -> TERAG:
    """Quick function to load TERAG from graph file"""
    return TERAG.from_graph_file(graph_file, **kwargs)


if __name__ == "__main__":
    print("TERAG Main Interface - Test")
    print("=" * 60)

    # Test with simple chunks
    test_chunks = [
        {
            "content": "Apple Inc announced strong revenue growth in Q4 2024, reaching $120 billion.",
            "metadata": {"source": "financial_report", "page": 1}
        },
        {
            "content": "Microsoft Corporation reported significant achievements in cloud computing.",
            "metadata": {"source": "financial_report", "page": 2}
        },
        {
            "content": "The technology sector saw increased competition between Apple and Microsoft.",
            "metadata": {"source": "market_analysis", "page": 1}
        },
        {
            "content": "Q4 2024 was a strong quarter for major tech companies with record revenues.",
            "metadata": {"source": "market_analysis", "page": 2}
        }
    ]

    # Build TERAG
    config = TERAGConfig(
        min_concept_freq=1,
        top_k=3
    )

    terag = TERAG.from_chunks(test_chunks, config=config, verbose=True)

    # Test queries
    test_queries = [
        "What was Apple's revenue in Q4 2024?",
        "How did Microsoft perform?",
        "Which companies competed in technology?"
    ]

    print("\n" + "=" * 60)
    print("Testing Retrieval")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        results, metrics = terag.retrieve(query, verbose=True)

        print("\nTop Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Score: {result.score:.4f}]")
            print(f"   {result.content}")
            print(f"   Matched concepts: {result.matched_concepts}")

        print(f"\nMetrics: Retrieved {metrics.num_results} results in {metrics.retrieval_time:.3f}s")
        print("=" * 60)
