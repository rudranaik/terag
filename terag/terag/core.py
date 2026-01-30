"""
TERAG Main Retriever Interface

Complete TERAG retrieval system integrating:
1. Graph construction from chunks
2. Named Entity Recognition
3. Personalized PageRank retrieval
4. Result ranking and formatting
"""

import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

from terag.graph.builder import TERAGGraph, GraphBuilder
from terag.ingestion.ner_extractor import NERExtractor, extract_concepts_from_text
from terag.ingestion.query_ner import ImprovedQueryNER
from terag.retrieval.ppr import TERAGRetriever as PPRRetriever, RetrievalResult, RetrievalMetrics

logger = logging.getLogger(__name__)


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
    default_retrieval_method: str = "ppr"  # "ppr", "semantic", or "hybrid"

    # Semantic entity matching
    use_semantic_entity_matching: bool = True  # Enable semantic similarity-based entity matching
    semantic_match_threshold: float = 0.7  # Cosine similarity threshold for semantic matching
    
    # Optional LLM for enhanced NER
    use_llm_for_ner: bool = False
    llm_provider: str = "groq"  # "groq" or "openai"
    llm_api_key: Optional[str] = None  # Override env var
    
    # Graph persistence
    auto_save_graph: bool = False
    graph_save_path: Optional[str] = "terag_graph.json"  # Default save location


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

        # Initialize LLM-based query NER with better error handling
        self.query_ner = None
        if config.use_llm_for_ner:
            try:
                api_key = config.llm_api_key or os.getenv(
                    "GROQ_API_KEY" if config.llm_provider == "groq" else "OPENAI_API_KEY"
                )
                
                if not api_key:
                    logger.warning(
                        f"LLM-based NER requested but no API key found. "
                        f"Please set {config.llm_provider.upper()}_API_KEY environment variable "
                        f"or pass llm_api_key in config. Falling back to regex-based NER."
                    )
                    self.query_ner = None
                else:
                    self.query_ner = ImprovedQueryNER(
                        use_llm=True,
                        provider=config.llm_provider,
                        api_key=api_key
                    )
                    logger.info(f"Initialized LLM-based query NER using {config.llm_provider}")
                    
            except Exception as e:
                logger.warning(
                    f"Failed to initialize LLM-based NER: {e}. "
                    f"Falling back to regex-based NER."
                )
                self.query_ner = None
        
        # If LLM NER failed or not requested, use regex-based fallback
        if self.query_ner is None:
            self.query_ner = ImprovedQueryNER(
                use_llm=False,
                provider=config.llm_provider,
                api_key=None
            )

        # Initialize NER extractor (for graph building)
        self.ner_extractor = NERExtractor(
            use_llm=config.use_llm_for_ner, 
            provider=config.llm_provider,
            api_key=config.llm_api_key
        )
        
        # Initialize PPR retriever
        self.retriever = PPRRetriever(
            graph=graph,
            embedding_model=embedding_model,
            alpha=config.ppr_alpha,
            ppr_max_iterations=config.ppr_max_iterations,
            use_semantic_matching=config.use_semantic_entity_matching,
            semantic_threshold=config.semantic_match_threshold
        )
        
        # Lazy initialization for semantic and hybrid retrievers
        self._semantic_retriever = None
        self._hybrid_retriever = None

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
                           If None and OPENAI_API_KEY is available, will auto-create EmbeddingManager
            verbose: Print progress

        Returns:
            TERAG system ready for retrieval
        """
        if config is None:
            config = TERAGConfig()

        if verbose:
            print("Building TERAG system from chunks...")
            print(f"  Chunks: {len(chunks)}")
            print(f"  LLM-based NER: {'ENABLED' if config.use_llm_for_ner else 'DISABLED (using regex fallback)'}")
            if config.use_llm_for_ner:
                print(f"  Provider: {config.llm_provider}")
                
                # Check for appropriate API key
                import os
                key_name = f"{config.llm_provider.upper()}_API_KEY"
                has_key = config.llm_api_key or os.getenv(key_name)
                
                if has_key:
                    print(f"  API Key: Found ({'config override' if config.llm_api_key else key_name})")
                else:
                    print(f"  API Key: Not found (will use regex fallback)")

        # Auto-create EmbeddingManager if not provided and OPENAI_API_KEY is available
        if embedding_model is None and config.use_semantic_entity_matching:
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    from terag.embeddings.manager import EmbeddingManager
                    embedding_model = EmbeddingManager(api_key=openai_key)
                    if verbose:
                        print(f"  Semantic Matching: ENABLED (auto-created EmbeddingManager)")
                        print(f"    Model: text-embedding-3-small")
                        print(f"    Threshold: {config.semantic_match_threshold}")
                except ImportError:
                    if verbose:
                        print(f"  Semantic Matching: DISABLED (EmbeddingManager not available)")
            else:
                if verbose:
                    print(f"  Semantic Matching: DISABLED (no OPENAI_API_KEY found)")
        elif embedding_model is not None:
            if verbose:
                print(f"  Semantic Matching: ENABLED (using provided embedding model)")
                print(f"    Threshold: {config.semantic_match_threshold}")
        else:
            if verbose:
                print(f"  Semantic Matching: DISABLED (use_semantic_entity_matching=False)")

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
        
        # Auto-save graph if configured
        if config.auto_save_graph and config.graph_save_path:
            save_path = config.graph_save_path
            terag.save_graph(save_path)
            if verbose:
                print(f"\n✓ Graph auto-saved to: {save_path}")
                print(f"  To load: TERAG.from_graph_file('{save_path}')")

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
                           If None and OPENAI_API_KEY is available, will auto-create EmbeddingManager
            verbose: Print progress

        Returns:
            TERAG system
        """
        if verbose:
            print(f"Loading graph from {graph_file}...")

        # Load graph and embeddings
        graph_data = TERAGGraph.load_from_file(graph_file)
        
        # Handle both old format (just graph) and new format (graph, embeddings)
        if isinstance(graph_data, tuple):
            graph, saved_embeddings = graph_data
        else:
            graph = graph_data
            saved_embeddings = None

        if config is None:
            config = TERAGConfig()

        if verbose:
            stats = graph.get_statistics()
            print(f"  Passages: {stats['num_passages']}")
            print(f"  Concepts: {stats['num_concepts']}")
            print(f"  Edges: {stats['num_edges']}")
            if saved_embeddings:
                print(f"  Loaded {len(saved_embeddings)} concept embeddings from graph file")

        # Auto-create EmbeddingManager if not provided and OPENAI_API_KEY is available
        if embedding_model is None and config.use_semantic_entity_matching:
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    from terag.embeddings.manager import EmbeddingManager
                    embedding_model = EmbeddingManager(api_key=openai_key)
                    if verbose:
                        print(f"  Semantic Matching: ENABLED (auto-created EmbeddingManager)")
                        print(f"    Model: text-embedding-3-small")
                        print(f"    Threshold: {config.semantic_match_threshold}")
                except ImportError:
                    if verbose:
                        print(f"  Semantic Matching: DISABLED (EmbeddingManager not available)")
            else:
                if verbose:
                    print(f"  Semantic Matching: DISABLED (no OPENAI_API_KEY found)")
        elif embedding_model is not None:
            if verbose:
                print(f"  Semantic Matching: ENABLED (using provided embedding model)")
                print(f"    Threshold: {config.semantic_match_threshold}")
        else:
            if verbose:
                print(f"  Semantic Matching: DISABLED (use_semantic_entity_matching=False)")

        # Create TERAG instance
        terag = cls(graph=graph, config=config, embedding_model=embedding_model)
        
        # Restore saved embeddings to retriever if available
        if saved_embeddings and hasattr(terag, 'retriever'):
            terag.retriever.concept_embeddings = saved_embeddings
            if verbose:
                print(f"  ✓ Restored {len(saved_embeddings)} concept embeddings to retriever")
        
        return terag

    def retrieve(
        self,
        query: str,
        method: Optional[str] = None,
        top_k: Optional[int] = None,
        ppr_weight: float = 0.6,
        semantic_weight: float = 0.4,
        min_score_threshold: Optional[float] = None,
        verbose: bool = False
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Retrieve relevant passages for a query using specified method

        Args:
            query: User query
            method: Retrieval method - "ppr", "semantic", or "hybrid" (default: from config)
            top_k: Number of results (default: from config)
            ppr_weight: Weight for PPR scores in hybrid retrieval (default: 0.6)
            semantic_weight: Weight for semantic scores in hybrid retrieval (default: 0.4)
            min_score_threshold: Minimum score threshold (method-specific)
            verbose: Print progress

        Returns:
            (results, metrics) tuple
            
        Examples:
            # PPR retrieval (default)
            results, metrics = terag.retrieve("What is the revenue?")
            
            # Semantic retrieval
            results, metrics = terag.retrieve("What is the revenue?", method="semantic")
            
            # Hybrid retrieval
            results, metrics = terag.retrieve(
                "What is the revenue?", 
                method="hybrid",
                ppr_weight=0.7,
                semantic_weight=0.3
            )
        """
        if top_k is None:
            top_k = self.config.top_k
            
        if method is None:
            method = self.config.default_retrieval_method
        
        # Validate method
        self._validate_retrieval_method(method)
        
        if verbose:
            print(f"Using {method.upper()} retrieval method")
        
        # Route to appropriate retrieval method
        if method == "ppr":
            return self._retrieve_ppr(query, top_k, min_score_threshold, verbose)
        elif method == "semantic":
            return self._retrieve_semantic(query, top_k, min_score_threshold, verbose)
        elif method == "hybrid":
            return self._retrieve_hybrid(
                query, top_k, ppr_weight, semantic_weight, min_score_threshold, verbose
            )
        else:
            raise ValueError(f"Invalid retrieval method: {method}")
    
    def _validate_retrieval_method(self, method: str) -> None:
        """Validate retrieval method and check prerequisites"""
        valid_methods = ["ppr", "semantic", "hybrid"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid retrieval method '{method}'. "
                f"Must be one of: {', '.join(valid_methods)}"
            )
        
        if method in ["semantic", "hybrid"] and self.embedding_model is None:
            raise ValueError(
                f"Retrieval method '{method}' requires an embedding model. "
                f"Please provide embedding_model when creating TERAG instance."
            )
    
    def _retrieve_ppr(
        self,
        query: str,
        top_k: int,
        min_score_threshold: Optional[float],
        verbose: bool
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """PPR retrieval implementation"""
        # Extract query entities
        query_entities = self.query_ner.extract_query_entities(query, verbose=verbose)

        # Retrieve using PPR
        results, metrics = self.retriever.retrieve(
            query=query,
            query_entities=query_entities,
            top_k=top_k,
            semantic_weight=self.config.semantic_weight,
            frequency_weight=self.config.frequency_weight,
            verbose=verbose
        )
        
        # Apply threshold if specified
        if min_score_threshold is not None:
            results = [r for r in results if r.score >= min_score_threshold]

        return results, metrics
    
    def _retrieve_semantic(
        self,
        query: str,
        top_k: int,
        min_score_threshold: Optional[float],
        verbose: bool
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Semantic retrieval implementation"""
        # Lazy initialize semantic retriever
        if self._semantic_retriever is None:
            from terag.retrieval.semantic import SemanticRetriever
            from terag.embeddings.manager import EmbeddingManager
            
            # Ensure we have an embedding manager
            if not isinstance(self.embedding_model, EmbeddingManager):
                # Wrap in EmbeddingManager if needed
                embedding_manager = EmbeddingManager()
                embedding_manager.model = self.embedding_model
            else:
                embedding_manager = self.embedding_model
            
            self._semantic_retriever = SemanticRetriever(
                embedding_manager=embedding_manager,
                min_similarity_threshold=min_score_threshold or 0.5
            )
            self._semantic_retriever.load_passage_embeddings(self.graph)
            
            if verbose:
                print(f"Initialized semantic retriever with {len(self.graph.passages)} passages")
        
        # Process query
        from terag.retrieval.query_processor import QueryProcessor
        query_processor = QueryProcessor(embedding_manager=self._semantic_retriever.embedding_manager)
        query_processor.load_graph_entities(self.graph)
        processed_query = query_processor.process_query(query)
        
        # Retrieve
        start_time = time.time()
        passages = self._semantic_retriever.retrieve_passages(
            processed_query=processed_query,
            top_k=top_k
        )
        retrieval_time = time.time() - start_time
        
        # Convert to RetrievalResult format
        results = []
        for passage_id, score in passages:
            passage = self.graph.passages.get(passage_id)
            if passage:
                results.append(RetrievalResult(
                    passage_id=passage_id,
                    content=passage.content,
                    score=score,
                    matched_concepts=[],  # Semantic doesn't use concept matching
                    metadata=passage.metadata
                ))
        
        metrics = RetrievalMetrics(
            num_query_entities=0,
            num_matched_concepts=0,
            ppr_iterations=0,  # Not applicable for semantic retrieval
            retrieval_time=retrieval_time,
            num_results=len(results)
        )
        
        return results, metrics
    
    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        ppr_weight: float,
        semantic_weight: float,
        min_score_threshold: Optional[float],
        verbose: bool
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Hybrid retrieval implementation"""
        # Lazy initialize hybrid retriever
        if self._hybrid_retriever is None:
            from terag.retrieval.hybrid import HybridRetriever
            from terag.embeddings.manager import EmbeddingManager
            
            # Ensure we have an embedding manager
            if not isinstance(self.embedding_model, EmbeddingManager):
                embedding_manager = EmbeddingManager()
                embedding_manager.model = self.embedding_model
            else:
                embedding_manager = self.embedding_model
            
            self._hybrid_retriever = HybridRetriever(
                graph=self.graph,
                embedding_manager=embedding_manager,
                ppr_weight=ppr_weight,
                semantic_weight=semantic_weight,
                use_llm_for_ner=self.config.use_llm_for_ner,
                llm_provider=self.config.llm_provider,
                llm_api_key=self.config.llm_api_key
            )
            
            if verbose:
                print(f"Initialized hybrid retriever (PPR: {ppr_weight}, Semantic: {semantic_weight})")
        
        # Retrieve
        start_time = time.time()
        hybrid_results, analysis = self._hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            enable_analysis=verbose
        )
        retrieval_time = time.time() - start_time
        
        # Apply threshold if specified
        if min_score_threshold is not None:
            hybrid_results = [r for r in hybrid_results if r.hybrid_score >= min_score_threshold]
        
        # Convert to RetrievalResult format
        results = []
        for hr in hybrid_results:
            results.append(RetrievalResult(
                passage_id=hr.passage_id,
                content=hr.content,
                score=hr.hybrid_score,
                matched_concepts=hr.entity_matches or [],
                metadata=hr.metadata or {}
            ))
        
        metrics = RetrievalMetrics(
            num_query_entities=0,
            num_matched_concepts=len(set(sum([r.entity_matches or [] for r in hybrid_results], []))),
            ppr_iterations=0,  # Not directly tracked in hybrid mode
            retrieval_time=retrieval_time,
            num_results=len(results)
        )
        
        if verbose and analysis:
            print(f"\nHybrid Retrieval Analysis:")
            print(f"  Total results: {analysis.total_results}")
            print(f"  PPR-only: {analysis.ppr_only_results}")
            print(f"  Semantic-only: {analysis.semantic_only_results}")
            print(f"  Combined: {analysis.combined_results}")
        
        return results, metrics

    def save_graph(self, filepath: str):
        """Save graph to file for reuse, including concept embeddings if available"""
        import os
        
        # Get concept embeddings from retriever if available
        concept_embeddings = None
        if hasattr(self, 'retriever') and hasattr(self.retriever, 'concept_embeddings'):
            if self.retriever.concept_embeddings:
                concept_embeddings = self.retriever.concept_embeddings
                print(f"  Saving {len(concept_embeddings)} concept embeddings with graph")
        
        self.graph.save_to_file(filepath, concept_embeddings=concept_embeddings)
        abs_path = os.path.abspath(filepath)
        print(f"✓ Graph saved to: {abs_path}")

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
