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
import time

from terag.config import TERAGConfig
from terag.graph.builder import TERAGGraph, GraphBuilder
from terag.ingestion.ner_extractor import NERExtractor
from terag.ingestion.query_ner import ImprovedQueryNER
from terag.retrieval.ppr import TERAGRetriever as PPRRetriever, RetrievalResult, RetrievalMetrics

logger = logging.getLogger(__name__)


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
                        api_key=api_key,
                        model=config.llm_model
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
            cache_dir=config.extraction_cache_dir,
            use_llm=config.use_llm_for_ner, 
            provider=config.llm_provider,
            api_key=config.llm_api_key,
            model=config.llm_model
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
    def empty(
        cls,
        config: Optional[TERAGConfig] = None,
        embedding_model: Optional[object] = None
    ) -> 'TERAG':
        """Create an empty TERAG instance that can receive chunks later."""
        if config is None:
            config = TERAGConfig()
        return cls(graph=TERAGGraph(), config=config, embedding_model=embedding_model)

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
            verbose: Log progress when application logging is configured

        Returns:
            TERAG system ready for retrieval
        """
        if config is None:
            config = TERAGConfig()

        if verbose:
            logger.info("Building TERAG system from %s chunks", len(chunks))
            logger.info(
                "LLM-based NER: %s",
                "enabled" if config.use_llm_for_ner else "disabled (using regex fallback)",
            )
            if config.use_llm_for_ner:
                logger.info("LLM provider: %s", config.llm_provider)
                if config.llm_model:
                    logger.info("LLM model: %s", config.llm_model)
                
                # Check for appropriate API key
                import os
                key_name = f"{config.llm_provider.upper()}_API_KEY"
                has_key = config.llm_api_key or os.getenv(key_name)
                
                if has_key:
                    logger.info(
                        "API key found via %s",
                        "config override" if config.llm_api_key else key_name,
                    )
                else:
                    logger.info("API key not found; regex fallback will be used")

        # Auto-create EmbeddingManager if not provided and OPENAI_API_KEY is available
        if embedding_model is None and config.use_semantic_entity_matching:
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    from terag.embeddings.manager import EmbeddingManager
                    embedding_model = EmbeddingManager(api_key=openai_key)
                    if verbose:
                        logger.info(
                            "Semantic matching enabled with auto-created EmbeddingManager "
                            "(model=text-embedding-3-small threshold=%s)",
                            config.semantic_match_threshold,
                        )
                except ImportError:
                    if verbose:
                        logger.info("Semantic matching disabled; EmbeddingManager is not available")
            else:
                if verbose:
                    logger.info("Semantic matching disabled; OPENAI_API_KEY was not found")
        elif embedding_model is not None:
            if verbose:
                logger.info(
                    "Semantic matching enabled with provided embedding model (threshold=%s)",
                    config.semantic_match_threshold,
                )
        else:
            if verbose:
                logger.info("Semantic matching disabled by config")

        start_time = time.time()

        # Build graph
        builder = GraphBuilder(
            min_concept_freq=config.min_concept_freq,
            max_concept_freq_ratio=config.max_concept_freq_ratio,
            enable_concept_clustering=config.enable_concept_clustering
        )

        graph_ner_extractor = NERExtractor(
            cache_dir=config.extraction_cache_dir,
            use_llm=config.use_llm_for_ner,
            provider=config.llm_provider,
            api_key=config.llm_api_key,
            model=config.llm_model,
            enable_progress_reporting=verbose
        )

        graph = builder.build_graph_from_chunks(
            chunks=chunks,
            extract_concepts_fn=graph_ner_extractor.extract_entities_and_concepts,
            verbose=verbose
        )

        build_time = time.time() - start_time

        if verbose:
            stats = graph.get_statistics()
            logger.info(
                "Graph built in %.2fs: passages=%s concepts=%s edges=%s",
                build_time,
                stats['num_passages'],
                stats['num_concepts'],
                stats['num_edges'],
            )

        # Create TERAG system
        terag = cls(graph=graph, config=config, embedding_model=embedding_model)
        
        # Auto-save graph if configured
        if config.auto_save_graph and config.graph_save_path:
            save_path = config.graph_save_path
            terag.save_graph(save_path)
            if verbose:
                logger.info("Graph auto-saved to %s", save_path)

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
            verbose: Log progress when application logging is configured

        Returns:
            TERAG system
        """
        if verbose:
            logger.info("Loading chunks from %s", chunks_file)

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
            verbose: Log progress when application logging is configured

        Returns:
            TERAG system
        """
        if verbose:
            logger.info("Loading graph from %s", graph_file)

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
            logger.info(
                "Loaded graph: passages=%s concepts=%s edges=%s",
                stats['num_passages'],
                stats['num_concepts'],
                stats['num_edges'],
            )
            if saved_embeddings:
                logger.info("Loaded %s concept embeddings from graph file", len(saved_embeddings))

        # Auto-create EmbeddingManager if not provided and OPENAI_API_KEY is available
        if embedding_model is None and config.use_semantic_entity_matching:
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    from terag.embeddings.manager import EmbeddingManager
                    embedding_model = EmbeddingManager(api_key=openai_key)
                    if verbose:
                        logger.info(
                            "Semantic matching enabled with auto-created EmbeddingManager "
                            "(model=text-embedding-3-small threshold=%s)",
                            config.semantic_match_threshold,
                        )
                except ImportError:
                    if verbose:
                        logger.info("Semantic matching disabled; EmbeddingManager is not available")
            else:
                if verbose:
                    logger.info("Semantic matching disabled; OPENAI_API_KEY was not found")
        elif embedding_model is not None:
            if verbose:
                logger.info(
                    "Semantic matching enabled with provided embedding model (threshold=%s)",
                    config.semantic_match_threshold,
                )
        else:
            if verbose:
                logger.info("Semantic matching disabled by config")

        # Create TERAG instance
        terag = cls(graph=graph, config=config, embedding_model=embedding_model)
        
        # Restore saved embeddings to retriever if available
        if saved_embeddings and hasattr(terag, 'retriever'):
            terag.retriever.concept_embeddings = saved_embeddings
            if verbose:
                logger.info("Restored %s concept embeddings to retriever", len(saved_embeddings))
        
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
            verbose: Log progress when application logging is configured

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
            logger.info("Using %s retrieval method", method.upper())
        
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

    def query(
        self,
        query: str,
        method: Optional[str] = None,
        top_k: Optional[int] = None,
        return_metrics: bool = False,
        **kwargs
    ):
        """
        Query TERAG and return retrieval results by default.

        This is the simple user-facing API. Use ``retrieve()`` when you need
        backward-compatible ``(results, metrics)`` tuple behavior.
        """
        results, metrics = self.retrieve(
            query=query,
            method=method,
            top_k=top_k,
            **kwargs
        )
        if return_metrics:
            return results, metrics
        return results

    def add_chunks(
        self,
        chunks: List[Dict],
        rebuild: bool = True,
        verbose: bool = True
    ) -> 'TERAG':
        """
        Add chunks to the index.

        Current implementation rebuilds the in-memory graph from existing and
        new chunks. This keeps behavior correct while preserving a future path
        for true incremental graph updates.
        """
        if not rebuild:
            raise NotImplementedError(
                "Incremental graph updates are not implemented yet. "
                "Call add_chunks(..., rebuild=True) to rebuild the graph."
            )

        existing_chunks = [
            {"content": passage.content, "metadata": passage.metadata}
            for passage in self.graph.passages.values()
        ]
        updated = self.from_chunks(
            existing_chunks + chunks,
            config=self.config,
            embedding_model=self.embedding_model,
            verbose=verbose
        )
        self.graph = updated.graph
        self.retriever = updated.retriever
        self.query_ner = updated.query_ner
        self.ner_extractor = updated.ner_extractor
        self._semantic_retriever = None
        self._hybrid_retriever = None
        return self

    def add_documents(
        self,
        documents: List,
        rebuild: bool = True,
        verbose: bool = True
    ) -> 'TERAG':
        """
        Add plain strings or document-like objects to the index.

        Supported inputs:
        - string documents
        - dicts with ``content`` or ``text`` and optional ``metadata``
        - objects with ``page_content`` and optional ``metadata`` attributes
        """
        chunks = [self._document_to_chunk(document, i) for i, document in enumerate(documents)]
        return self.add_chunks(chunks, rebuild=rebuild, verbose=verbose)

    insert = add_documents

    @staticmethod
    def _document_to_chunk(document, index: int) -> Dict:
        """Convert common document shapes into TERAG chunk dictionaries."""
        if isinstance(document, str):
            return {"content": document, "metadata": {"document_index": index}}

        if isinstance(document, dict):
            if "content" in document:
                return {
                    "content": document["content"],
                    "metadata": document.get("metadata", {})
                }
            if "text" in document:
                return {
                    "content": document["text"],
                    "metadata": document.get("metadata", {})
                }
            raise ValueError("Document dict must include 'content' or 'text'.")

        page_content = getattr(document, "page_content", None)
        if page_content is not None:
            return {
                "content": page_content,
                "metadata": getattr(document, "metadata", {}) or {}
            }

        raise TypeError(
            "Documents must be strings, dicts with 'content' or 'text', "
            "or objects with a 'page_content' attribute."
        )
    
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
                logger.info(
                    "Initialized semantic retriever with %s passages",
                    len(self.graph.passages),
                )
        
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
                logger.info(
                    "Initialized hybrid retriever (PPR: %s, Semantic: %s)",
                    ppr_weight,
                    semantic_weight,
                )
        
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
            logger.info(
                "Hybrid retrieval analysis: total=%s ppr_only=%s semantic_only=%s combined=%s",
                analysis.total_results,
                analysis.ppr_only_results,
                analysis.semantic_only_results,
                analysis.combined_results,
            )
        
        return results, metrics

    def save_graph(self, filepath: str):
        """Save graph to file for reuse, including concept embeddings if available"""
        import os
        
        # Get concept embeddings from retriever if available
        concept_embeddings = None
        if hasattr(self, 'retriever') and hasattr(self.retriever, 'concept_embeddings'):
            if self.retriever.concept_embeddings:
                concept_embeddings = self.retriever.concept_embeddings
                logger.info(
                    "Saving %s concept embeddings with graph",
                    len(concept_embeddings),
                )
        
        self.graph.save_to_file(filepath, concept_embeddings=concept_embeddings)
        abs_path = os.path.abspath(filepath)
        logger.info("Graph saved to %s", abs_path)

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
        logger.info("Loading embedding model: %s", embedding_model_name)
        embedding_model = SentenceTransformer(embedding_model_name)
    except ImportError:
        logger.warning("sentence-transformers not available; semantic weighting disabled")

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
