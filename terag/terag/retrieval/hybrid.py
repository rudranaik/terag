"""
TERAG Hybrid Retriever

Combines Entity-based PPR retrieval and Direct semantic retrieval using
intelligent scoring fusion. Provides the best of both graph-structured 
reasoning and semantic similarity matching.
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from .query_processor import ProcessedQuery, QueryProcessor
from .ppr import TERAGRetriever as PPRRetriever
from .semantic import SemanticRetriever
from terag.embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrievalResult:
    """Combined result from hybrid retrieval"""
    passage_id: str
    content: str
    hybrid_score: float
    ppr_score: float = 0.0
    semantic_score: float = 0.0
    entity_matches: List[str] = None
    confidence: float = 0.0
    explanation: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.entity_matches is None:
            self.entity_matches = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalAnalysis:
    """Analysis of retrieval performance and coverage"""
    query: str
    total_results: int
    ppr_only_results: int
    semantic_only_results: int
    combined_results: int
    avg_ppr_score: float
    avg_semantic_score: float
    coverage_overlap: float
    query_confidence: float


class HybridRetriever:
    """
    Hybrid retrieval system combining entity-based PPR and semantic similarity
    """
    
    def __init__(
        self,
        graph,
        embedding_manager: EmbeddingManager,
        ppr_weight: float = 0.6,
        semantic_weight: float = 0.4,
        score_fusion_method: str = "weighted_sum",  # "weighted_sum", "max", "harmonic_mean"
        min_ppr_score: float = 1e-6,
        min_semantic_score: float = 0.3,
        result_diversification: bool = True
    ):
        """
        Initialize hybrid retriever
        
        Args:
            graph: TERAGGraph instance
            embedding_manager: Pre-configured embedding manager
            ppr_weight: Weight for PPR scores in fusion
            semantic_weight: Weight for semantic scores in fusion
            score_fusion_method: Method to combine scores
            min_ppr_score: Minimum PPR score threshold
            min_semantic_score: Minimum semantic score threshold
            result_diversification: Whether to diversify results
        """
        self.graph = graph
        self.embedding_manager = embedding_manager
        self.ppr_weight = ppr_weight
        self.semantic_weight = semantic_weight
        self.score_fusion_method = score_fusion_method
        self.min_ppr_score = min_ppr_score
        self.min_semantic_score = min_semantic_score
        self.result_diversification = result_diversification
        
        # Initialize components
        logger.info("Initializing hybrid retrieval components...")
        
        # Query processor
        self.query_processor = QueryProcessor(
            embedding_manager=embedding_manager,
            entity_similarity_threshold=0.7
        )
        self.query_processor.load_graph_entities(graph)
        
        # PPR retriever (using existing implementation)
        self.ppr_retriever = PPRRetriever(graph, alpha=0.15)
        
        # Semantic retriever
        self.semantic_retriever = SemanticRetriever(
            embedding_manager=embedding_manager,
            min_similarity_threshold=min_semantic_score
        )
        self.semantic_retriever.load_passage_embeddings(graph)
        
        # Statistics
        self.retrieval_stats = {
            "queries_processed": 0,
            "avg_ppr_results": 0,
            "avg_semantic_results": 0,
            "avg_combined_results": 0,
            "fusion_method_used": score_fusion_method
        }
        
        logger.info("Hybrid retriever initialized successfully")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        ppr_top_k: int = 30,
        semantic_top_k: int = 30,
        enable_analysis: bool = False
    ) -> Tuple[List[HybridRetrievalResult], Optional[RetrievalAnalysis]]:
        """
        Perform hybrid retrieval combining PPR and semantic approaches
        
        Args:
            query: User query string
            top_k: Final number of results to return
            ppr_top_k: Number of results from PPR retrieval
            semantic_top_k: Number of results from semantic retrieval
            enable_analysis: Whether to return detailed analysis
            
        Returns:
            (hybrid_results, analysis) tuple
        """
        logger.info(f"Hybrid retrieval for query: '{query}'")
        
        # Step 1: Process query
        processed_query = self.query_processor.process_query(query)
        logger.info(f"Query processing: {len(processed_query.extracted_entities)} entities extracted, "
                   f"confidence: {processed_query.confidence_score:.3f}")
        
        # Step 2: Run PPR retrieval
        ppr_results = []
        if processed_query.extracted_entities:
            query_entities = [entity.entity_text for entity in processed_query.extracted_entities]
            ppr_results, ppr_metrics = self.ppr_retriever.retrieve(
                query=query,
                query_entities=query_entities,
                top_k=ppr_top_k,
                verbose=False
            )
            logger.info(f"PPR retrieval: {len(ppr_results)} results")
        else:
            logger.info("No entities extracted, skipping PPR retrieval")
        
        # Step 3: Run semantic retrieval
        semantic_results = self.semantic_retriever.retrieve_passages(
            processed_query=processed_query,
            top_k=semantic_top_k
        )
        logger.info(f"Semantic retrieval: {len(semantic_results)} results")
        
        # Step 4: Combine and score results
        hybrid_results = self._fuse_results(
            query=query,
            processed_query=processed_query,
            ppr_results=ppr_results,
            semantic_results=semantic_results,
            top_k=top_k
        )
        
        logger.info(f"Hybrid fusion: {len(hybrid_results)} final results")
        
        # Step 5: Optional analysis
        analysis = None
        if enable_analysis:
            analysis = self._analyze_retrieval(
                query=query,
                processed_query=processed_query,
                ppr_results=ppr_results,
                semantic_results=semantic_results,
                hybrid_results=hybrid_results
            )
        
        # Update statistics
        self.retrieval_stats["queries_processed"] += 1
        self.retrieval_stats["avg_ppr_results"] = self._update_avg(
            self.retrieval_stats["avg_ppr_results"], len(ppr_results)
        )
        self.retrieval_stats["avg_semantic_results"] = self._update_avg(
            self.retrieval_stats["avg_semantic_results"], len(semantic_results)
        )
        self.retrieval_stats["avg_combined_results"] = self._update_avg(
            self.retrieval_stats["avg_combined_results"], len(hybrid_results)
        )
        
        return hybrid_results, analysis
    
    def _fuse_results(
        self,
        query: str,
        processed_query: ProcessedQuery,
        ppr_results: List,
        semantic_results: List[Tuple[str, float]],
        top_k: int
    ) -> List[HybridRetrievalResult]:
        """
        Fuse PPR and semantic results using configurable fusion method
        
        Args:
            query: Original query
            processed_query: Processed query object
            ppr_results: Results from PPR retrieval
            semantic_results: Results from semantic retrieval
            top_k: Number of final results
            
        Returns:
            List of fused HybridRetrievalResult objects
        """
        # Collect all passage scores
        ppr_scores = {}
        semantic_scores = {}
        
        # Extract PPR scores
        for result in ppr_results:
            passage_id = result.passage_id if hasattr(result, 'passage_id') else result[0]
            score = result.score if hasattr(result, 'score') else result[1]
            if score >= self.min_ppr_score:
                ppr_scores[passage_id] = score
        
        # Extract semantic scores
        for passage_id, score in semantic_results:
            if score >= self.min_semantic_score:
                semantic_scores[passage_id] = score
        
        # Get all unique passages
        all_passage_ids = set(ppr_scores.keys()) | set(semantic_scores.keys())
        
        logger.debug(f"Fusion input: {len(ppr_scores)} PPR passages, {len(semantic_scores)} semantic passages, "
                    f"{len(all_passage_ids)} unique passages")
        
        # Calculate hybrid scores
        hybrid_scores = []
        
        for passage_id in all_passage_ids:
            ppr_score = ppr_scores.get(passage_id, 0.0)
            semantic_score = semantic_scores.get(passage_id, 0.0)
            
            # Apply fusion method
            if self.score_fusion_method == "weighted_sum":
                hybrid_score = (self.ppr_weight * ppr_score + 
                               self.semantic_weight * semantic_score)
            elif self.score_fusion_method == "max":
                hybrid_score = max(ppr_score * self.ppr_weight, 
                                 semantic_score * self.semantic_weight)
            elif self.score_fusion_method == "harmonic_mean":
                if ppr_score > 0 and semantic_score > 0:
                    weighted_ppr = ppr_score * self.ppr_weight
                    weighted_semantic = semantic_score * self.semantic_weight
                    hybrid_score = 2 * weighted_ppr * weighted_semantic / (weighted_ppr + weighted_semantic)
                else:
                    hybrid_score = (self.ppr_weight * ppr_score + 
                                   self.semantic_weight * semantic_score)
            else:
                # Default to weighted sum
                hybrid_score = (self.ppr_weight * ppr_score + 
                               self.semantic_weight * semantic_score)
            
            # Calculate confidence based on score sources
            confidence = self._calculate_confidence(ppr_score, semantic_score, processed_query)
            
            hybrid_scores.append((passage_id, hybrid_score, ppr_score, semantic_score, confidence))
        
        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply result diversification if enabled
        if self.result_diversification:
            hybrid_scores = self._diversify_results(hybrid_scores, top_k)
        else:
            hybrid_scores = hybrid_scores[:top_k]
        
        # Create HybridRetrievalResult objects
        final_results = []
        for passage_id, hybrid_score, ppr_score, semantic_score, confidence in hybrid_scores:
            
            # Get passage content and metadata
            passage = self.graph.passages.get(passage_id)
            if not passage:
                continue
            
            # Find entity matches for this passage
            entity_matches = []
            if processed_query.extracted_entities:
                passage_concepts = self.graph.get_passage_neighbors(passage_id)
                for entity in processed_query.extracted_entities:
                    if entity.graph_concept_id in passage_concepts:
                        entity_matches.append(entity.entity_text)
            
            # Generate explanation
            explanation = self._generate_explanation(
                ppr_score, semantic_score, entity_matches, hybrid_score
            )
            
            result = HybridRetrievalResult(
                passage_id=passage_id,
                content=passage.content,
                hybrid_score=hybrid_score,
                ppr_score=ppr_score,
                semantic_score=semantic_score,
                entity_matches=entity_matches,
                confidence=confidence,
                explanation=explanation,
                metadata=passage.metadata
            )
            
            final_results.append(result)
        
        return final_results
    
    def _calculate_confidence(
        self, 
        ppr_score: float, 
        semantic_score: float,
        processed_query: ProcessedQuery
    ) -> float:
        """Calculate confidence score for a result"""
        
        # Base confidence from scores
        score_confidence = 0.0
        if ppr_score > 0 and semantic_score > 0:
            score_confidence = 0.9  # Both approaches agree
        elif ppr_score > 0:
            score_confidence = 0.7  # PPR only
        elif semantic_score > 0:
            score_confidence = 0.6  # Semantic only
        
        # Boost based on query confidence
        query_confidence_boost = processed_query.confidence_score * 0.2
        
        # Boost based on score magnitude
        score_magnitude = max(ppr_score, semantic_score)
        magnitude_boost = min(0.2, score_magnitude * 0.1)
        
        final_confidence = min(1.0, score_confidence + query_confidence_boost + magnitude_boost)
        return final_confidence
    
    def _generate_explanation(
        self,
        ppr_score: float,
        semantic_score: float, 
        entity_matches: List[str],
        hybrid_score: float
    ) -> str:
        """Generate human-readable explanation for why passage was retrieved"""
        
        explanations = []
        
        if ppr_score > 0 and semantic_score > 0:
            explanations.append(f"Strong match: graph connections (PPR: {ppr_score:.3f}) + semantic similarity ({semantic_score:.3f})")
        elif ppr_score > 0:
            explanations.append(f"Graph-based match through entity connections (PPR: {ppr_score:.3f})")
        elif semantic_score > 0:
            explanations.append(f"Semantic similarity match (similarity: {semantic_score:.3f})")
        
        if entity_matches:
            explanations.append(f"Directly mentions: {', '.join(entity_matches)}")
        
        if not explanations:
            explanations.append("Retrieved through weak connections")
        
        return " | ".join(explanations)
    
    def _diversify_results(
        self, 
        scored_results: List[Tuple], 
        target_k: int
    ) -> List[Tuple]:
        """Apply diversification to avoid redundant results"""
        
        if len(scored_results) <= target_k:
            return scored_results
        
        diversified = []
        content_seen = []  # Use list of sets instead of set of sets
        
        for result in scored_results:
            passage_id = result[0]
            
            # Get passage content for similarity check
            passage = self.graph.passages.get(passage_id)
            if not passage:
                continue
            
            # Simple diversification: avoid very similar content
            content_words = set(passage.content.lower().split()[:20])  # First 20 words
            
            # Check if this content is too similar to already selected
            is_similar = False
            for seen_words in content_seen:
                overlap = len(content_words & seen_words) / max(len(content_words), 1)
                if overlap > 0.7:  # 70% word overlap threshold
                    is_similar = True
                    break
            
            if not is_similar:
                diversified.append(result)
                content_seen.append(content_words)  # Append to list instead of add to set
                
                if len(diversified) >= target_k:
                    break
        
        # Fill remaining slots if diversification was too aggressive
        if len(diversified) < target_k:
            for result in scored_results:
                if result not in diversified:
                    diversified.append(result)
                    if len(diversified) >= target_k:
                        break
        
        return diversified
    
    def _analyze_retrieval(
        self,
        query: str,
        processed_query: ProcessedQuery,
        ppr_results: List,
        semantic_results: List[Tuple[str, float]],
        hybrid_results: List[HybridRetrievalResult]
    ) -> RetrievalAnalysis:
        """Analyze retrieval performance and coverage"""
        
        ppr_passage_ids = {r.passage_id if hasattr(r, 'passage_id') else r[0] for r in ppr_results}
        semantic_passage_ids = {passage_id for passage_id, _ in semantic_results}
        hybrid_passage_ids = {r.passage_id for r in hybrid_results}
        
        # Calculate overlaps
        ppr_only = len(ppr_passage_ids - semantic_passage_ids)
        semantic_only = len(semantic_passage_ids - ppr_passage_ids)
        combined = len(ppr_passage_ids & semantic_passage_ids)
        
        # Calculate average scores
        avg_ppr = sum(r.score if hasattr(r, 'score') else r[1] for r in ppr_results) / max(len(ppr_results), 1)
        avg_semantic = sum(score for _, score in semantic_results) / max(len(semantic_results), 1)
        
        # Coverage overlap
        all_unique = len(ppr_passage_ids | semantic_passage_ids)
        coverage_overlap = combined / max(all_unique, 1)
        
        analysis = RetrievalAnalysis(
            query=query,
            total_results=len(hybrid_results),
            ppr_only_results=ppr_only,
            semantic_only_results=semantic_only,
            combined_results=combined,
            avg_ppr_score=avg_ppr,
            avg_semantic_score=avg_semantic,
            coverage_overlap=coverage_overlap,
            query_confidence=processed_query.confidence_score
        )
        
        return analysis
    
    def _update_avg(self, current_avg: float, new_value: float) -> float:
        """Update running average"""
        n = self.retrieval_stats["queries_processed"]
        if n == 0:
            return new_value
        return (current_avg * n + new_value) / (n + 1)
    
    def get_retrieval_statistics(self) -> Dict:
        """Get comprehensive retrieval statistics"""
        stats = self.retrieval_stats.copy()
        stats.update({
            "ppr_weight": self.ppr_weight,
            "semantic_weight": self.semantic_weight,
            "fusion_method": self.score_fusion_method,
            "min_ppr_threshold": self.min_ppr_score,
            "min_semantic_threshold": self.min_semantic_score,
            "diversification_enabled": self.result_diversification,
            "query_processor_stats": self.query_processor.get_entity_matching_statistics(),
            "semantic_retriever_stats": self.semantic_retriever.get_retrieval_statistics()
        })
        return stats


# Convenience function
def create_hybrid_retriever(graph, embedding_manager: Optional[EmbeddingManager] = None, **kwargs) -> HybridRetriever:
    """
    Create and configure a hybrid retriever for a graph
    
    Args:
        graph: TERAGGraph instance
        embedding_manager: Optional pre-configured embedding manager
        **kwargs: Additional arguments for HybridRetriever
        
    Returns:
        Configured HybridRetriever
    """
    if embedding_manager is None:
        embedding_manager = EmbeddingManager()
    
    return HybridRetriever(graph, embedding_manager, **kwargs)


if __name__ == "__main__":
    # Test hybrid retriever
    print("🔄 HYBRID RETRIEVER TEST")
    print("=" * 60)
    
    print("💡 To test hybrid retriever:")
    print("   1. Load a TERAG graph")
    print("   2. Create embedding manager")
    print("   3. Initialize hybrid retriever")
    print("   4. Run retrieval on test queries")
    
    print(f"\n📋 Example usage:")
    print("""
    from hybrid_retriever import create_hybrid_retriever
    from embedding_manager import EmbeddingManager
    from graph_builder import TERAGGraph
    
    # Load graph and create retriever
    graph = TERAGGraph.load_from_file("graph.json")
    embedding_manager = EmbeddingManager()
    retriever = create_hybrid_retriever(graph, embedding_manager)
    
    # Perform retrieval
    results, analysis = retriever.retrieve(
        "How is the music business performing?", 
        top_k=10,
        enable_analysis=True
    )
    
    print(f"Retrieved {len(results)} results")
    for result in results[:3]:
        print(f"Score: {result.hybrid_score:.3f} - {result.explanation}")
    """)