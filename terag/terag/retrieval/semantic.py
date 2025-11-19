"""
TERAG Semantic Passage Retriever

Implements direct semantic similarity retrieval between queries and passages.
Uses OpenAI embeddings to find semantically similar content regardless of 
entity matches in the graph structure.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from terag.embeddings.manager import EmbeddingManager
from .query_processor import ProcessedQuery

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Direct semantic similarity retrieval for passages
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        min_similarity_threshold: float = 0.5,
        passage_length_boost: bool = True,
        normalize_scores: bool = True
    ):
        """
        Initialize semantic retriever
        
        Args:
            embedding_manager: Pre-configured embedding manager
            min_similarity_threshold: Minimum cosine similarity for results
            passage_length_boost: Boost scores for longer passages
            normalize_scores: Normalize similarity scores to [0,1] range
        """
        self.embedding_manager = embedding_manager
        self.min_similarity_threshold = min_similarity_threshold
        self.passage_length_boost = passage_length_boost
        self.normalize_scores = normalize_scores
        
        # Cache for passage embeddings
        self.passage_embeddings = {}
        self.passage_metadata = {}
        
        # Statistics
        self.retrieval_stats = {
            "queries_processed": 0,
            "total_passages_scored": 0,
            "avg_similarity_scores": [],
            "cache_hits": 0
        }
    
    def load_passage_embeddings(self, graph) -> int:
        """
        Load and embed all passages from the graph
        
        Args:
            graph: TERAGGraph instance
            
        Returns:
            Number of passages loaded
        """
        logger.info("Loading passage embeddings for semantic retrieval...")
        
        # Extract passage data
        passage_texts = []
        passage_ids = []
        
        for passage_id, passage in graph.passages.items():
            passage_texts.append(passage.content)
            passage_ids.append(passage_id)
            
            # Store passage metadata
            self.passage_metadata[passage_id] = {
                "content_length": len(passage.content),
                "word_count": len(passage.content.split()),
                "chunk_index": passage.chunk_index,
                "metadata": passage.metadata
            }
        
        logger.info(f"Embedding {len(passage_texts)} passages...")
        
        # Get embeddings for all passages
        text_embeddings = self.embedding_manager.embed_texts_batch(
            passage_texts, show_progress=True
        )
        
        # Map embeddings to passage IDs
        for passage_id, passage_text in zip(passage_ids, passage_texts):
            if passage_text in text_embeddings:
                self.passage_embeddings[passage_id] = text_embeddings[passage_text]
        
        logger.info(f"Loaded {len(self.passage_embeddings)} passage embeddings")
        return len(self.passage_embeddings)
    
    def retrieve_passages(
        self,
        processed_query: ProcessedQuery,
        top_k: int = 20,
        boost_factor: float = 1.2
    ) -> List[Tuple[str, float]]:
        """
        Retrieve passages using direct semantic similarity
        
        Args:
            processed_query: ProcessedQuery with query embedding
            top_k: Number of top passages to return
            boost_factor: Factor to boost scores for longer passages
            
        Returns:
            List of (passage_id, similarity_score) tuples
        """
        if not self.passage_embeddings:
            logger.warning("No passage embeddings loaded. Call load_passage_embeddings() first.")
            return []
        
        logger.info(f"Running semantic retrieval for query: '{processed_query.original_query}'")
        
        query_embedding = processed_query.query_embedding
        passage_scores = []
        
        # Calculate similarity with all passages
        for passage_id, passage_embedding in self.passage_embeddings.items():
            # Cosine similarity
            similarity = cosine_similarity(
                [query_embedding], [passage_embedding]
            )[0][0]
            
            # Apply length boost if enabled
            if self.passage_length_boost:
                passage_meta = self.passage_metadata.get(passage_id, {})
                word_count = passage_meta.get("word_count", 0)
                
                # Boost longer passages (up to a limit)
                if word_count > 50:  # Longer passages
                    length_boost = min(1.1, 1.0 + (word_count - 50) / 500)
                    similarity *= length_boost
            
            # Apply threshold filter
            if similarity >= self.min_similarity_threshold:
                passage_scores.append((passage_id, similarity))
        
        # Sort by similarity and take top-k
        passage_scores.sort(key=lambda x: x[1], reverse=True)
        top_passages = passage_scores[:top_k]
        
        # Normalize scores if requested
        if self.normalize_scores and top_passages:
            max_score = top_passages[0][1]
            if max_score > 0:
                top_passages = [
                    (passage_id, score / max_score) 
                    for passage_id, score in top_passages
                ]
        
        logger.info(f"Semantic retrieval found {len(top_passages)} passages above threshold")
        
        if top_passages:
            logger.debug(f"Top semantic scores: {[(pid[:8], f'{score:.3f}') for pid, score in top_passages[:5]]}")
        
        # Update stats
        self.retrieval_stats["queries_processed"] += 1
        self.retrieval_stats["total_passages_scored"] += len(passage_scores)
        if passage_scores:
            avg_score = sum(score for _, score in passage_scores) / len(passage_scores)
            self.retrieval_stats["avg_similarity_scores"].append(avg_score)
        
        return top_passages
    
    def get_passage_similarity(
        self,
        query_text: str,
        passage_id: str
    ) -> Optional[float]:
        """
        Get semantic similarity between query and specific passage
        
        Args:
            query_text: Query text
            passage_id: Target passage ID
            
        Returns:
            Cosine similarity score or None if passage not found
        """
        if passage_id not in self.passage_embeddings:
            return None
        
        # Embed query
        query_embedding = self.embedding_manager.embed_text(query_text)
        if query_embedding is None:
            return None
        
        # Calculate similarity
        passage_embedding = self.passage_embeddings[passage_id]
        similarity = cosine_similarity(
            [query_embedding], [passage_embedding]
        )[0][0]
        
        return similarity
    
    def explain_retrieval(
        self,
        processed_query: ProcessedQuery,
        passage_id: str,
        graph
    ) -> Dict[str, any]:
        """
        Explain why a passage was retrieved semantically
        
        Args:
            processed_query: Original processed query
            passage_id: Retrieved passage ID
            graph: TERAGGraph for passage content
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            "passage_id": passage_id,
            "query": processed_query.original_query,
            "similarity_score": 0.0,
            "explanation": "",
            "semantic_overlap": [],
            "passage_preview": ""
        }
        
        if passage_id not in self.passage_embeddings:
            explanation["explanation"] = "Passage not found in semantic index"
            return explanation
        
        # Calculate similarity
        similarity = cosine_similarity(
            [processed_query.query_embedding], 
            [self.passage_embeddings[passage_id]]
        )[0][0]
        
        explanation["similarity_score"] = similarity
        
        # Get passage content for preview
        if passage_id in graph.passages:
            content = graph.passages[passage_id].content
            explanation["passage_preview"] = content[:200] + "..." if len(content) > 200 else content
        
        # Generate explanation
        if similarity > 0.8:
            explanation["explanation"] = "Very high semantic similarity - passage closely matches query intent"
        elif similarity > 0.6:
            explanation["explanation"] = "High semantic similarity - passage discusses related topics"
        elif similarity > 0.4:
            explanation["explanation"] = "Moderate semantic similarity - passage has some topical overlap"
        else:
            explanation["explanation"] = "Low semantic similarity - weak topical connection"
        
        return explanation
    
    def batch_retrieve_passages(
        self,
        processed_queries: List[ProcessedQuery],
        top_k: int = 20
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch process multiple queries for efficiency
        
        Args:
            processed_queries: List of ProcessedQuery objects
            top_k: Number of results per query
            
        Returns:
            List of passage score lists for each query
        """
        logger.info(f"Batch semantic retrieval for {len(processed_queries)} queries")
        
        all_results = []
        
        for i, processed_query in enumerate(processed_queries):
            try:
                passages = self.retrieve_passages(processed_query, top_k)
                all_results.append(passages)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(processed_queries)} queries")
                    
            except Exception as e:
                logger.error(f"Failed to process query {i}: {e}")
                all_results.append([])
        
        logger.info(f"Batch semantic retrieval completed")
        return all_results
    
    def get_retrieval_statistics(self) -> Dict:
        """Get statistics about semantic retrieval performance"""
        stats = self.retrieval_stats.copy()
        
        if stats["avg_similarity_scores"]:
            stats["overall_avg_similarity"] = sum(stats["avg_similarity_scores"]) / len(stats["avg_similarity_scores"])
            stats["max_similarity_seen"] = max(stats["avg_similarity_scores"])
            stats["min_similarity_seen"] = min(stats["avg_similarity_scores"])
        else:
            stats["overall_avg_similarity"] = 0.0
            stats["max_similarity_seen"] = 0.0
            stats["min_similarity_seen"] = 0.0
        
        stats.update({
            "loaded_passages": len(self.passage_embeddings),
            "similarity_threshold": self.min_similarity_threshold,
            "length_boost_enabled": self.passage_length_boost,
            "embedding_model": self.embedding_manager.model
        })
        
        return stats
    
    def clear_cache(self):
        """Clear passage embeddings cache"""
        self.passage_embeddings.clear()
        self.passage_metadata.clear()
        logger.info("Semantic retrieval cache cleared")


# Convenience functions
def create_semantic_retriever(
    graph, 
    embedding_manager: Optional[EmbeddingManager] = None, 
    **kwargs
) -> SemanticRetriever:
    """
    Create and configure a semantic retriever for a graph
    
    Args:
        graph: TERAGGraph instance
        embedding_manager: Optional pre-configured embedding manager
        **kwargs: Additional arguments for SemanticRetriever
        
    Returns:
        Configured SemanticRetriever with loaded passage embeddings
    """
    if embedding_manager is None:
        embedding_manager = EmbeddingManager()
    
    retriever = SemanticRetriever(embedding_manager, **kwargs)
    retriever.load_passage_embeddings(graph)
    
    return retriever


if __name__ == "__main__":
    # Test semantic retriever
    print("🔍 SEMANTIC RETRIEVER TEST")
    print("=" * 60)
    
    print("💡 To test semantic retriever:")
    print("   1. Load a TERAG graph")
    print("   2. Create embedding manager")
    print("   3. Initialize semantic retriever with passage embeddings")
    print("   4. Process query and run semantic retrieval")
    
    print(f"\n📋 Example usage:")
    print("""
    from semantic_retriever import create_semantic_retriever
    from query_processor import create_query_processor
    from embedding_manager import EmbeddingManager
    from graph_builder import TERAGGraph
    
    # Load graph and create components
    graph = TERAGGraph.load_from_file("graph.json")
    embedding_manager = EmbeddingManager()
    semantic_retriever = create_semantic_retriever(graph, embedding_manager)
    query_processor = create_query_processor(graph, embedding_manager)
    
    # Process query and retrieve
    processed_query = query_processor.process_query("How is the music business?")
    passages = semantic_retriever.retrieve_passages(processed_query, top_k=10)
    
    print(f"Retrieved {len(passages)} semantically similar passages")
    """)