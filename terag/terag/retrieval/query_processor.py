"""
TERAG Query Processor

Processes user queries for dual-layer retrieval:
1. Extract entities from query using embedding similarity to graph entities
2. Embed the full query for direct semantic passage matching

Handles query preprocessing, entity extraction, and query embedding.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from terag.embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class QueryEntity:
    """Extracted entity from query"""
    entity_text: str
    graph_concept_id: str
    graph_concept_text: str
    similarity_score: float
    entity_type: str  # "named_entity" or "document_concept"


@dataclass
class ProcessedQuery:
    """Processed query with extracted entities and embeddings"""
    original_query: str
    cleaned_query: str
    query_embedding: np.ndarray
    extracted_entities: List[QueryEntity]
    confidence_score: float
    
    def get_matched_concept_ids(self) -> List[str]:
        """Get list of matched graph concept IDs"""
        return [entity.graph_concept_id for entity in self.extracted_entities]


class QueryProcessor:
    """
    Processes queries for TERAG dual-layer retrieval
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        entity_similarity_threshold: float = 0.7,
        max_entities_per_query: int = 10,
        min_query_length: int = 3,
        enable_query_expansion: bool = True
    ):
        """
        Initialize query processor
        
        Args:
            embedding_manager: Pre-configured embedding manager
            entity_similarity_threshold: Minimum similarity for entity matching
            max_entities_per_query: Maximum entities to extract per query
            min_query_length: Minimum query length in characters
            enable_query_expansion: Whether to expand queries with synonyms
        """
        self.embedding_manager = embedding_manager
        self.entity_threshold = entity_similarity_threshold
        self.max_entities = max_entities_per_query
        self.min_query_length = min_query_length
        self.enable_query_expansion = enable_query_expansion
        
        # Cache for graph entities and their embeddings
        self.graph_entity_embeddings = {}
        self.graph_entities_metadata = {}  # concept_id -> concept info
        
        # Query preprocessing patterns
        self.stop_words = {
            "what", "how", "when", "where", "why", "who", "which", "is", "are", "was", "were",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "be", "been", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can", "about", "tell", "me"
        }
    
    def load_graph_entities(self, graph) -> int:
        """
        Load and embed all entities from the graph for matching
        
        Args:
            graph: TERAGGraph instance
            
        Returns:
            Number of entities loaded
        """
        logger.info("Loading graph entities for query processing...")
        
        # Extract entity texts and metadata
        entity_texts = []
        for concept_id, concept in graph.concepts.items():
            entity_texts.append(concept.concept_text)
            self.graph_entities_metadata[concept_id] = {
                "concept_text": concept.concept_text,
                "concept_type": concept.concept_type,
                "frequency": concept.frequency,
                "passage_count": len(concept.passage_ids)
            }
        
        # Get embeddings for all entities
        logger.info(f"Embedding {len(entity_texts)} graph entities...")
        text_embeddings = self.embedding_manager.embed_texts_batch(entity_texts, show_progress=True)
        
        # Map embeddings to concept IDs
        for concept_id, concept in graph.concepts.items():
            concept_text = concept.concept_text
            if concept_text in text_embeddings:
                self.graph_entity_embeddings[concept_id] = text_embeddings[concept_text]
        
        logger.info(f"Loaded {len(self.graph_entity_embeddings)} entity embeddings for query matching")
        return len(self.graph_entity_embeddings)
    
    def process_query(self, query: str) -> ProcessedQuery:
        """
        Process a user query for retrieval
        
        Args:
            query: Raw user query string
            
        Returns:
            ProcessedQuery object with extracted entities and embeddings
        """
        logger.info(f"Processing query: '{query}'")
        
        # Validate query
        if len(query.strip()) < self.min_query_length:
            raise ValueError(f"Query too short (minimum {self.min_query_length} characters)")
        
        # Clean and preprocess query
        cleaned_query = self._clean_query(query)
        logger.debug(f"Cleaned query: '{cleaned_query}'")
        
        # Embed the full query
        query_embedding = self.embedding_manager.embed_text(cleaned_query)
        if query_embedding is None:
            raise RuntimeError("Failed to embed query")
        
        # Extract entities from query
        extracted_entities = self._extract_query_entities(query, query_embedding)
        logger.info(f"Extracted {len(extracted_entities)} entities from query")
        
        # Calculate overall confidence score
        confidence_score = self._calculate_query_confidence(extracted_entities, query_embedding)
        
        # Create processed query
        processed_query = ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            query_embedding=query_embedding,
            extracted_entities=extracted_entities,
            confidence_score=confidence_score
        )
        
        logger.info(f"Query processing complete: {len(extracted_entities)} entities, confidence: {confidence_score:.3f}")
        
        # Store for demo access
        self._last_processed_query = processed_query
        
        return processed_query
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text (minimal cleaning for entity extraction)"""
        # Convert to lowercase
        cleaned = query.lower().strip()
        
        # Remove special characters but keep spaces and basic punctuation
        cleaned = re.sub(r'[^\w\s\-\'\.]', ' ', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Don't remove stop words for entity extraction - they provide context
        # The original query is better for entity matching
        return cleaned
    
    def _extract_query_entities(self, original_query: str, query_embedding: np.ndarray) -> List[QueryEntity]:
        """
        Extract entities from query using multiple strategies:
        1. Text-based extraction (n-grams matching)
        2. Embedding similarity with individual terms
        3. Embedding similarity with full query (fallback)
        
        Args:
            original_query: Original query text
            query_embedding: Embedding of the query
            
        Returns:
            List of extracted QueryEntity objects
        """
        if not self.graph_entity_embeddings:
            logger.warning("No graph entity embeddings loaded. Call load_graph_entities() first.")
            return []
        
        logger.info(f"Extracting entities from query: '{original_query}'")
        
        # Strategy 1: Direct text matching (most reliable)
        text_matched_entities = self._extract_entities_by_text_matching(original_query)
        logger.info(f"Text matching found {len(text_matched_entities)} entities")
        
        # Strategy 2: Embed individual query terms and match
        term_matched_entities = self._extract_entities_by_term_embedding(original_query)
        logger.info(f"Term embedding found {len(term_matched_entities)} entities")
        
        # Strategy 3: Full query embedding similarity (fallback)
        query_matched_entities = self._extract_entities_by_query_embedding(original_query, query_embedding)
        logger.info(f"Query embedding found {len(query_matched_entities)} entities")
        
        # Combine and deduplicate all strategies
        all_entities = {}  # concept_id -> best_score_entity
        
        # Add text matches (highest priority)
        for entity in text_matched_entities:
            all_entities[entity.graph_concept_id] = entity
        
        # Add term matches (medium priority, but don't override text matches)
        for entity in term_matched_entities:
            if entity.graph_concept_id not in all_entities:
                all_entities[entity.graph_concept_id] = entity
            else:
                # Keep the one with higher score
                existing = all_entities[entity.graph_concept_id]
                if entity.similarity_score > existing.similarity_score:
                    all_entities[entity.graph_concept_id] = entity
        
        # Add query matches (lowest priority)
        for entity in query_matched_entities:
            if entity.graph_concept_id not in all_entities:
                all_entities[entity.graph_concept_id] = entity
        
        # Convert to list and sort by score
        final_entities = list(all_entities.values())
        final_entities.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Take top entities
        final_entities = final_entities[:self.max_entities]
        
        logger.info(f"Final entity extraction: {len(final_entities)} entities")
        for entity in final_entities[:5]:  # Log top 5
            logger.info(f"  - '{entity.entity_text}' (score: {entity.similarity_score:.3f})")
        
        return final_entities
    
    def _extract_entities_by_text_matching(self, query: str) -> List[QueryEntity]:
        """Extract entities using direct text matching (n-grams)"""
        query_lower = query.lower()
        matched_entities = []
        
        # Check all graph entities for text overlap
        for concept_id, metadata in self.graph_entities_metadata.items():
            concept_text = metadata["concept_text"].lower()
            
            # Check if entity text appears in query
            overlap_score = self._calculate_text_overlap_score(query_lower, concept_text)
            
            if overlap_score > 0:
                # High score for text matches
                final_score = 0.8 + overlap_score * 0.2  # 0.8 to 1.0 range
                
                entity = QueryEntity(
                    entity_text=metadata["concept_text"],
                    graph_concept_id=concept_id,
                    graph_concept_text=metadata["concept_text"],
                    similarity_score=final_score,
                    entity_type=metadata["concept_type"]
                )
                matched_entities.append(entity)
        
        return matched_entities
    
    def _extract_entities_by_term_embedding(self, query: str) -> List[QueryEntity]:
        """Extract entities by embedding individual query terms"""
        matched_entities = []
        
        # Extract meaningful terms from query (skip stop words)
        query_terms = []
        words = query.lower().split()
        
        # Add individual words
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                query_terms.append(word)
        
        # Add bigrams and trigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if not any(w in self.stop_words for w in [words[i], words[i+1]]):
                query_terms.append(bigram)
        
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if not any(w in self.stop_words for w in [words[i], words[i+1], words[i+2]]):
                query_terms.append(trigram)
        
        logger.debug(f"Query terms to match: {query_terms}")
        
        # Embed each term and find matches
        for term in query_terms:
            term_embedding = self.embedding_manager.embed_text(term)
            if term_embedding is None:
                continue
            
            # Find best matching graph entities
            for concept_id, entity_embedding in self.graph_entity_embeddings.items():
                similarity = cosine_similarity([term_embedding], [entity_embedding])[0][0]
                
                # Lower threshold for individual terms
                if similarity >= 0.6:  # Lower threshold than full query
                    metadata = self.graph_entities_metadata[concept_id]
                    
                    # Boost score if there's also text overlap
                    text_overlap = self._calculate_text_overlap_score(term.lower(), metadata["concept_text"].lower())
                    final_score = similarity + text_overlap * 0.2
                    
                    entity = QueryEntity(
                        entity_text=metadata["concept_text"],
                        graph_concept_id=concept_id,
                        graph_concept_text=metadata["concept_text"],
                        similarity_score=final_score,
                        entity_type=metadata["concept_type"]
                    )
                    matched_entities.append(entity)
        
        # Deduplicate by concept_id, keeping highest score
        best_entities = {}
        for entity in matched_entities:
            if entity.graph_concept_id not in best_entities:
                best_entities[entity.graph_concept_id] = entity
            elif entity.similarity_score > best_entities[entity.graph_concept_id].similarity_score:
                best_entities[entity.graph_concept_id] = entity
        
        return list(best_entities.values())
    
    def _extract_entities_by_query_embedding(self, query: str, query_embedding: np.ndarray) -> List[QueryEntity]:
        """Extract entities using full query embedding (fallback method)"""
        matched_entities = []
        
        # This is the original method - keep as fallback
        for concept_id, entity_embedding in self.graph_entity_embeddings.items():
            similarity = cosine_similarity([query_embedding], [entity_embedding])[0][0]
            
            # Use lower threshold since this is less reliable
            if similarity >= 0.5:  # Lower threshold for full query matching
                metadata = self.graph_entities_metadata[concept_id]
                
                entity = QueryEntity(
                    entity_text=metadata["concept_text"],
                    graph_concept_id=concept_id,
                    graph_concept_text=metadata["concept_text"],
                    similarity_score=similarity,
                    entity_type=metadata["concept_type"]
                )
                matched_entities.append(entity)
        
        return matched_entities
    
    def _calculate_text_overlap_score(self, query: str, entity: str) -> float:
        """Calculate text overlap score between query and entity (0.0 to 1.0)"""
        query_words = set(query.split())
        entity_words = set(entity.split())
        
        # Exact substring match gets highest score
        if entity in query or query in entity:
            return 1.0
        
        # Calculate word overlap
        if not entity_words:
            return 0.0
            
        overlap = len(query_words & entity_words)
        entity_word_count = len(entity_words)
        
        # Different scoring for single vs multi-word entities
        if entity_word_count == 1:
            # Single word: needs exact match
            return 1.0 if overlap > 0 else 0.0
        else:
            # Multi-word: proportional overlap
            overlap_ratio = overlap / entity_word_count
            
            # Require at least 50% overlap for multi-word entities
            if overlap_ratio >= 0.5:
                return overlap_ratio
            else:
                return 0.0
    
    def _check_text_overlap(self, query: str, entity: str) -> bool:
        """Check if entity text has significant overlap with query (legacy method)"""
        return self._calculate_text_overlap_score(query, entity) > 0.0
    
    def _calculate_query_confidence(self, extracted_entities: List[QueryEntity], query_embedding: np.ndarray) -> float:
        """
        Calculate overall confidence score for query processing
        
        Args:
            extracted_entities: List of extracted entities
            query_embedding: Query embedding vector
            
        Returns:
            Confidence score between 0 and 1
        """
        if not extracted_entities:
            return 0.1  # Low confidence with no entities
        
        # Base confidence from entity similarities
        avg_similarity = sum(entity.similarity_score for entity in extracted_entities) / len(extracted_entities)
        
        # Boost confidence if we found multiple entities
        entity_count_bonus = min(0.2, len(extracted_entities) * 0.05)
        
        # Boost confidence if we found high-frequency entities (important concepts)
        importance_bonus = 0.0
        for entity in extracted_entities:
            concept_id = entity.graph_concept_id
            if concept_id in self.graph_entities_metadata:
                frequency = self.graph_entities_metadata[concept_id]["frequency"]
                if frequency > 5:  # High-frequency entity
                    importance_bonus += 0.05
        
        importance_bonus = min(0.2, importance_bonus)
        
        # Calculate final confidence
        confidence = avg_similarity + entity_count_bonus + importance_bonus
        return min(1.0, confidence)
    
    def batch_process_queries(self, queries: List[str]) -> List[ProcessedQuery]:
        """
        Process multiple queries efficiently
        
        Args:
            queries: List of query strings
            
        Returns:
            List of ProcessedQuery objects
        """
        logger.info(f"Batch processing {len(queries)} queries...")
        
        processed_queries = []
        for i, query in enumerate(queries):
            try:
                processed_query = self.process_query(query)
                processed_queries.append(processed_query)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(queries)} queries")
                    
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                # Create empty processed query for failed cases
                processed_queries.append(ProcessedQuery(
                    original_query=query,
                    cleaned_query=query,
                    query_embedding=np.zeros(1536),  # text-embedding-3-small dimension
                    extracted_entities=[],
                    confidence_score=0.0
                ))
        
        logger.info(f"Batch processing complete: {len(processed_queries)} queries processed")
        return processed_queries
    
    def get_entity_matching_statistics(self) -> Dict:
        """Get statistics about entity matching performance"""
        return {
            "total_graph_entities": len(self.graph_entity_embeddings),
            "entity_similarity_threshold": self.entity_threshold,
            "max_entities_per_query": self.max_entities,
            "embedding_model": self.embedding_manager.model,
            "loaded_entity_types": {
                entity_type: sum(1 for metadata in self.graph_entities_metadata.values() 
                              if metadata["concept_type"] == entity_type)
                for entity_type in set(metadata["concept_type"] for metadata in self.graph_entities_metadata.values())
            }
        }


# Convenience functions
def create_query_processor(graph, embedding_manager: Optional[EmbeddingManager] = None, **kwargs) -> QueryProcessor:
    """
    Create and configure a query processor for a graph
    
    Args:
        graph: TERAGGraph instance
        embedding_manager: Optional pre-configured embedding manager
        **kwargs: Additional arguments for QueryProcessor
        
    Returns:
        Configured QueryProcessor
    """
    if embedding_manager is None:
        embedding_manager = EmbeddingManager()
    
    processor = QueryProcessor(embedding_manager, **kwargs)
    processor.load_graph_entities(graph)
    
    return processor


if __name__ == "__main__":
    # Test query processor
    print("🔍 QUERY PROCESSOR TEST")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "What did Tim Cook say about revenue growth?",
        "How is the music business performing?",
        "Saregama financial results Q2",
        "Live events and concerts revenue",
        "Apple announcement earnings"
    ]
    
    print("Sample test queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query}")
    
    print(f"\n💡 To test query processor:")
    print(f"   1. Load a TERAG graph")
    print(f"   2. Create embedding manager")
    print(f"   3. Initialize query processor with graph entities")
    print(f"   4. Process queries to extract matching entities")
    
    print(f"\n📋 Example usage:")
    print("""
    from query_processor import create_query_processor
    from embedding_manager import EmbeddingManager
    from graph_builder import TERAGGraph
    
    # Load graph and create processor
    graph = TERAGGraph.load_from_file("graph.json")
    embedding_manager = EmbeddingManager()
    processor = create_query_processor(graph, embedding_manager)
    
    # Process a query
    processed_query = processor.process_query("What did Tim Cook say about revenue?")
    print(f"Extracted {len(processed_query.extracted_entities)} entities")
    """)