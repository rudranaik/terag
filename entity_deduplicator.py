"""
3-Phase Entity Deduplication for TERAG

Phase 1: String similarity (fuzzy matching)
Phase 2: Embedding-based semantic similarity  
Phase 3: Graph-based validation using co-occurrence

Preserves graph relationships while consolidating duplicate entities.
"""

import re
import math
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging

# String similarity
try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DuplicateCandidate:
    """Potential duplicate entity pair"""
    entity1: str
    entity2: str
    string_similarity: float
    embedding_similarity: float = 0.0
    graph_similarity: float = 0.0
    confidence_score: float = 0.0
    phase_detected: str = ""


@dataclass
class EntityCluster:
    """Cluster of duplicate entities"""
    canonical_entity: str  # The "main" entity name to use
    duplicate_entities: Set[str]
    confidence_score: float
    detection_method: str


class EntityDeduplicator:
    """
    3-phase entity deduplication for TERAG graphs
    """
    
    def __init__(
        self,
        string_similarity_threshold: float = 0.8,
        embedding_similarity_threshold: float = 0.85,
        graph_similarity_threshold: float = 0.6,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.string_threshold = string_similarity_threshold
        self.embedding_threshold = embedding_similarity_threshold
        self.graph_threshold = graph_similarity_threshold
        
        # Initialize string similarity
        if not FUZZYWUZZY_AVAILABLE:
            logger.warning("fuzzywuzzy not available. Install with: pip install fuzzywuzzy python-levenshtein")
        
        # Initialize embedding model
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        else:
            logger.warning("Sentence transformers not available. Install with: pip install sentence-transformers")
        
        # Cache embeddings to avoid recomputation
        self.embedding_cache = {}
        
    def deduplicate_entities_from_graph(self, graph) -> Dict[str, str]:
        """
        Main deduplication pipeline
        
        Args:
            graph: TERAGGraph instance
            
        Returns:
            Dict mapping old_entity -> canonical_entity
        """
        logger.info(f"Starting 3-phase deduplication on {len(graph.concepts)} entities")
        
        # Get all entities (filter by type if needed)
        entities = [
            concept.concept_text for concept in graph.concepts.values()
            if concept.concept_type in ["named_entity", "concept"]  # Include both types
        ]
        
        logger.info(f"Processing {len(entities)} entities for deduplication")
        
        # Phase 1: String similarity
        logger.info("Phase 1: String similarity matching...")
        string_candidates = self._phase1_string_similarity(entities)
        logger.info(f"Found {len(string_candidates)} string similarity candidates")
        
        # Phase 2: Embedding similarity
        logger.info("Phase 2: Embedding-based similarity...")
        embedding_candidates = self._phase2_embedding_similarity(entities, string_candidates)
        logger.info(f"Found {len(embedding_candidates)} embedding similarity candidates")
        
        # Phase 3: Graph-based validation
        logger.info("Phase 3: Graph-based validation...")
        validated_candidates = self._phase3_graph_validation(
            embedding_candidates, graph
        )
        logger.info(f"Validated {len(validated_candidates)} candidates using graph context")
        
        # Create entity clusters
        logger.info("Creating entity clusters...")
        clusters = self._create_entity_clusters(validated_candidates)
        logger.info(f"Created {len(clusters)} entity clusters")
        
        # Create mapping
        entity_mapping = self._create_entity_mapping(clusters)
        
        logger.info(f"Deduplication complete: {len(entity_mapping)} entities will be merged")
        return entity_mapping
    
    def _phase1_string_similarity(self, entities: List[str]) -> List[DuplicateCandidate]:
        """Phase 1: Fuzzy string matching"""
        candidates = []
        
        # Compare all pairs
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                
                # Skip if same entity
                if entity1 == entity2:
                    continue
                
                # Calculate similarity scores
                similarity_score = self._calculate_string_similarity(entity1, entity2)
                
                if similarity_score >= self.string_threshold:
                    candidate = DuplicateCandidate(
                        entity1=entity1,
                        entity2=entity2,
                        string_similarity=similarity_score,
                        phase_detected="string"
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _phase2_embedding_similarity(
        self, 
        entities: List[str], 
        string_candidates: List[DuplicateCandidate]
    ) -> List[DuplicateCandidate]:
        """Phase 2: Embedding-based semantic similarity"""
        
        if not self.embedding_model:
            logger.warning("Embedding model not available, skipping phase 2")
            return string_candidates
        
        # Get embeddings for all entities
        logger.info("Computing embeddings for all entities...")
        entity_embeddings = self._get_embeddings_batch(entities)
        
        # Add embedding scores to string candidates
        enhanced_candidates = []
        for candidate in string_candidates:
            emb1 = entity_embeddings.get(candidate.entity1)
            emb2 = entity_embeddings.get(candidate.entity2)
            
            if emb1 is not None and emb2 is not None:
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                candidate.embedding_similarity = similarity
                enhanced_candidates.append(candidate)
        
        # Find new candidates through embedding similarity only
        logger.info("Finding new embedding-based candidates...")
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity1 == entity2:
                    continue
                
                # Skip if already found in string phase
                if any(
                    (c.entity1 == entity1 and c.entity2 == entity2) or 
                    (c.entity1 == entity2 and c.entity2 == entity1) 
                    for c in string_candidates
                ):
                    continue
                
                emb1 = entity_embeddings.get(entity1)
                emb2 = entity_embeddings.get(entity2)
                
                if emb1 is not None and emb2 is not None:
                    similarity = cosine_similarity([emb1], [emb2])[0][0]
                    
                    if similarity >= self.embedding_threshold:
                        candidate = DuplicateCandidate(
                            entity1=entity1,
                            entity2=entity2,
                            string_similarity=self._calculate_string_similarity(entity1, entity2),
                            embedding_similarity=similarity,
                            phase_detected="embedding"
                        )
                        enhanced_candidates.append(candidate)
        
        return enhanced_candidates
    
    def _phase3_graph_validation(
        self, 
        candidates: List[DuplicateCandidate], 
        graph
    ) -> List[DuplicateCandidate]:
        """Phase 3: Validate candidates using graph co-occurrence"""
        
        validated_candidates = []
        
        for candidate in candidates:
            # Get passages connected to each entity
            passages1 = self._get_entity_passages(candidate.entity1, graph)
            passages2 = self._get_entity_passages(candidate.entity2, graph)
            
            # Calculate graph similarity (Jaccard similarity of connected passages)
            graph_sim = self._calculate_graph_similarity(passages1, passages2)
            candidate.graph_similarity = graph_sim
            
            # Calculate overall confidence score
            candidate.confidence_score = self._calculate_confidence_score(candidate)
            
            # Apply graph validation threshold
            if graph_sim >= self.graph_threshold:
                candidate.phase_detected = candidate.phase_detected + "_graph_validated"
                validated_candidates.append(candidate)
            elif candidate.string_similarity >= 0.95:  # Very high string similarity overrides graph
                candidate.phase_detected = candidate.phase_detected + "_high_string_override"
                validated_candidates.append(candidate)
        
        return validated_candidates
    
    def _calculate_string_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate string similarity between two entities"""
        
        if not FUZZYWUZZY_AVAILABLE:
            # Fallback to simple methods
            return self._simple_string_similarity(entity1, entity2)
        
        # Use multiple fuzzy matching methods
        ratio = fuzz.ratio(entity1.lower(), entity2.lower()) / 100.0
        partial_ratio = fuzz.partial_ratio(entity1.lower(), entity2.lower()) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(entity1.lower(), entity2.lower()) / 100.0
        
        # Take the maximum of different methods
        return max(ratio, partial_ratio, token_sort_ratio)
    
    def _simple_string_similarity(self, entity1: str, entity2: str) -> float:
        """Simple string similarity without fuzzywuzzy"""
        entity1_lower = entity1.lower().strip()
        entity2_lower = entity2.lower().strip()
        
        # Exact match
        if entity1_lower == entity2_lower:
            return 1.0
        
        # One is substring of other
        if entity1_lower in entity2_lower or entity2_lower in entity1_lower:
            return 0.9
        
        # Simple Levenshtein-like similarity
        max_len = max(len(entity1_lower), len(entity2_lower))
        if max_len == 0:
            return 0.0
        
        # Count common characters (very simple)
        common_chars = len(set(entity1_lower) & set(entity2_lower))
        return common_chars / max_len
    
    def _get_embeddings_batch(self, entities: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for all entities, with caching"""
        entity_embeddings = {}
        
        # Check cache first
        uncached_entities = []
        for entity in entities:
            if entity in self.embedding_cache:
                entity_embeddings[entity] = self.embedding_cache[entity]
            else:
                uncached_entities.append(entity)
        
        # Compute embeddings for uncached entities
        if uncached_entities:
            try:
                embeddings = self.embedding_model.encode(uncached_entities, convert_to_numpy=True)
                for entity, embedding in zip(uncached_entities, embeddings):
                    self.embedding_cache[entity] = embedding
                    entity_embeddings[entity] = embedding
            except Exception as e:
                logger.error(f"Failed to compute embeddings: {e}")
        
        return entity_embeddings
    
    def _get_entity_passages(self, entity: str, graph) -> Set[str]:
        """Get all passage IDs connected to an entity"""
        # Find the concept node for this entity
        for concept_id, concept in graph.concepts.items():
            if concept.concept_text == entity:
                return concept.passage_ids.copy()
        return set()
    
    def _calculate_graph_similarity(self, passages1: Set[str], passages2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets of passages"""
        if not passages1 and not passages2:
            return 1.0  # Both empty
        if not passages1 or not passages2:
            return 0.0  # One empty
        
        intersection = len(passages1 & passages2)
        union = len(passages1 | passages2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_confidence_score(self, candidate: DuplicateCandidate) -> float:
        """Calculate overall confidence score for a duplicate candidate"""
        # Weighted combination of similarity scores
        string_weight = 0.3
        embedding_weight = 0.4
        graph_weight = 0.3
        
        score = (
            candidate.string_similarity * string_weight +
            candidate.embedding_similarity * embedding_weight +
            candidate.graph_similarity * graph_weight
        )
        
        return min(1.0, score)
    
    def _create_entity_clusters(self, candidates: List[DuplicateCandidate]) -> List[EntityCluster]:
        """Create clusters of duplicate entities"""
        
        # Build graph of entity relationships
        entity_graph = defaultdict(set)
        for candidate in candidates:
            entity_graph[candidate.entity1].add(candidate.entity2)
            entity_graph[candidate.entity2].add(candidate.entity1)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for entity in entity_graph:
            if entity not in visited:
                cluster_entities = self._dfs_cluster(entity, entity_graph, visited)
                
                if len(cluster_entities) > 1:
                    # Choose canonical entity (longest or most common)
                    canonical = self._choose_canonical_entity(cluster_entities)
                    
                    # Calculate cluster confidence
                    cluster_confidence = self._calculate_cluster_confidence(
                        cluster_entities, candidates
                    )
                    
                    clusters.append(EntityCluster(
                        canonical_entity=canonical,
                        duplicate_entities=cluster_entities - {canonical},
                        confidence_score=cluster_confidence,
                        detection_method="multi_phase"
                    ))
        
        return clusters
    
    def _dfs_cluster(self, entity: str, entity_graph: dict, visited: set) -> Set[str]:
        """DFS to find connected component (cluster)"""
        cluster = set()
        stack = [entity]
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                cluster.add(current)
                stack.extend(entity_graph[current])
        
        return cluster
    
    def _choose_canonical_entity(self, entities: Set[str]) -> str:
        """Choose the canonical (representative) entity from a cluster"""
        # Strategy: Choose the longest, most complete entity name
        return max(entities, key=lambda x: (len(x), x.lower()))
    
    def _calculate_cluster_confidence(
        self, 
        cluster_entities: Set[str], 
        candidates: List[DuplicateCandidate]
    ) -> float:
        """Calculate confidence score for an entity cluster"""
        
        # Get all candidate pairs within this cluster
        cluster_candidates = [
            c for c in candidates 
            if c.entity1 in cluster_entities and c.entity2 in cluster_entities
        ]
        
        if not cluster_candidates:
            return 0.0
        
        # Average confidence of all pairs in cluster
        avg_confidence = sum(c.confidence_score for c in cluster_candidates) / len(cluster_candidates)
        return avg_confidence
    
    def _create_entity_mapping(self, clusters: List[EntityCluster]) -> Dict[str, str]:
        """Create mapping from duplicate entities to canonical entities"""
        entity_mapping = {}
        
        for cluster in clusters:
            for duplicate in cluster.duplicate_entities:
                entity_mapping[duplicate] = cluster.canonical_entity
        
        return entity_mapping


def deduplicate_graph_entities(graph, **deduplicator_kwargs) -> Tuple[Dict[str, str], List[EntityCluster]]:
    """
    Convenience function to deduplicate entities in a TERAG graph
    
    Args:
        graph: TERAGGraph instance
        **deduplicator_kwargs: Arguments for EntityDeduplicator
        
    Returns:
        (entity_mapping, clusters) - mapping dict and cluster info
    """
    deduplicator = EntityDeduplicator(**deduplicator_kwargs)
    entity_mapping = deduplicator.deduplicate_entities_from_graph(graph)
    
    # Get cluster info for reporting
    candidates = deduplicator._phase1_string_similarity(
        [concept.concept_text for concept in graph.concepts.values()]
    )
    clusters = deduplicator._create_entity_clusters(candidates)
    
    return entity_mapping, clusters


if __name__ == "__main__":
    # Test deduplication with sample entities
    test_entities = [
        "Apple Inc",
        "Apple Corporation", 
        "Apple",
        "Tim Cook",
        "Timothy Cook",
        "CEO Tim Cook",
        "Microsoft Corporation",
        "Microsoft Corp",
        "Q4 2024",
        "fourth quarter 2024",
        "revenue growth",
        "revenue increase"
    ]
    
    print("🔍 ENTITY DEDUPLICATION TEST")
    print("=" * 60)
    
    deduplicator = EntityDeduplicator()
    
    # Test string similarity
    print("\n📝 String Similarity Tests:")
    test_pairs = [
        ("Apple Inc", "Apple Corporation"),
        ("Tim Cook", "Timothy Cook"), 
        ("Q4 2024", "fourth quarter 2024"),
        ("Microsoft Corporation", "Microsoft Corp")
    ]
    
    for entity1, entity2 in test_pairs:
        similarity = deduplicator._calculate_string_similarity(entity1, entity2)
        print(f"   {entity1:<25} vs {entity2:<25}: {similarity:.3f}")
    
    # Test embedding similarity if available
    if deduplicator.embedding_model:
        print("\n🧠 Embedding Similarity Tests:")
        embeddings = deduplicator._get_embeddings_batch([pair[0] for pair in test_pairs] + 
                                                       [pair[1] for pair in test_pairs])
        
        for entity1, entity2 in test_pairs:
            if entity1 in embeddings and entity2 in embeddings:
                similarity = cosine_similarity([embeddings[entity1]], [embeddings[entity2]])[0][0]
                print(f"   {entity1:<25} vs {entity2:<25}: {similarity:.3f}")
    else:
        print("\n⚠️  Embedding model not available for testing")
    
    print(f"\n✅ Deduplication test completed!")
    print(f"💡 To test with real graph data, run:")
    print(f"   python entity_deduplicator.py with your TERAGGraph instance")