#!/usr/bin/env python3
"""
Smart deduplication using OpenAI embeddings
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from .builder import TERAGGraph
from terag.embeddings.manager import EmbeddingManager
from .merger import apply_deduplication_to_graph
from .types import DuplicateCandidate, EntityCluster
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dotenv
dotenv.load_dotenv()

# String similarity
try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenAIEntityDeduplicator:
    """Entity deduplicator using OpenAI embeddings"""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        string_similarity_threshold: float = 0.8,
        embedding_similarity_threshold: float = 0.85,
        graph_similarity_threshold: float = 0.6
    ):
        self.embedding_manager = embedding_manager
        self.string_threshold = string_similarity_threshold
        self.embedding_threshold = embedding_similarity_threshold
        self.graph_threshold = graph_similarity_threshold
    
    def deduplicate_entities_from_graph(self, graph: TERAGGraph) -> Dict[str, str]:
        """Main deduplication pipeline using OpenAI embeddings"""
        
        print(f"🔍 Starting OpenAI-based deduplication on {len(graph.concepts)} entities")
        
        # Get all entities
        entities = [
            concept.concept_text for concept in graph.concepts.values()
            if concept.concept_type in ["named_entity", "concept"]
        ]
        
        print(f"📊 Processing {len(entities)} entities for deduplication")
        
        # Phase 1: String similarity
        print("🔤 Phase 1: String similarity matching...")
        string_candidates = self._phase1_string_similarity(entities)
        print(f"   Found {len(string_candidates)} string similarity candidates")
        
        # Phase 2: OpenAI Embedding similarity
        if self.embedding_threshold < 0.99:  # Only if embeddings enabled
            print("🧠 Phase 2: OpenAI embedding similarity...")
            print(f"   Using OpenAI {self.embedding_manager.model} model")
            print(f"   Batch size: {self.embedding_manager.batch_size} entities per API call")
            
            embedding_candidates = self._phase2_openai_embedding_similarity(entities, string_candidates)
            print(f"   Found {len(embedding_candidates)} embedding similarity candidates")
        else:
            print("⏭️  Phase 2: Skipping embeddings (threshold = 0.99)")
            embedding_candidates = string_candidates
        
        # Phase 3: Graph validation
        print("🕸️  Phase 3: Graph-based validation...")
        validated_candidates = self._phase3_graph_validation(embedding_candidates, graph)
        print(f"   Validated {len(validated_candidates)} candidates using graph context")
        
        # Create clusters
        clusters = self._create_entity_clusters(validated_candidates)
        entity_mapping = self._create_entity_mapping(clusters)
        
        print(f"✅ Deduplication complete: {len(entity_mapping)} entities will be merged")
        return entity_mapping, clusters
    
    def _phase1_string_similarity(self, entities: List[str]) -> List[DuplicateCandidate]:
        """Phase 1: Fast string similarity matching"""
        candidates = []
        
        print(f"   🔄 Comparing {len(entities)} entities...")
        
        for i, entity1 in enumerate(entities):
            if (i + 1) % 500 == 0:
                print(f"      Processed {i+1}/{len(entities)} entities...")
                
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity1 == entity2:
                    continue
                
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
    
    def _phase2_openai_embedding_similarity(
        self,
        entities: List[str],
        string_candidates: List[DuplicateCandidate]
    ) -> List[DuplicateCandidate]:
        """Phase 2: OpenAI embedding similarity"""
        
        # Get OpenAI embeddings for all entities
        print(f"   📡 Getting OpenAI embeddings for {len(entities)} entities...")
        
        start_time = time.time()
        entity_embeddings = self.embedding_manager.embed_texts_batch(entities, show_progress=True)
        embedding_time = time.time() - start_time
        
        print(f"   ✅ OpenAI embeddings completed in {embedding_time:.2f}s")
        print(f"   💰 Estimated cost: ${len(entities) * 0.00002:.4f} (approx)")
        
        # Add embedding scores to string candidates
        enhanced_candidates = []
        for candidate in string_candidates:
            emb1 = entity_embeddings.get(candidate.entity1)
            emb2 = entity_embeddings.get(candidate.entity2)
            
            if emb1 is not None and emb2 is not None:
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                candidate.embedding_similarity = similarity
                enhanced_candidates.append(candidate)
        
        # Find new embedding-based candidates (not found by string matching)
        print(f"   🔄 Finding additional semantic similarities...")
        
        # For efficiency, sample entities rather than all pairs
        import random
        sample_size = min(1000, len(entities))  # Limit to 1000 entities for embedding comparison
        sampled_entities = random.sample(entities, sample_size) if len(entities) > sample_size else entities
        
        new_candidates = []
        for i, entity1 in enumerate(sampled_entities):
            for j, entity2 in enumerate(sampled_entities[i+1:], i+1):
                
                # Skip if already found by string matching
                if any((c.entity1 == entity1 and c.entity2 == entity2) or 
                      (c.entity1 == entity2 and c.entity2 == entity1) 
                      for c in string_candidates):
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
                        new_candidates.append(candidate)
        
        print(f"   Found {len(new_candidates)} additional semantic candidates")
        return enhanced_candidates + new_candidates
    
    def _phase3_graph_validation(self, candidates: List[DuplicateCandidate], graph: TERAGGraph) -> List[DuplicateCandidate]:
        """Phase 3: Graph co-occurrence validation"""
        
        validated_candidates = []
        
        for candidate in candidates:
            # Get passages connected to each entity
            passages1 = self._get_entity_passages(candidate.entity1, graph)
            passages2 = self._get_entity_passages(candidate.entity2, graph)
            
            # Calculate graph similarity
            graph_sim = self._calculate_graph_similarity(passages1, passages2)
            candidate.graph_similarity = graph_sim
            
            # Calculate confidence
            candidate.confidence_score = self._calculate_confidence_score(candidate)
            
            # Apply validation
            if graph_sim >= self.graph_threshold:
                candidate.phase_detected = candidate.phase_detected + "_graph_validated"
                validated_candidates.append(candidate)
            elif candidate.string_similarity >= 0.95:  # Very high string similarity overrides
                candidate.phase_detected = candidate.phase_detected + "_high_string_override"
                validated_candidates.append(candidate)
        
        return validated_candidates
    
    def _calculate_string_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate string similarity"""
        if not FUZZYWUZZY_AVAILABLE:
            return self._simple_string_similarity(entity1, entity2)
        
        ratio = fuzz.ratio(entity1.lower(), entity2.lower()) / 100.0
        partial_ratio = fuzz.partial_ratio(entity1.lower(), entity2.lower()) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(entity1.lower(), entity2.lower()) / 100.0
        
        return max(ratio, partial_ratio, token_sort_ratio)
    
    def _simple_string_similarity(self, entity1: str, entity2: str) -> float:
        """Simple string similarity fallback"""
        entity1_lower = entity1.lower().strip()
        entity2_lower = entity2.lower().strip()
        
        if entity1_lower == entity2_lower:
            return 1.0
        if entity1_lower in entity2_lower or entity2_lower in entity1_lower:
            return 0.9
        
        max_len = max(len(entity1_lower), len(entity2_lower))
        if max_len == 0:
            return 0.0
        
        common_chars = len(set(entity1_lower) & set(entity2_lower))
        return common_chars / max_len
    
    def _get_entity_passages(self, entity: str, graph: TERAGGraph) -> Set[str]:
        """Get passages connected to entity"""
        for concept_id, concept in graph.concepts.items():
            if concept.concept_text == entity:
                return concept.passage_ids.copy()
        return set()
    
    def _calculate_graph_similarity(self, passages1: Set[str], passages2: Set[str]) -> float:
        """Calculate Jaccard similarity between passage sets"""
        if not passages1 and not passages2:
            return 1.0
        if not passages1 or not passages2:
            return 0.0
        
        intersection = len(passages1 & passages2)
        union = len(passages1 | passages2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_confidence_score(self, candidate: DuplicateCandidate) -> float:
        """Calculate confidence score"""
        string_weight = 0.3
        embedding_weight = 0.4
        graph_weight = 0.3
        
        score = (
            string_weight * candidate.string_similarity +
            embedding_weight * candidate.embedding_similarity +
            graph_weight * candidate.graph_similarity
        )
        
        return min(1.0, score)
    
    def _create_entity_clusters(self, candidates: List[DuplicateCandidate]) -> List[EntityCluster]:
        """Create entity clusters from candidates"""
        # Build adjacency graph
        entity_graph = defaultdict(set)
        for candidate in candidates:
            entity_graph[candidate.entity1].add(candidate.entity2)
            entity_graph[candidate.entity2].add(candidate.entity1)
        
        # Find connected components
        visited = set()
        clusters = []
        
        for entity in entity_graph:
            if entity in visited:
                continue
            
            # DFS to find connected component
            cluster_entities = set()
            stack = [entity]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                
                visited.add(current)
                cluster_entities.add(current)
                
                for neighbor in entity_graph[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            if len(cluster_entities) > 1:
                # Choose canonical entity (longest text)
                canonical = max(cluster_entities, key=len)
                duplicates = cluster_entities - {canonical}
                
                cluster = EntityCluster(
                    canonical_entity=canonical,
                    duplicate_entities=duplicates,
                    confidence_score=0.8,  # Default confidence
                    detection_method="string_and_embedding"
                )
                clusters.append(cluster)
        
        return clusters
    
    def _create_entity_mapping(self, clusters: List[EntityCluster]) -> Dict[str, str]:
        """Create mapping from duplicates to canonical entities"""
        mapping = {}
        
        for cluster in clusters:
            canonical = cluster.canonical_entity
            for duplicate in cluster.duplicate_entities:
                mapping[duplicate] = canonical
        
        return mapping


def main():
    # Your current graph file
    graph_file = "terag_data/graph_temp/terag_graph.json"
    output_dir = "terag_data"
    
    print("🚀 OPENAI SMART DEDUPLICATION")
    print("=" * 60)
    
    try:
        print("📖 Loading graph...")
        graph = TERAGGraph.load_from_file(graph_file)
        original_stats = graph.get_statistics()
        
        print(f"📊 Original: {original_stats['num_concepts']} concepts, {original_stats['num_passages']} passages")
        
        # Initialize OpenAI embedding manager with larger batch size
        print(f"🔧 Initializing OpenAI embedding manager...")
        embedding_manager = EmbeddingManager(batch_size=500)
        print(f"   Model: {embedding_manager.model}")
        print(f"   Batch size: {embedding_manager.batch_size} entities per API call")
        
        # Phase 1: String + embedding deduplication
        print(f"\n🔤 PHASE 1: String + embedding deduplication")
        
        phase1_deduplicator = OpenAIEntityDeduplicator(
            embedding_manager=embedding_manager,
            string_similarity_threshold=0.85,
            embedding_similarity_threshold=0.82,  # Enabled for embeddings
            graph_similarity_threshold=0.6
        )
        
        start_time = time.time()
        entity_mapping_1, clusters_1 = phase1_deduplicator.deduplicate_entities_from_graph(graph)
        phase1_time = time.time() - start_time
        
        print(f"⏱️  Phase 1 completed in {phase1_time:.2f} seconds")
        print(f"🎯 Found {len(entity_mapping_1)} string-based duplicates")
        
        if entity_mapping_1:
            graph = apply_deduplication_to_graph(graph, entity_mapping_1)
            phase1_stats = graph.get_statistics()
            concepts_reduced_1 = original_stats['num_concepts'] - phase1_stats['num_concepts']
            print(f"✅ Reduced concepts by {concepts_reduced_1} in phase 1")
            print(f"📊 Remaining: {phase1_stats['num_concepts']} concepts")
        
        # Phase 2: OpenAI embedding-based deduplication
        print(f"\n🧠 PHASE 2: OpenAI embedding-based deduplication")
        
        phase2_deduplicator = OpenAIEntityDeduplicator(
            embedding_manager=embedding_manager,
            string_similarity_threshold=0.7,
            embedding_similarity_threshold=0.82,  # Enabled
            graph_similarity_threshold=0.6
        )
        
        start_time = time.time()
        entity_mapping_2, clusters_2 = phase2_deduplicator.deduplicate_entities_from_graph(graph)
        phase2_time = time.time() - start_time
        
        print(f"⏱️  Phase 2 completed in {phase2_time:.2f} seconds")
        print(f"🎯 Found {len(entity_mapping_2)} semantic duplicates")
        
        if entity_mapping_2:
            final_graph = apply_deduplication_to_graph(graph, entity_mapping_2)
            
            final_stats = final_graph.get_statistics()
            total_concepts_reduced = original_stats['num_concepts'] - final_stats['num_concepts']
            total_reduction_pct = (total_concepts_reduced / original_stats['num_concepts']) * 100
            
            print(f"✅ Total concepts reduced: {total_concepts_reduced} ({total_reduction_pct:.1f}%)")
            print(f"📊 Final: {final_stats['num_concepts']} concepts, {final_stats['num_edges']} edges")
            
            # Show examples
            if clusters_2:
                print(f"\n📋 Semantic duplicate examples:")
                sorted_clusters = sorted(clusters_2, key=lambda c: len(c.duplicate_entities), reverse=True)
                for i, cluster in enumerate(sorted_clusters[:3]):
                    print(f"   {i+1}. '{cluster.canonical_entity}' ← {list(cluster.duplicate_entities)}")
        else:
            final_graph = graph
        
        # Save final graph
        final_file = f"{output_dir}/terag_graph.json"
        final_graph.save_to_file(final_file)
        
        total_time = phase1_time + phase2_time
        print(f"\n🎉 OPENAI DEDUPLICATION COMPLETE!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"💾 Final graph saved: {final_file}")
        
        # Estimate cost
        total_entities = original_stats['num_concepts']
        estimated_cost = total_entities * 0.00002  # Rough estimate for text-embedding-3-small
        print(f"💰 Estimated OpenAI cost: ${estimated_cost:.4f}")
        
        print(f"\n🔍 Ready for retrieval:")
        print(f"python retrieval_demo.py --graph-file {final_file} --interactive")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()