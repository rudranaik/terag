"""
Entity Merger for TERAG Graphs

Applies entity deduplication results to graphs while preserving all relationships.
Merges nodes and redirects edges safely.
"""

import logging
from typing import Dict, List, Set
from collections import defaultdict
from .builder import TERAGGraph, ConceptNode
from .types import EntityCluster

logger = logging.getLogger(__name__)


class EntityMerger:
    """
    Merges duplicate entities in TERAG graphs while preserving relationships
    """
    
    def __init__(self):
        self.merge_stats = {
            "entities_merged": 0,
            "edges_redirected": 0,
            "nodes_removed": 0,
            "clusters_processed": 0
        }
    
    def apply_entity_mapping(self, graph: TERAGGraph, entity_mapping: Dict[str, str]) -> TERAGGraph:
        """
        Apply entity deduplication mapping to graph
        
        Args:
            graph: Original TERAGGraph
            entity_mapping: Dict mapping duplicate_entity -> canonical_entity
            
        Returns:
            New TERAGGraph with merged entities
        """
        logger.info(f"Applying entity mapping: {len(entity_mapping)} entities to merge")
        
        # Create new graph
        merged_graph = TERAGGraph()
        
        # Copy all passages (unchanged)
        for passage_id, passage in graph.passages.items():
            merged_graph.add_passage(passage)
        
        # Process concepts with merging
        concept_mapping = self._create_concept_mapping(graph, entity_mapping)
        
        # Add merged concepts
        merged_concepts = self._merge_concepts(graph, concept_mapping)
        for concept in merged_concepts:
            merged_graph.add_concept(concept)
        
        # Rebuild edges with merged entities
        self._rebuild_edges(graph, merged_graph, concept_mapping)
        
        logger.info(f"Merge completed: {self.merge_stats}")
        return merged_graph
    
    def _create_concept_mapping(self, graph: TERAGGraph, entity_mapping: Dict[str, str]) -> Dict[str, str]:
        """
        Create mapping from concept_id to canonical concept_id
        
        Args:
            graph: Original graph
            entity_mapping: Entity text mapping
            
        Returns:
            Dict mapping old_concept_id -> new_concept_id
        """
        concept_mapping = {}
        
        for concept_id, concept in graph.concepts.items():
            concept_text = concept.concept_text
            
            # Check if this entity should be merged
            if concept_text in entity_mapping:
                canonical_entity = entity_mapping[concept_text]
                canonical_concept_id = self._normalize_concept_id(canonical_entity)
                concept_mapping[concept_id] = canonical_concept_id
                
                logger.debug(f"Mapping concept '{concept_text}' -> '{canonical_entity}'")
            else:
                # No change needed
                concept_mapping[concept_id] = concept_id
        
        return concept_mapping
    
    def _merge_concepts(self, graph: TERAGGraph, concept_mapping: Dict[str, str]) -> List[ConceptNode]:
        """
        Create merged concept nodes
        
        Args:
            graph: Original graph
            concept_mapping: Mapping from old to new concept IDs
            
        Returns:
            List of merged ConceptNode objects
        """
        # Group concepts by their target (canonical) concept ID
        concept_groups = defaultdict(list)
        for old_concept_id, new_concept_id in concept_mapping.items():
            if old_concept_id in graph.concepts:
                concept_groups[new_concept_id].append(graph.concepts[old_concept_id])
        
        merged_concepts = []
        
        for canonical_concept_id, concepts_to_merge in concept_groups.items():
            if len(concepts_to_merge) == 1:
                # No merging needed - just update the ID if needed
                concept = concepts_to_merge[0]
                if concept.concept_id != canonical_concept_id:
                    # Create new concept with canonical ID
                    merged_concept = ConceptNode(
                        concept_id=canonical_concept_id,
                        concept_text=concept.concept_text,
                        concept_type=concept.concept_type,
                        frequency=concept.frequency,
                        passage_ids=concept.passage_ids.copy()
                    )
                    merged_concepts.append(merged_concept)
                else:
                    merged_concepts.append(concept)
            else:
                # Merge multiple concepts
                merged_concept = self._merge_concept_nodes(concepts_to_merge, canonical_concept_id)
                merged_concepts.append(merged_concept)
                self.merge_stats["entities_merged"] += len(concepts_to_merge) - 1
                self.merge_stats["clusters_processed"] += 1
        
        return merged_concepts
    
    def _merge_concept_nodes(self, concepts: List[ConceptNode], canonical_concept_id: str) -> ConceptNode:
        """
        Merge multiple ConceptNode objects into one
        
        Args:
            concepts: List of concepts to merge
            canonical_concept_id: ID for the merged concept
            
        Returns:
            Merged ConceptNode
        """
        # Choose canonical text (longest/most complete)
        canonical_text = max(concepts, key=lambda c: len(c.concept_text)).concept_text
        
        # Merge passage IDs
        all_passage_ids = set()
        for concept in concepts:
            all_passage_ids.update(concept.passage_ids)
        
        # Use the most common concept type
        concept_types = [c.concept_type for c in concepts]
        canonical_type = max(set(concept_types), key=concept_types.count)
        
        # Sum frequencies (though this will be recalculated)
        total_frequency = len(all_passage_ids)
        
        merged_concept = ConceptNode(
            concept_id=canonical_concept_id,
            concept_text=canonical_text,
            concept_type=canonical_type,
            frequency=total_frequency,
            passage_ids=all_passage_ids
        )
        
        logger.debug(f"Merged {len(concepts)} concepts into '{canonical_text}' "
                    f"with {len(all_passage_ids)} passages")
        
        return merged_concept
    
    def _rebuild_edges(self, original_graph: TERAGGraph, merged_graph: TERAGGraph, concept_mapping: Dict[str, str]):
        """
        Rebuild edges in merged graph, combining weights for merged entities
        
        Args:
            original_graph: Original graph with old edges
            merged_graph: New graph to add edges to
            concept_mapping: Mapping from old to new concept IDs
        """
        # Group edges by (passage_id, canonical_concept_id)
        edge_weights = defaultdict(list)
        
        # Collect all edge weights, grouping by canonical concept
        for passage_id, concept_weights in original_graph.passage_to_concepts.items():
            for old_concept_id, weight in concept_weights.items():
                if old_concept_id in concept_mapping:
                    canonical_concept_id = concept_mapping[old_concept_id]
                    edge_weights[(passage_id, canonical_concept_id)].append(weight)
        
        # Add merged edges to new graph
        edges_added = 0
        for (passage_id, concept_id), weights in edge_weights.items():
            # Combine weights (take maximum to preserve strongest signal)
            combined_weight = max(weights) if weights else 1.0
            
            # Add edge if both nodes exist
            if passage_id in merged_graph.passages and concept_id in merged_graph.concepts:
                merged_graph.add_edge(passage_id, concept_id, combined_weight)
                edges_added += 1
                
                if len(weights) > 1:
                    self.merge_stats["edges_redirected"] += len(weights) - 1
        
        logger.info(f"Rebuilt {edges_added} edges in merged graph")
    
    def _normalize_concept_id(self, concept_text: str) -> str:
        """Create normalized concept ID from concept text"""
        return " ".join(concept_text.lower().strip().split())
    
    def get_merge_statistics(self) -> Dict:
        """Get statistics about the merge operation"""
        return self.merge_stats.copy()


def apply_deduplication_to_graph(
    graph: TERAGGraph, 
    entity_mapping: Dict[str, str]
) -> TERAGGraph:
    """
    Convenience function to apply entity deduplication to a graph
    
    Args:
        graph: Original TERAG graph
        entity_mapping: Mapping from duplicate to canonical entities
        
    Returns:
        New graph with merged entities
    """
    merger = EntityMerger()
    return merger.apply_entity_mapping(graph, entity_mapping)


def create_deduplication_report(
    original_graph: TERAGGraph,
    merged_graph: TERAGGraph, 
    entity_mapping: Dict[str, str],
    clusters: List[EntityCluster]
) -> Dict:
    """
    Create detailed report about deduplication results
    
    Args:
        original_graph: Graph before deduplication
        merged_graph: Graph after deduplication  
        entity_mapping: Entity mapping used
        clusters: Entity clusters found
        
    Returns:
        Detailed report dictionary
    """
    original_stats = original_graph.get_statistics()
    merged_stats = merged_graph.get_statistics()
    
    # Calculate improvements
    concept_reduction = original_stats['num_concepts'] - merged_stats['num_concepts']
    concept_reduction_pct = (concept_reduction / original_stats['num_concepts'] * 100 
                            if original_stats['num_concepts'] > 0 else 0)
    
    # Analyze clusters
    cluster_sizes = [len(cluster.duplicate_entities) + 1 for cluster in clusters]
    avg_cluster_confidence = (sum(cluster.confidence_score for cluster in clusters) / len(clusters) 
                             if clusters else 0)
    
    report = {
        "summary": {
            "entities_before": original_stats['num_concepts'],
            "entities_after": merged_stats['num_concepts'],
            "entities_merged": concept_reduction,
            "reduction_percentage": concept_reduction_pct,
            "clusters_created": len(clusters)
        },
        "cluster_analysis": {
            "total_clusters": len(clusters),
            "average_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "average_confidence": avg_cluster_confidence
        },
        "graph_changes": {
            "passages_before": original_stats['num_passages'],
            "passages_after": merged_stats['num_passages'],
            "edges_before": original_stats['num_edges'], 
            "edges_after": merged_stats['num_edges'],
            "avg_concepts_per_passage_before": original_stats['avg_concepts_per_passage'],
            "avg_concepts_per_passage_after": merged_stats['avg_concepts_per_passage']
        },
        "top_merges": [
            {
                "canonical_entity": cluster.canonical_entity,
                "merged_entities": list(cluster.duplicate_entities),
                "confidence": cluster.confidence_score
            }
            for cluster in sorted(clusters, key=lambda c: c.confidence_score, reverse=True)[:10]
        ]
    }
    
    return report


if __name__ == "__main__":
    # Test entity merger
    print("🔄 ENTITY MERGER TEST")
    print("=" * 60)
    
    # This would normally be called with real graph and mapping
    print("💡 To test entity merger:")
    print("   1. Load a TERAG graph")
    print("   2. Run entity deduplication") 
    print("   3. Apply merger with entity mapping")
    print("   4. Compare before/after statistics")
    
    print("\n📊 Example usage:")
    print("""
    from entity_deduplicator import deduplicate_graph_entities
    from entity_merger import apply_deduplication_to_graph, create_deduplication_report
    
    # Deduplicate
    entity_mapping, clusters = deduplicate_graph_entities(original_graph)
    
    # Merge
    merged_graph = apply_deduplication_to_graph(original_graph, entity_mapping)
    
    # Report
    report = create_deduplication_report(original_graph, merged_graph, entity_mapping, clusters)
    """)