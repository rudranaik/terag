"""
TERAG Query Explainer

Provides detailed step-by-step explanation of how queries are processed
and how the dual-layer retrieval system finds relevant passages.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from graph_builder import TERAGGraph
from query_processor import ProcessedQuery, QueryProcessor
from ppr_retriever import TERAGRetriever as PPRRetriever
from semantic_retriever import SemanticRetriever
from hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class EntityExplanation:
    """Explanation of how an entity was extracted and matched"""
    query_text: str
    entity_text: str
    graph_concept_id: str
    similarity_score: float
    entity_type: str
    frequency_in_graph: int
    connected_passages: int


@dataclass
class PathExplanation:
    """Explanation of a graph path from entity to passage"""
    entity_id: str
    entity_text: str
    passage_id: str
    path_length: int
    path_description: str
    edge_weight: float
    ppr_contribution: float


@dataclass
class RetrievalExplanation:
    """Complete explanation of query retrieval process"""
    query: str
    processed_query: ProcessedQuery
    entity_explanations: List[EntityExplanation]
    ppr_paths: List[PathExplanation]
    semantic_matches: List[Tuple[str, float, str]]  # (passage_id, score, explanation)
    hybrid_fusion: Dict[str, Dict]  # passage_id -> fusion details
    final_ranking_explanation: List[str]


class QueryExplainer:
    """
    Explains the step-by-step process of TERAG query retrieval
    """
    
    def __init__(self, graph: TERAGGraph, hybrid_retriever: HybridRetriever):
        self.graph = graph
        self.hybrid_retriever = hybrid_retriever
        self.query_processor = hybrid_retriever.query_processor
        self.ppr_retriever = hybrid_retriever.ppr_retriever
        self.semantic_retriever = hybrid_retriever.semantic_retriever
        
    def explain_query_retrieval(
        self, 
        query: str, 
        top_k: int = 10,
        show_all_entities: bool = True,
        show_graph_paths: bool = True,
        show_semantic_details: bool = True
    ) -> RetrievalExplanation:
        """
        Provide complete explanation of query retrieval process
        
        Args:
            query: User query to explain
            top_k: Number of results to analyze
            show_all_entities: Show all extracted entities (not just top matches)
            show_graph_paths: Show detailed graph paths
            show_semantic_details: Show semantic matching details
            
        Returns:
            Complete RetrievalExplanation object
        """
        print(f"\n🔍 EXPLAINING QUERY RETRIEVAL")
        print(f"📋 Query: '{query}'")
        print("=" * 80)
        
        # Step 1: Query Processing and Entity Extraction
        print(f"\n📝 STEP 1: QUERY PROCESSING & ENTITY EXTRACTION")
        print("-" * 60)
        
        processed_query = self.query_processor.process_query(query)
        entity_explanations = self._explain_entity_extraction(processed_query, show_all_entities)
        
        # Step 2: PPR Graph Traversal
        print(f"\n🕸️  STEP 2: GRAPH-BASED RETRIEVAL (PPR)")
        print("-" * 60)
        
        ppr_paths = []
        if processed_query.extracted_entities:
            ppr_paths = self._explain_ppr_retrieval(processed_query, show_graph_paths)
        else:
            print("❌ No entities extracted - skipping PPR retrieval")
        
        # Step 3: Semantic Retrieval
        print(f"\n🧠 STEP 3: SEMANTIC RETRIEVAL")
        print("-" * 60)
        
        semantic_matches = self._explain_semantic_retrieval(processed_query, show_semantic_details)
        
        # Step 4: Hybrid Fusion
        print(f"\n🔄 STEP 4: HYBRID FUSION & FINAL RANKING")
        print("-" * 60)
        
        # Get actual hybrid results for detailed fusion analysis
        hybrid_results, _ = self.hybrid_retriever.retrieve(query, top_k=top_k)
        
        hybrid_fusion, final_ranking_explanation = self._explain_hybrid_fusion(
            processed_query, ppr_paths, semantic_matches, hybrid_results
        )
        
        # Create complete explanation
        explanation = RetrievalExplanation(
            query=query,
            processed_query=processed_query,
            entity_explanations=entity_explanations,
            ppr_paths=ppr_paths,
            semantic_matches=semantic_matches,
            hybrid_fusion=hybrid_fusion,
            final_ranking_explanation=final_ranking_explanation
        )
        
        print(f"\n✅ QUERY EXPLANATION COMPLETE")
        print("=" * 80)
        
        return explanation
    
    def _explain_entity_extraction(
        self, 
        processed_query: ProcessedQuery, 
        show_all: bool
    ) -> List[EntityExplanation]:
        """Explain how entities were extracted from the query"""
        
        print(f"🎯 Query: '{processed_query.original_query}'")
        print(f"🔧 Cleaned: '{processed_query.cleaned_query}'")
        print(f"🎯 Confidence: {processed_query.confidence_score:.3f}")
        print(f"📊 Entities found: {len(processed_query.extracted_entities)}")
        
        entity_explanations = []
        
        if not processed_query.extracted_entities:
            print("❌ No entities extracted from query")
            return entity_explanations
        
        print(f"\n📋 EXTRACTED ENTITIES:")
        for i, entity in enumerate(processed_query.extracted_entities):
            # Get additional info about this entity in the graph
            concept = self.graph.concepts.get(entity.graph_concept_id)
            if concept:
                frequency = concept.frequency
                connected_passages = len(concept.passage_ids)
            else:
                frequency = 0
                connected_passages = 0
            
            explanation = EntityExplanation(
                query_text=processed_query.original_query,
                entity_text=entity.entity_text,
                graph_concept_id=entity.graph_concept_id,
                similarity_score=entity.similarity_score,
                entity_type=entity.entity_type,
                frequency_in_graph=frequency,
                connected_passages=connected_passages
            )
            entity_explanations.append(explanation)
            
            print(f"  {i+1}. '{entity.entity_text}' (similarity: {entity.similarity_score:.3f})")
            print(f"     📍 Graph ID: {entity.graph_concept_id}")
            print(f"     🏷️  Type: {entity.entity_type}")
            print(f"     📊 Appears in {connected_passages} passages, frequency: {frequency}")
            
            if not show_all and i >= 4:  # Limit to top 5 if not showing all
                remaining = len(processed_query.extracted_entities) - i - 1
                if remaining > 0:
                    print(f"     ... and {remaining} more entities")
                break
        
        return entity_explanations
    
    def _explain_ppr_retrieval(
        self, 
        processed_query: ProcessedQuery, 
        show_paths: bool
    ) -> List[PathExplanation]:
        """Explain how PPR traverses the graph from entities to passages"""
        
        print(f"🚀 Running PPR from {len(processed_query.extracted_entities)} entities...")
        
        # Run PPR retrieval to get results
        query_entities = [entity.entity_text for entity in processed_query.extracted_entities]
        ppr_results, ppr_metrics = self.ppr_retriever.retrieve(
            query=processed_query.original_query,
            query_entities=query_entities,
            top_k=20,  # Get more for analysis
            verbose=False
        )
        
        print(f"📈 PPR found {len(ppr_results)} relevant passages")
        print(f"⏱️  Processing time: {ppr_metrics.retrieval_time:.3f}s")
        
        path_explanations = []
        
        if show_paths and ppr_results:
            print(f"\n🛤️  TOP GRAPH PATHS:")
            
            # Analyze top PPR results to show paths
            for i, result in enumerate(ppr_results[:8]):  # Show top 8 paths
                passage_id = result.passage_id
                ppr_score = result.score
                
                # Find which entities led to this passage
                passage_neighbors = self.graph.get_passage_neighbors(passage_id)
                
                # Find connections to query entities
                entity_connections = []
                for entity in processed_query.extracted_entities:
                    concept_id = entity.graph_concept_id
                    if concept_id in passage_neighbors:
                        edge_weight = passage_neighbors[concept_id]
                        entity_connections.append((entity, edge_weight))
                
                # Determine path description
                if entity_connections:
                    # Direct connection
                    path_desc = f"Direct: {' + '.join([e[0].entity_text for e in entity_connections])}"
                    path_length = 1
                    primary_entity = entity_connections[0][0]
                    edge_weight = entity_connections[0][1]
                else:
                    # Multi-hop connection (harder to trace, but PPR found it)
                    path_desc = "Multi-hop: Connected through intermediate entities"
                    path_length = 2  # Approximate
                    primary_entity = processed_query.extracted_entities[0]  # Use first entity
                    edge_weight = 0.1  # Estimated
                
                path_explanation = PathExplanation(
                    entity_id=primary_entity.graph_concept_id,
                    entity_text=primary_entity.entity_text,
                    passage_id=passage_id,
                    path_length=path_length,
                    path_description=path_desc,
                    edge_weight=edge_weight,
                    ppr_contribution=ppr_score
                )
                path_explanations.append(path_explanation)
                
                # Get passage preview
                passage = self.graph.passages.get(passage_id)
                content_preview = passage.content[:100] + "..." if passage and len(passage.content) > 100 else (passage.content if passage else "")
                
                print(f"  {i+1}. [{ppr_score:.4f}] {passage_id[:8]}... via {path_desc}")
                print(f"     Edge weight: {edge_weight:.3f}, Path length: {path_length}")
                print(f"     Content: {content_preview}")
        else:
            print("🔍 Graph path details not shown (use show_graph_paths=True to see)")
        
        return path_explanations
    
    def _explain_semantic_retrieval(
        self, 
        processed_query: ProcessedQuery, 
        show_details: bool
    ) -> List[Tuple[str, float, str]]:
        """Explain how semantic retrieval finds similar passages"""
        
        print(f"🧠 Running semantic similarity matching...")
        
        # Run semantic retrieval
        semantic_results = self.semantic_retriever.retrieve_passages(
            processed_query=processed_query,
            top_k=20  # Get more for analysis
        )
        
        print(f"📈 Semantic retrieval found {len(semantic_results)} passages above threshold")
        print(f"🎯 Similarity threshold: {self.semantic_retriever.min_similarity_threshold}")
        
        semantic_matches = []
        
        if show_details and semantic_results:
            print(f"\n🎯 TOP SEMANTIC MATCHES:")
            
            for i, (passage_id, similarity_score) in enumerate(semantic_results[:8]):
                # Get passage content for analysis
                passage = self.graph.passages.get(passage_id)
                if not passage:
                    continue
                
                content_preview = passage.content[:100] + "..." if len(passage.content) > 100 else passage.content
                
                # Generate explanation based on similarity score
                if similarity_score > 0.8:
                    explanation = "Very high semantic similarity - query and passage discuss very similar topics"
                elif similarity_score > 0.6:
                    explanation = "High semantic similarity - strong topical overlap"
                elif similarity_score > 0.4:
                    explanation = "Moderate semantic similarity - some shared themes"
                else:
                    explanation = "Low semantic similarity - weak topical connection"
                
                semantic_matches.append((passage_id, similarity_score, explanation))
                
                print(f"  {i+1}. [{similarity_score:.4f}] {passage_id[:8]}...")
                print(f"     {explanation}")
                print(f"     Content: {content_preview}")
        else:
            # Still collect basic info even if not showing details
            for passage_id, similarity_score in semantic_results:
                if similarity_score > 0.6:
                    explanation = "High semantic similarity"
                elif similarity_score > 0.4:
                    explanation = "Moderate semantic similarity"
                else:
                    explanation = "Low semantic similarity"
                semantic_matches.append((passage_id, similarity_score, explanation))
            
            print("🔍 Semantic match details not shown (use show_semantic_details=True to see)")
        
        return semantic_matches
    
    def _explain_hybrid_fusion(
        self,
        processed_query: ProcessedQuery,
        ppr_paths: List[PathExplanation],
        semantic_matches: List[Tuple[str, float, str]],
        hybrid_results: List
    ) -> Tuple[Dict[str, Dict], List[str]]:
        """Explain how PPR and semantic scores were fused"""
        
        print(f"🔄 Combining PPR and semantic retrieval results...")
        print(f"⚖️  Fusion weights: PPR={self.hybrid_retriever.ppr_weight}, Semantic={self.hybrid_retriever.semantic_weight}")
        print(f"🔧 Fusion method: {self.hybrid_retriever.score_fusion_method}")
        
        # Collect fusion details for each result
        hybrid_fusion = {}
        
        print(f"\n🏆 FINAL HYBRID RESULTS:")
        for i, result in enumerate(hybrid_results[:10]):
            passage_id = result.passage_id
            
            fusion_details = {
                "ppr_score": result.ppr_score,
                "semantic_score": result.semantic_score,
                "hybrid_score": result.hybrid_score,
                "confidence": result.confidence,
                "entity_matches": result.entity_matches,
                "explanation": result.explanation
            }
            
            hybrid_fusion[passage_id] = fusion_details
            
            # Get passage preview
            content_preview = result.content[:80] + "..." if len(result.content) > 80 else result.content
            
            print(f"  {i+1}. [{result.hybrid_score:.4f}] {passage_id[:8]}...")
            print(f"     PPR: {result.ppr_score:.4f}, Semantic: {result.semantic_score:.4f}")
            print(f"     Confidence: {result.confidence:.3f}")
            print(f"     {result.explanation}")
            print(f"     Content: {content_preview}")
            if result.entity_matches:
                print(f"     🎯 Matched entities: {', '.join(result.entity_matches)}")
        
        # Generate ranking explanation
        final_ranking_explanation = [
            f"Hybrid fusion combined {len([r for r in hybrid_results if r.ppr_score > 0])} PPR results "
            f"and {len([r for r in hybrid_results if r.semantic_score > 0])} semantic results",
            f"Used {self.hybrid_retriever.score_fusion_method} fusion with weights PPR:{self.hybrid_retriever.ppr_weight}, Semantic:{self.hybrid_retriever.semantic_weight}",
            f"Final ranking prioritizes passages with both graph connections and semantic similarity"
        ]
        
        # Count result types
        ppr_only = len([r for r in hybrid_results if r.ppr_score > 0 and r.semantic_score == 0])
        semantic_only = len([r for r in hybrid_results if r.ppr_score == 0 and r.semantic_score > 0])
        combined = len([r for r in hybrid_results if r.ppr_score > 0 and r.semantic_score > 0])
        
        print(f"\n📊 RESULT BREAKDOWN:")
        print(f"   🕸️  PPR only: {ppr_only} results")
        print(f"   🧠 Semantic only: {semantic_only} results")  
        print(f"   🔄 Both approaches: {combined} results")
        print(f"   📈 Total unique: {len(hybrid_results)} results")
        
        return hybrid_fusion, final_ranking_explanation


def explain_query_step_by_step(
    query: str,
    graph: TERAGGraph,
    hybrid_retriever: HybridRetriever,
    top_k: int = 10,
    detailed: bool = True
) -> RetrievalExplanation:
    """
    Convenience function to explain a query step-by-step
    
    Args:
        query: Query to explain
        graph: TERAG graph
        hybrid_retriever: Configured hybrid retriever
        top_k: Number of results to analyze
        detailed: Whether to show detailed explanations
        
    Returns:
        Complete RetrievalExplanation
    """
    explainer = QueryExplainer(graph, hybrid_retriever)
    
    explanation = explainer.explain_query_retrieval(
        query=query,
        top_k=top_k,
        show_all_entities=detailed,
        show_graph_paths=detailed,
        show_semantic_details=detailed
    )
    
    return explanation


if __name__ == "__main__":
    print("🔍 QUERY EXPLAINER")
    print("=" * 60)
    print("This module provides detailed explanations of TERAG query retrieval.")
    print("Use it with a graph and hybrid retriever to understand the retrieval process.")
    
    print(f"\n📋 Example usage:")
    print("""
    from query_explainer import explain_query_step_by_step
    from hybrid_retriever import create_hybrid_retriever
    from graph_builder import TERAGGraph
    from embedding_manager import EmbeddingManager
    
    # Load components
    graph = TERAGGraph.load_from_file("graph.json")
    embedding_manager = EmbeddingManager()
    retriever = create_hybrid_retriever(graph, embedding_manager)
    
    # Explain a query step by step
    explanation = explain_query_step_by_step(
        "How is the music business performing?",
        graph, 
        retriever,
        detailed=True
    )
    """)