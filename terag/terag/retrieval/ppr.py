"""
TERAG Personalized PageRank Retrieval

Implements Personalized PageRank (PPR) algorithm for passage retrieval:
1. Match query entities to graph concepts
2. Create restart vector based on frequency and semantic weights
3. Run PPR to propagate relevance through graph
4. Rank passages by final PPR scores
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

from terag.graph.builder import TERAGGraph


@dataclass
class RetrievalResult:
    """Result of TERAG retrieval"""
    passage_id: str
    content: str
    score: float
    matched_concepts: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval performance"""
    num_query_entities: int
    num_matched_concepts: int
    ppr_iterations: int
    retrieval_time: float
    num_results: int


class PersonalizedPageRank:
    """
    Personalized PageRank algorithm for TERAG retrieval

    PPR Formula:
    π(t+1) = α * R + (1 - α) * M^T * π(t)

    Where:
    - π: PageRank vector
    - α: Teleportation (restart) probability (typically 0.15)
    - R: Restart vector (personalization)
    - M: Transition matrix (normalized adjacency matrix)
    """

    def __init__(
        self,
        graph: TERAGGraph,
        alpha: float = 0.15,  # Teleportation probability
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ):
        self.graph = graph
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Build transition matrix structures
        self._build_transition_structures()

    def _build_transition_structures(self):
        """
        Build transition matrix structures for efficient PPR computation

        Instead of storing full matrix, use adjacency lists with normalized weights
        """
        # Create node index mapping
        self.node_to_idx = {}
        self.idx_to_node = {}

        idx = 0
        # Add passage nodes
        for passage_id in self.graph.passages:
            self.node_to_idx[passage_id] = idx
            self.idx_to_node[idx] = passage_id
            idx += 1

        # Add concept nodes
        for concept_id in self.graph.concepts:
            self.node_to_idx[concept_id] = idx
            self.idx_to_node[idx] = concept_id
            idx += 1

        self.num_nodes = idx

        # Build normalized adjacency lists (transition probabilities)
        self.transitions = defaultdict(list)  # node_idx -> [(neighbor_idx, weight), ...]

        # Passage -> Concept edges
        for passage_id, concept_ids in self.graph.passage_to_concepts.items():
            passage_idx = self.node_to_idx[passage_id]
            num_neighbors = len(concept_ids)

            if num_neighbors > 0:
                weight = 1.0 / num_neighbors
                for concept_id in concept_ids:
                    concept_idx = self.node_to_idx[concept_id]
                    self.transitions[passage_idx].append((concept_idx, weight))

        # Concept -> Passage edges
        for concept_id, passage_ids in self.graph.concept_to_passages.items():
            concept_idx = self.node_to_idx[concept_id]
            num_neighbors = len(passage_ids)

            if num_neighbors > 0:
                weight = 1.0 / num_neighbors
                for passage_id in passage_ids:
                    passage_idx = self.node_to_idx[passage_id]
                    self.transitions[concept_idx].append((passage_idx, weight))

    def compute_ppr(
        self,
        restart_vector: Dict[str, float],
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Compute Personalized PageRank

        Args:
            restart_vector: Dict mapping node_id -> restart probability
            verbose: Print progress

        Returns:
            Dict mapping node_id -> PPR score
        """
        # Convert restart vector to numpy array
        R = np.zeros(self.num_nodes)
        for node_id, prob in restart_vector.items():
            if node_id in self.node_to_idx:
                idx = self.node_to_idx[node_id]
                R[idx] = prob

        # Normalize restart vector
        R_sum = R.sum()
        if R_sum > 0:
            R = R / R_sum
        else:
            # Uniform restart if empty
            R = np.ones(self.num_nodes) / self.num_nodes

        # Initialize PageRank vector
        pi = R.copy()

        # Power iteration
        for iteration in range(self.max_iterations):
            pi_new = self.alpha * R

            # Add transition contributions: (1 - α) * M^T * π
            for node_idx in range(self.num_nodes):
                for neighbor_idx, weight in self.transitions[node_idx]:
                    pi_new[neighbor_idx] += (1 - self.alpha) * weight * pi[node_idx]

            # Check convergence
            diff = np.abs(pi_new - pi).sum()
            pi = pi_new

            if verbose and (iteration + 1) % 10 == 0:
                print(f"  PPR iteration {iteration + 1}: diff={diff:.6f}")

            if diff < self.tolerance:
                if verbose:
                    print(f"  PPR converged in {iteration + 1} iterations")
                break

        # Convert back to dictionary
        ppr_scores = {}
        for node_id, idx in self.node_to_idx.items():
            ppr_scores[node_id] = float(pi[idx])

        return ppr_scores


class TERAGRetriever:
    """
    Main TERAG retrieval interface

    Combines:
    1. Query NER
    2. Concept matching
    3. Restart vector construction (frequency + semantic weights)
    4. Personalized PageRank
    5. Passage ranking
    """

    def __init__(
        self,
        graph: TERAGGraph,
        embedding_model: Optional[object] = None,
        alpha: float = 0.15,
        ppr_max_iterations: int = 100
    ):
        self.graph = graph
        self.embedding_model = embedding_model
        self.ppr = PersonalizedPageRank(
            graph=graph,
            alpha=alpha,
            max_iterations=ppr_max_iterations
        )

        # Pre-compute concept embeddings if model available
        self.concept_embeddings = {}
        if embedding_model:
            self._precompute_concept_embeddings()

    def _precompute_concept_embeddings(self):
        """Pre-compute embeddings for all concepts"""
        print("Pre-computing concept embeddings...")
        concept_texts = [c.concept_text for c in self.graph.concepts.values()]

        if hasattr(self.embedding_model, 'encode'):
            embeddings = self.embedding_model.encode(concept_texts, show_progress_bar=True)

            for concept_id, embedding in zip(self.graph.concepts.keys(), embeddings):
                self.concept_embeddings[concept_id] = embedding

    def retrieve(
        self,
        query: str,
        query_entities: List[str],
        top_k: int = 10,
        semantic_weight: float = 0.5,
        frequency_weight: float = 0.5,
        verbose: bool = False
    ) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """
        Retrieve relevant passages using TERAG

        Args:
            query: User query text
            query_entities: Extracted query entities
            top_k: Number of passages to return
            semantic_weight: Weight for semantic similarity
            frequency_weight: Weight for frequency-based score
            verbose: Print progress

        Returns:
            (results, metrics)
        """
        start_time = time.time()

        if verbose:
            print(f"\nTERAG Retrieval for: '{query}'")
            print(f"Query entities: {query_entities}")

        # Step 1: Match query entities to graph concepts
        matched_concepts = self._match_entities_to_concepts(query_entities)

        if verbose:
            print(f"Matched concepts: {len(matched_concepts)}")

        if not matched_concepts:
            # No matches - return empty
            return [], RetrievalMetrics(
                num_query_entities=len(query_entities),
                num_matched_concepts=0,
                ppr_iterations=0,
                retrieval_time=time.time() - start_time,
                num_results=0
            )

        # Step 2: Build restart vector with frequency + semantic weights
        restart_vector = self._build_restart_vector(
            query=query,
            matched_concepts=matched_concepts,
            semantic_weight=semantic_weight,
            frequency_weight=frequency_weight
        )

        if verbose:
            print(f"Restart vector size: {len(restart_vector)}")

        # Step 3: Run Personalized PageRank
        ppr_scores = self.ppr.compute_ppr(restart_vector, verbose=verbose)

        # Step 4: Extract and rank passages
        passage_scores = {
            pid: score
            for pid, score in ppr_scores.items()
            if pid in self.graph.passages
        }

        # Sort by score
        ranked_passages = sorted(
            passage_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Step 5: Build results
        results = []
        for passage_id, score in ranked_passages:
            passage = self.graph.passages[passage_id]

            # Find which concepts matched
            matched = [
                c for c in matched_concepts
                if c in passage.concepts
            ]

            results.append(RetrievalResult(
                passage_id=passage_id,
                content=passage.content,
                score=score,
                matched_concepts=matched,
                metadata=passage.metadata
            ))

        # Build metrics
        metrics = RetrievalMetrics(
            num_query_entities=len(query_entities),
            num_matched_concepts=len(matched_concepts),
            ppr_iterations=self.ppr.max_iterations,  # Approximate
            retrieval_time=time.time() - start_time,
            num_results=len(results)
        )

        if verbose:
            print(f"\nRetrieved {len(results)} passages in {metrics.retrieval_time:.3f}s")

        return results, metrics

    def _match_entities_to_concepts(self, query_entities: List[str]) -> Set[str]:
        """
        Match query entities to graph concept IDs

        Uses fuzzy matching to handle variations
        """
        matched = set()

        for entity in query_entities:
            entity_lower = entity.lower().strip()

            # Exact match
            if entity_lower in self.graph.concepts:
                matched.add(entity_lower)
                continue

            # Partial match (entity contained in concept or vice versa)
            for concept_id, concept in self.graph.concepts.items():
                concept_text_lower = concept.concept_text.lower()

                if entity_lower in concept_text_lower or concept_text_lower in entity_lower:
                    matched.add(concept_id)

        return matched

    def _build_restart_vector(
        self,
        query: str,
        matched_concepts: Set[str],
        semantic_weight: float,
        frequency_weight: float
    ) -> Dict[str, float]:
        """
        Build restart vector combining frequency and semantic weights

        weight(concept) = frequency_weight * inv_freq(concept) + semantic_weight * sem_sim(query, concept)
        """
        restart_vector = {}

        # Get total number of passages for IDF calculation
        total_passages = len(self.graph.passages)

        # Compute query embedding if model available
        query_embedding = None
        if self.embedding_model and hasattr(self.embedding_model, 'encode'):
            query_embedding = self.embedding_model.encode([query])[0]

        for concept_id in matched_concepts:
            concept = self.graph.concepts[concept_id]

            # Frequency-based weight (inverse document frequency)
            freq = concept.frequency
            idf = np.log(total_passages / (freq + 1))
            freq_score = idf / 10  # Normalize

            # Semantic weight
            sem_score = 0.5  # Default if no embeddings

            if query_embedding is not None and concept_id in self.concept_embeddings:
                concept_emb = self.concept_embeddings[concept_id]
                # Cosine similarity (assuming normalized embeddings)
                similarity = np.dot(query_embedding, concept_emb)
                sem_score = max(0, similarity)  # Clamp to [0, 1]

            # Combined weight
            weight = frequency_weight * freq_score + semantic_weight * sem_score

            restart_vector[concept_id] = weight

        # Normalize
        total = sum(restart_vector.values())
        if total > 0:
            restart_vector = {k: v / total for k, v in restart_vector.items()}

        return restart_vector


if __name__ == "__main__":
    # Test PPR with simple graph
    from graph_builder import TERAGGraph, PassageNode, ConceptNode

    print("Testing TERAG Retrieval")
    print("=" * 60)

    # Create simple test graph
    graph = TERAGGraph()

    # Add passages
    passages = [
        PassageNode("p1", "Apple announced revenue growth in Q4 2024.", 0),
        PassageNode("p2", "Microsoft reported strong cloud performance.", 1),
        PassageNode("p3", "Apple and Microsoft compete in technology.", 2),
    ]

    for p in passages:
        graph.add_passage(p)

    # Add concepts
    concepts = [
        ConceptNode("apple", "Apple", "named_entity"),
        ConceptNode("microsoft", "Microsoft", "named_entity"),
        ConceptNode("revenue", "revenue", "document_concept"),
        ConceptNode("q4 2024", "Q4 2024", "date"),
    ]

    for c in concepts:
        graph.add_concept(c)

    # Add edges
    graph.add_edge("p1", "apple")
    graph.add_edge("p1", "revenue")
    graph.add_edge("p1", "q4 2024")
    graph.add_edge("p2", "microsoft")
    graph.add_edge("p3", "apple")
    graph.add_edge("p3", "microsoft")

    # Test retrieval
    retriever = TERAGRetriever(graph, alpha=0.15)

    query = "What was Apple's revenue?"
    query_entities = ["Apple", "revenue"]

    results, metrics = retriever.retrieve(
        query=query,
        query_entities=query_entities,
        top_k=3,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [Score: {result.score:.4f}] {result.passage_id}")
        print(f"   {result.content}")
        print(f"   Matched: {result.matched_concepts}")

    print(f"\nMetrics: {metrics}")
