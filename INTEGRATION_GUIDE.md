# TERAG Integration Guide

## Integrating TERAG with Agentic RAG

This guide shows how to combine TERAG (Token-Efficient Graph-Based RAG) with the existing agentic retrieval system for enhanced performance.

## Architecture Overview

```
Query → TERAG (Graph) → Agentic RAG (Dense+Sparse) → Combined Results
         │                    │
         └────────────────────┴──→ Hybrid Ranking → Final Answer
```

### Complementary Strengths

| Aspect | TERAG | Agentic RAG |
|--------|-------|-------------|
| **Retrieval Strategy** | Graph-based (PPR) | Dense + Sparse (Hybrid) |
| **Strength** | Multi-hop reasoning, entity relationships | Semantic similarity, keyword matching |
| **Cost** | 3-11% token consumption | Moderate (no graph construction) |
| **Speed** | Fast (graph traversal) | Fast (vector search) |
| **Coverage** | Entity-centric queries | Broad semantic queries |

## Integration Patterns

### Pattern 1: Sequential (TERAG → Agentic)

Use TERAG for initial broad retrieval, then Agentic RAG for refinement.

```python
from terag import TERAG, TERAGConfig
from agentic_retrieval import create_retriever_from_chunks, RetrievalConfig

# Initialize both systems
terag_config = TERAGConfig(min_concept_freq=2, top_k=20)
terag = TERAG.from_chunks_file("chunks.json", config=terag_config)

agentic_config = RetrievalConfig(max_query_rewrites=2, top_k=10)
agentic = create_retriever_from_chunks("chunks.json", config=agentic_config)

# Retrieve
query = "What was the revenue growth?"

# Step 1: TERAG for graph-based retrieval
terag_results, _ = terag.retrieve(query, top_k=20)

# Step 2: Agentic for dense+sparse retrieval
agentic_results, _ = agentic.retrieve(query)

# Step 3: Combine and deduplicate
combined = merge_results(terag_results, agentic_results)
```

### Pattern 2: Parallel (TERAG || Agentic)

Run both retrievers in parallel and merge results.

```python
import concurrent.futures

def parallel_retrieve(query: str):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both retrieval tasks
        terag_future = executor.submit(terag.retrieve, query, top_k=15)
        agentic_future = executor.submit(agentic.retrieve, query)

        # Wait for both
        terag_results, terag_metrics = terag_future.result()
        agentic_results, agentic_metrics = agentic_future.result()

    # Merge with weighted scoring
    combined = merge_with_weights(
        terag_results,
        agentic_results,
        terag_weight=0.4,
        agentic_weight=0.6
    )

    return combined
```

### Pattern 3: Query Routing

Route queries to the best retriever based on query type.

```python
def route_and_retrieve(query: str):
    query_type = classify_query(query)

    if query_type == "entity_focused":
        # Entity-centric queries → TERAG
        # Examples: "Who is the CEO?", "What companies are mentioned?"
        results, _ = terag.retrieve(query, top_k=10)

    elif query_type == "semantic_broad":
        # Broad semantic queries → Agentic RAG
        # Examples: "Explain the business model", "What are the risks?"
        results, _ = agentic.retrieve(query)

    else:
        # Hybrid for complex queries
        terag_results, _ = terag.retrieve(query, top_k=15)
        agentic_results, _ = agentic.retrieve(query)
        results = merge_results(terag_results, agentic_results)

    return results

def classify_query(query: str) -> str:
    """Simple query classification"""
    entity_keywords = ["who", "which company", "what person", "ceo", "board"]
    semantic_keywords = ["explain", "describe", "what is", "how does"]

    query_lower = query.lower()

    if any(kw in query_lower for kw in entity_keywords):
        return "entity_focused"
    elif any(kw in query_lower for kw in semantic_keywords):
        return "semantic_broad"
    else:
        return "hybrid"
```

## Complete Integration Example

```python
#!/usr/bin/env python
"""
Complete TERAG + Agentic RAG Integration
"""

import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass

# TERAG imports
from terag import TERAG, TERAGConfig, RetrievalResult as TERAGResult

# Agentic RAG imports
sys.path.insert(0, "..")
from agentic_retrieval import create_retriever_from_chunks, RetrievalConfig


@dataclass
class HybridResult:
    """Combined result from both retrievers"""
    content: str
    score: float
    source: str  # "TERAG", "Agentic", or "Both"
    terag_score: float = 0.0
    agentic_score: float = 0.0
    matched_concepts: List[str] = None


class HybridRetriever:
    """
    Hybrid retrieval system combining TERAG and Agentic RAG
    """

    def __init__(
        self,
        chunks_file: str,
        terag_config: TERAGConfig = None,
        agentic_config: RetrievalConfig = None,
        terag_weight: float = 0.5,
        agentic_weight: float = 0.5
    ):
        self.terag_weight = terag_weight
        self.agentic_weight = agentic_weight

        # Initialize TERAG
        print("Initializing TERAG...")
        self.terag = TERAG.from_chunks_file(
            chunks_file,
            config=terag_config or TERAGConfig(min_concept_freq=2),
            verbose=True
        )

        # Initialize Agentic RAG
        print("\nInitializing Agentic RAG...")
        self.agentic = create_retriever_from_chunks(
            chunks_file,
            config=agentic_config or RetrievalConfig(max_query_rewrites=2)
        )

        print("\nHybrid retriever ready!")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        strategy: str = "parallel",  # "parallel", "sequential", "route"
        verbose: bool = False
    ) -> List[HybridResult]:
        """
        Retrieve using hybrid strategy
        """
        if strategy == "parallel":
            return self._retrieve_parallel(query, top_k, verbose)
        elif strategy == "sequential":
            return self._retrieve_sequential(query, top_k, verbose)
        elif strategy == "route":
            return self._retrieve_routed(query, top_k, verbose)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _retrieve_parallel(
        self,
        query: str,
        top_k: int,
        verbose: bool
    ) -> List[HybridResult]:
        """Parallel retrieval and merging"""
        if verbose:
            print(f"\n[Parallel] Retrieving for: {query}")

        # TERAG retrieval
        terag_results, terag_metrics = self.terag.retrieve(
            query,
            top_k=top_k * 2,  # Get more for merging
            verbose=verbose
        )

        # Agentic retrieval
        agentic_results, agentic_metrics = self.agentic.retrieve(
            query,
            verbose=verbose
        )

        # Merge results
        hybrid_results = self._merge_results(
            terag_results,
            agentic_results,
            top_k
        )

        if verbose:
            print(f"\n[Parallel] Combined {len(terag_results)} TERAG + "
                  f"{len(agentic_results)} Agentic → {len(hybrid_results)} hybrid")

        return hybrid_results

    def _retrieve_sequential(
        self,
        query: str,
        top_k: int,
        verbose: bool
    ) -> List[HybridResult]:
        """Sequential: TERAG first, then Agentic refinement"""
        if verbose:
            print(f"\n[Sequential] Step 1: TERAG retrieval")

        # Step 1: TERAG for broad entity-based retrieval
        terag_results, _ = self.terag.retrieve(query, top_k=top_k * 3, verbose=verbose)

        if verbose:
            print(f"\n[Sequential] Step 2: Agentic refinement")

        # Step 2: Agentic for semantic refinement
        agentic_results, _ = self.agentic.retrieve(query, verbose=verbose)

        # Merge with higher weight on Agentic (refinement)
        hybrid_results = self._merge_results(
            terag_results,
            agentic_results,
            top_k,
            terag_weight=0.3,
            agentic_weight=0.7
        )

        return hybrid_results

    def _retrieve_routed(
        self,
        query: str,
        top_k: int,
        verbose: bool
    ) -> List[HybridResult]:
        """Route query to best retriever"""
        query_type = self._classify_query(query)

        if verbose:
            print(f"\n[Routing] Query type: {query_type}")

        if query_type == "entity":
            # Favor TERAG for entity queries
            results, _ = self.terag.retrieve(query, top_k=top_k, verbose=verbose)
            return [self._terag_to_hybrid(r) for r in results]

        elif query_type == "semantic":
            # Favor Agentic for semantic queries
            results, _ = self.agentic.retrieve(query, verbose=verbose)
            return [self._agentic_to_hybrid(r) for r in results[:top_k]]

        else:
            # Use parallel for complex queries
            return self._retrieve_parallel(query, top_k, verbose)

    def _merge_results(
        self,
        terag_results: List[TERAGResult],
        agentic_results: List[Dict],
        top_k: int,
        terag_weight: float = None,
        agentic_weight: float = None
    ) -> List[HybridResult]:
        """Merge and deduplicate results from both retrievers"""
        if terag_weight is None:
            terag_weight = self.terag_weight
        if agentic_weight is None:
            agentic_weight = self.agentic_weight

        # Normalize weights
        total_weight = terag_weight + agentic_weight
        terag_weight /= total_weight
        agentic_weight /= total_weight

        # Create content → result mapping
        content_map = {}

        # Add TERAG results
        for r in terag_results:
            content_key = r.content[:100]  # Use first 100 chars as key
            if content_key not in content_map:
                content_map[content_key] = HybridResult(
                    content=r.content,
                    score=r.score * terag_weight,
                    source="TERAG",
                    terag_score=r.score,
                    matched_concepts=r.matched_concepts
                )

        # Add/merge Agentic results
        for r in agentic_results:
            content = r.get("content", "")
            content_key = content[:100]

            if content_key in content_map:
                # Already have this content from TERAG - merge scores
                existing = content_map[content_key]
                existing.score += agentic_weight  # Simplified scoring
                existing.agentic_score = 1.0
                existing.source = "Both"
            else:
                # New from Agentic
                content_map[content_key] = HybridResult(
                    content=content,
                    score=agentic_weight,
                    source="Agentic",
                    agentic_score=1.0
                )

        # Sort by combined score
        hybrid_results = sorted(
            content_map.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k]

        return hybrid_results

    def _classify_query(self, query: str) -> str:
        """Simple query classification"""
        query_lower = query.lower()

        entity_keywords = ["who", "which company", "ceo", "person", "organization"]
        semantic_keywords = ["explain", "describe", "what is", "how does"]

        if any(kw in query_lower for kw in entity_keywords):
            return "entity"
        elif any(kw in query_lower for kw in semantic_keywords):
            return "semantic"
        else:
            return "complex"

    def _terag_to_hybrid(self, terag_result: TERAGResult) -> HybridResult:
        """Convert TERAG result to hybrid format"""
        return HybridResult(
            content=terag_result.content,
            score=terag_result.score,
            source="TERAG",
            terag_score=terag_result.score,
            matched_concepts=terag_result.matched_concepts
        )

    def _agentic_to_hybrid(self, agentic_result: Dict) -> HybridResult:
        """Convert Agentic result to hybrid format"""
        return HybridResult(
            content=agentic_result.get("content", ""),
            score=1.0,  # Placeholder
            source="Agentic",
            agentic_score=1.0
        )


# Example usage
if __name__ == "__main__":
    # Create hybrid retriever
    hybrid = HybridRetriever(
        chunks_file="../chunks_full_metadata.json",
        terag_weight=0.5,
        agentic_weight=0.5
    )

    # Test queries
    queries = [
        "What was the company's revenue?",
        "Who are the board members?",
        "Explain the business model"
    ]

    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)

        results = hybrid.retrieve(query, top_k=5, strategy="parallel", verbose=True)

        print(f"\nTop 3 Results:")
        for i, r in enumerate(results[:3], 1):
            print(f"\n{i}. [{r.source}] [Score: {r.score:.4f}]")
            preview = r.content[:120] + "..."
            print(f"   {preview}")
            if r.matched_concepts:
                print(f"   Concepts: {r.matched_concepts[:3]}")
```

## Performance Comparison

### Expected Results

| Metric | TERAG Only | Agentic Only | Hybrid |
|--------|-----------|--------------|--------|
| **Precision** | 75-80% | 75-85% | 80-90% |
| **Recall** | 70-75% | 75-80% | 80-85% |
| **Query Time** | <1s | 1-2s | 1-2s |
| **Token Cost** | Very Low | Medium | Low-Medium |
| **Multi-hop** | Excellent | Good | Excellent |
| **Semantic** | Good | Excellent | Excellent |

## Best Practices

### When to Use TERAG

- ✅ Entity-centric queries ("Who is...", "Which companies...")
- ✅ Multi-hop reasoning (entity relationships)
- ✅ Cost-sensitive applications (low token usage)
- ✅ Large document collections with many entities

### When to Use Agentic RAG

- ✅ Broad semantic queries ("Explain...", "Describe...")
- ✅ Keyword-based searches
- ✅ When query rewriting is needed
- ✅ Dense similarity is most important

### When to Use Hybrid

- ✅ Complex queries requiring both approaches
- ✅ Production systems needing high accuracy
- ✅ When you're unsure which approach is best
- ✅ Applications with diverse query types

## Next Steps

1. **Benchmark**: Compare TERAG, Agentic, and Hybrid on your dataset
2. **Tune**: Adjust weights based on query patterns
3. **Optimize**: Profile and optimize the slower components
4. **Evaluate**: Use metrics like NDCG, MRR for evaluation
5. **Deploy**: Choose strategy based on latency/accuracy tradeoffs
