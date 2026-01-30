# TERAG Library Feedback & Shortcomings

Based on our recent evaluation and implementation of the retrieval pipeline, here are the primary shortcomings and areas for improvement in the `terag` library.

## 1. Unified Retrieval API (Top Priority)
Currently, to use PPR, Semantic, or Hybrid retrieval, a user must interact with different classes (`TERAG`, `SemanticRetriever`, `HybridRetriever`) and call different methods with varying parameter signatures.
- **Problem**: This creates a steep learning curve and makes it difficult to experiment with different strategies within the same application.
- **Recommendation**: Implement a single, high-level `retrieve` function (likely on the `TERAG` class) that accepts a `method` parameter. 
  - *Example*: `terag.retrieve(query, method="hybrid", top_k=10, weights=(0.6, 0.4))`
  - *Internal logic*: Should handle the orchestration and parameter routing for all three methods under a consistent signature.

## 2. Hybrid Retrieval Inconsistency
The `HybridRetriever` currently uses a simplified **Embedding-based QueryProcessor** by default, whereas the standard `TERAG.retrieve` (PPR) uses a more sophisticated **LLM-based ImprovedQueryNER**.
- **Impact**: There is a significant quality gap between standard PPR and the PPR component within Hybrid retrieval.
- **Recommendation**: `HybridRetriever` should support the same `ImprovedQueryNER` or at least allow injecting pre-extracted entities to ensure consistency.

## 2. Configuration & Parameter Fragmentation
Configuring the library across different retrievers is inconsistent:
- **Naming**: `min_similarity_threshold` (Semantic) vs `semantic_match_threshold` (PPR) vs `min_semantic_score` (Hybrid).
- **Initialization**: Some classes require an `EmbeddingManager`, while others (like `TERAG`) auto-initialize it, sometimes making the flow feel "black-box."
- **Threshold Defaults**: Default thresholds vary widely (0.3 vs 0.5 vs 0.7), leading to unexpected "zero-result" scenarios when switching methods.

## 3. Transparency & Observability
It is difficult to see what is happening "under the hood" without manual monkey-patching or deep debugging.
- **Entity Visibility**: Retrievers don't easily return the entities they successfully extracted or matched to the graph.
- **Latency Attribution**: It's hard to distinguish between LLM latency, embedding latency, and graph computation time.
- **Recommendation**: Provide a `RetrievalMetrics` or `RetrievalAnalysis` object that is consistent across all three methods.

## 4. Graph Access Lack of Convenience
The `TERAGGraph` object is powerful but lacks simple convenience methods for common read operations.
- **Example**: To get content, one must use `terag.graph.passages[passage_id].content` instead of a cleaner `terag.graph.get_passage(id)`.
- **Match Accuracy**: Small variations in entity spelling (e.g., "Krishna" vs "Krishna-Dwaipayana Vyasa") can result in 0 PPR results if the graph is sparse, even if the node exists.

## 5. Resilience to Graph Sparsity
The PageRank-based retrieval is highly sensitive to the graph structure.
- **Issue**: If a query matches an entity that has few or no edges, the result count is 0. There is no middle-ground "fuzzy" graph matching available without falling back entirely to the Semantic layer.
- **Recommendation**: Implement a "semantic expansion" for entities where the retriever can jump to nearby concepts based on embedding similarity if the exact match is isolated.

## 6. Error Handling & Fragility
- **LLM Detection**: There are intermittent reports of "LLM not found" even when API keys are provided, suggesting the initialization logic for `llm_providers` might be fragile or over-reliant on specific environment variable names.
- **Batch Processing**: While `QueryProcessor` supports batching, it's not consistently exposed across the higher-level `TERAG` or `HybridRetriever` APIs.
