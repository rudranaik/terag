# TERAG: Token-Efficient Graph-Based RAG

**Token-Efficient Graph-Based Retrieval-Augmented Generation**

Based on the research paper: [arXiv:2509.18667](https://arxiv.org/abs/2509.18667) (September 2025)

## Overview

TERAG is a lightweight graph-based RAG framework that achieves **80%+ of GraphRAG's accuracy while consuming only 3-11% of the output tokens**. It addresses the high cost associated with LLM token usage during graph construction that hinders large-scale adoption of graph-based RAG systems.

## Key Advantages

- **Cost-Efficient**: 89-97% reduction in token consumption vs. traditional graph RAG
- **High Performance**: Matches GraphRAG accuracy (EM: 51.2 vs. 51.4; F1: 57.8 vs. 58.6)
- **Lightweight**: Minimal LLM usage - only for query NER and answer generation
- **Scalable**: Efficient for large document collections

## Architecture

### 1. Graph Construction

TERAG uses a directed, unweighted graph **G = (V, E)** where:
- **Nodes (V)**:
  - **Passage nodes** (squares): Document chunks/passages
  - **Concept nodes** (circles): Named entities and document-level concepts
- **Edges (E ⊆ V×V)**: Directed connections between passages and concepts
- **Storage**: Adjacency lists for efficient neighborhood expansion

### 2. Concept Extraction

Lightweight concept extraction focusing on:
- **Named Entities**: People, organizations, locations, dates (dark green circles)
- **Document-level Concepts**: Key topics, themes, technical terms (dark blue circles)
- **Non-LLM Clustering**: Efficient grouping without heavy LLM usage

### 3. Retrieval Algorithm

**Personalized PageRank (PPR)** inspired by HippoRAG:

1. **Query NER**: Few-shot prompt extracts named entities from user query
2. **Node Matching**: Match query entities to graph concepts
3. **PPR Computation**: Run PPR biased towards query-relevant nodes
4. **Weighting**: Combine frequency and semantic weights
5. **Passage Ranking**: Return top-k most relevant passages

### 4. Weighting Scheme

Each matched node receives an unnormalized weight:

```
weight(node) = frequency_weight(node) × semantic_weight(node)
```

- **Frequency Weight**: Inverse of concept frequency (rarer = more important)
- **Semantic Weight**: Embedding similarity between query and concept

## Performance Comparison

| Method | Accuracy | Token Consumption | Relative Cost |
|--------|----------|-------------------|---------------|
| **GraphRAG** | 100% baseline | 100% baseline | 88x multiplier |
| **LightRAG** | ~75% | ~30% | High |
| **MiniRAG** | ~70% | ~25% | High |
| **TERAG** | 80-90% | 3-11% | **3-5x multiplier** |

## Algorithm Components

### Graph Construction Phase

```
1. Chunk documents into passages (P1, P2, ..., Pn)
2. For each passage Pi:
   a. Extract named entities → ENT(Pi)
   b. Extract document concepts → CON(Pi)
   c. Create passage node → V_passage
3. Cluster similar concepts (non-LLM)
4. Create concept nodes → V_concepts
5. Build edges:
   - Pi → concept (if concept in Pi)
   - concept → Pi (bidirectional)
6. Store as adjacency list graph G = (V, E)
```

### Retrieval Phase

```
1. Query Q arrives
2. Extract query entities → ENT(Q) [Few-shot LLM]
3. Match ENT(Q) to graph concepts → matched_nodes
4. Calculate restart vector R:
   R[node] = freq_weight[node] × semantic_weight[node] for matched nodes
   R[node] = 0 for unmatched nodes
5. Run Personalized PageRank:
   PPR(G, R, alpha=0.85, max_iter=100)
6. Rank passages by PPR scores
7. Return top-k passages
8. Generate answer using LLM with retrieved passages
```

## Integration with Agentic RAG

TERAG can augment the existing agentic retrieval system:

```python
from terag import TERAGRetriever
from agentic_retrieval import AgenticRetriever

# Hybrid approach
terag_retriever = TERAGRetriever.from_chunks(chunks)
agentic_retriever = AgenticRetriever.from_chunks(chunks)

# Use TERAG for initial broad retrieval
terag_results = terag_retriever.retrieve(query, top_k=20)

# Use Agentic RAG for refinement and query rewriting
final_results = agentic_retriever.refine(query, terag_results)
```

## Implementation Files

- `graph_builder.py`: Graph construction from chunks
- `ner_extractor.py`: Named Entity Recognition for queries/documents
- `ppr_retriever.py`: Personalized PageRank retrieval algorithm
- `terag_retriever.py`: Main TERAG retriever interface
- `example_usage.py`: Usage examples and integration tests

## Usage Example

```python
from terag import TERAGRetriever

# Build graph from chunks
retriever = TERAGRetriever.from_chunks_file(
    "chunks_full_metadata.json",
    extract_concepts=True,  # Extract named entities + concepts
    min_concept_freq=2      # Filter rare concepts
)

# Retrieve relevant passages
results = retriever.retrieve(
    query="What was the company's revenue in 2024?",
    top_k=10,
    alpha=0.85,  # PPR damping factor
    verbose=True
)

# Results include:
# - passages: List of relevant passages
# - scores: PPR scores for each passage
# - matched_concepts: Query concepts found in graph
# - graph_stats: Graph structure information
```

## Research References

- **TERAG Paper**: [arXiv:2509.18667](https://arxiv.org/abs/2509.18667) (2025)
- **HippoRAG**: Personalized PageRank for RAG (2024)
- **GraphRAG**: Microsoft's graph-based RAG (2023-2024)
- **Personalized PageRank**: Page et al., Stanford (1998)

## Performance Metrics

### Expected Results (based on paper)

- **Accuracy**: 80-90% of GraphRAG quality
- **Token Reduction**: 89-97% fewer tokens than GraphRAG
- **Retrieval Speed**: < 2 seconds per query
- **Graph Construction**: < 15 minutes for 400-page document

### Our Implementation Goals

- Match paper's token efficiency (3-11% consumption)
- Achieve F1 score > 55% on multi-hop queries
- Support 100K+ node graphs efficiently
- Integrate seamlessly with existing agentic RAG

## Future Enhancements

- [ ] Support for dynamic graph updates
- [ ] Advanced concept clustering (embeddings-based)
- [ ] Multi-hop reasoning chains visualization
- [ ] Hybrid retrieval combining TERAG + dense vectors
- [ ] Real-time graph construction for streaming documents

## License

MIT License
