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

## Supported Features & Limitations

### Supported Features
- **Graph Backend**: Built on **NetworkX** for efficient in-memory graph operations.
- **Ingestion**: Flexible JSON ingestion for chunks, Q&A pairs, and documents.
- **Retrieval Algorithms**:
    - **Personalized PageRank (PPR)**: Biased random walks for entity-centric retrieval.
    - **Hybrid Retrieval**: Combines PPR scores with semantic vector similarity.
- **Named Entity Recognition**:
    - **LLM-based**: Uses Groq (default) or OpenAI for high-accuracy extraction.
    - **Regex Fallback**: Pattern-based extraction when LLM is unavailable.
- **Graph Persistence**: Auto-save and load graphs for reuse.
- **Embeddings**: Integrated with **SentenceTransformers** for local semantic search.

### Limitations
- **Graph Database**: Currently supports **NetworkX** (in-memory) only. Native support for **Neo4j** or **ArangoDB** is **NOT** currently implemented.
- **Scalability**: Best suited for small to medium-sized graphs (up to ~100k nodes) that fit in memory.

## Semantic Entity Matching

### Overview

TERAG uses **hybrid entity matching** that combines three complementary strategies to match query entities to graph concepts:

1. **Exact Match**: Direct text match (e.g., "revenue" → "revenue")
2. **Partial Match**: Substring matching (e.g., "cash" → "cash flow")
3. **Semantic Match**: Embedding-based similarity (e.g., "cashflow" → "cash flow", "AI" → "artificial intelligence")

All three strategies run **in parallel** and their results are combined, ensuring maximum recall while maintaining precision.

### Why Semantic Matching?

Text-based matching alone fails to handle:
- **Spelling variations**: "cash flow" vs "cashflow"
- **Synonyms**: "revenue" vs "income", "CEO" vs "chief executive officer"
- **Abbreviations**: "AI" vs "artificial intelligence", "Q4" vs "fourth quarter"
- **Semantic equivalence**: "revenue growth" vs "income expansion"

Semantic matching uses embeddings to understand the **meaning** of entities, not just their text.

### Configuration

Semantic matching is **enabled by default** but requires an embedding model to function. If no embedding model is provided, TERAG automatically falls back to text-only matching.

#### Basic Setup

```python
from terag import TERAG, TERAGConfig
from terag.embeddings.manager import EmbeddingManager
import os

# Create embedding manager (required for semantic matching)
embedding_manager = EmbeddingManager(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"  # Default, can use other models
)

# Configure TERAG with semantic matching
config = TERAGConfig(
    use_semantic_entity_matching=True,  # Default: True
    semantic_match_threshold=0.7,       # Default: 0.7
    top_k=10
)

# IMPORTANT: Pass embedding_model during graph creation
terag = TERAG.from_chunks(
    chunks, 
    config=config,
    embedding_model=embedding_manager  # Required for semantic matching
)

# Now semantic matching is active
results, metrics = terag.retrieve("What is their cashflow strategy?")
# Will match "cashflow" to "cash flow" concept via semantic similarity
```

#### Setting Embedding Manager at Graph Creation

**Critical**: The embedding model must be provided when creating the graph, not just during retrieval:

```python
# ✅ CORRECT: Embedding model provided at graph creation
terag = TERAG.from_chunks(
    chunks,
    config=config,
    embedding_model=embedding_manager  # Embeddings computed here
)

# ❌ INCORRECT: Cannot add embedding model later
terag = TERAG.from_chunks(chunks, config=config)  # No embeddings!
# terag.embedding_model = embedding_manager  # Too late!
```

This is because TERAG pre-computes concept embeddings during graph construction for efficiency.

#### Loading from Saved Graph

When loading a pre-built graph, you still need to provide the embedding model:

```python
# Load graph from file
terag = TERAG.from_graph_file(
    "terag_graph.json",
    config=config,
    embedding_model=embedding_manager  # Still required!
)
```

### Threshold Configuration

The `semantic_match_threshold` controls how similar an entity and concept must be to match:

| Use Case | Threshold | Behavior |
|----------|-----------|----------|
| **High Precision** (legal, medical) | `0.85` | Only very similar matches |
| **Balanced** (general Q&A) | `0.70` | Good precision/recall balance (default) |
| **High Recall** (exploratory search) | `0.60` | Cast wider net, more matches |

```python
# High precision example
config = TERAGConfig(
    semantic_match_threshold=0.85  # Stricter matching
)

# High recall example
config = TERAGConfig(
    semantic_match_threshold=0.60  # More lenient matching
)
```

### Disabling Semantic Matching

If you prefer text-only matching or don't have an embedding model:

```python
config = TERAGConfig(
    use_semantic_entity_matching=False  # Disable semantic matching
)

terag = TERAG.from_chunks(
    chunks,
    config=config
    # No embedding_model needed
)
```

### Debugging and Logging

To see which matching strategies are being used:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

terag = TERAG.from_chunks(chunks, config=config, embedding_model=embedding_manager)
results, metrics = terag.retrieve("What is their cashflow strategy?")

# Output will show:
# DEBUG: Entity 'cashflow' matched 1 concepts using strategies: semantic
# DEBUG: Entity 'strategy' matched 2 concepts using strategies: exact, partial
```

### Performance Considerations

- **Minimal overhead**: Concept embeddings are pre-computed during graph creation
- **Query-time cost**: Only entity embeddings (typically 2-5 per query) are computed on-the-fly
- **Typical overhead**: < 50ms per query
- **Memory**: No additional memory beyond pre-computed concept embeddings

### Supported Embedding Models

TERAG works with any embedding model that has an `encode()` method:

**OpenAI (via EmbeddingManager)**:
```python
from terag.embeddings.manager import EmbeddingManager

embedding_manager = EmbeddingManager(
    api_key=your_key,
    model="text-embedding-3-small"  # or "text-embedding-3-large"
)
```

**SentenceTransformers (local, no API key needed)**:
```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
terag = TERAG.from_chunks(chunks, config=config, embedding_model=embedding_model)
```

**Other models**: Any model with an `encode(texts: List[str]) -> np.ndarray` method.

### Complete Example

```python
import os
from terag import TERAG, TERAGConfig
from terag.embeddings.manager import EmbeddingManager

# Sample documents
chunks = [
    {"content": "The company's cash flow improved significantly in Q4 2024.", "metadata": {"source": "report"}},
    {"content": "Artificial intelligence investments drove revenue growth.", "metadata": {"source": "report"}},
    {"content": "The CEO announced a new strategic initiative.", "metadata": {"source": "news"}}
]

# Setup embedding manager
embedding_manager = EmbeddingManager(api_key=os.getenv("OPENAI_API_KEY"))

# Configure with semantic matching
config = TERAGConfig(
    use_semantic_entity_matching=True,
    semantic_match_threshold=0.7,
    top_k=5,
    min_concept_freq=1  # Lower threshold for small datasets
)

# Create TERAG with embeddings
terag = TERAG.from_chunks(
    chunks,
    config=config,
    embedding_model=embedding_manager,
    verbose=True
)

# Test semantic matching capabilities
queries = [
    "What is their cashflow strategy?",  # "cashflow" → "cash flow" (spelling)
    "Tell me about AI investments",      # "AI" → "artificial intelligence" (abbreviation)
    "What did the chief executive say?"  # "chief executive" → "CEO" (synonym)
]

for query in queries:
    print(f"\nQuery: {query}")
    results, metrics = terag.retrieve(query, verbose=True)
    print(f"Found {len(results)} results")
    if results:
        print(f"Top result: {results[0].content[:100]}...")
```

### Best Practices

1. **Always provide embedding model at graph creation**, not later
2. **Start with default threshold (0.7)** and adjust based on results
3. **Enable debug logging** during development to understand matching behavior
4. **Use higher thresholds (0.85+)** for domains requiring high precision
5. **Consider local models** (SentenceTransformers) if API costs are a concern
6. **Test with your specific domain** - threshold effectiveness varies by use case
## Architecture

### 1. Graph Construction

TERAG uses a directed, unweighted graph **G = (V, E)** where:
- **Nodes (V)**:
  - **Passage nodes** (squares): Document chunks/passages
  - **Concept nodes** (circles): Named entities and document-level concepts
- **Edges (E ⊆ V×V)**: Directed connections between passages and concepts
- **Storage**: Adjacency lists for efficient neighborhood expansion

### Visual Representation

![TERAG Graph Structure](https://github.com/rudranaik/terag/blob/main/docs/images/graph_structure.png?raw=true)

**Legend:**
- **Passage Nodes (White Squares)**: The actual text chunks from your documents.
- **Entity Nodes (Blue Circles)**: Specific named entities (People, Orgs, Dates).
- **Concept Nodes (Purple Circles)**: Abstract topics or themes shared across passages.
- **Edges**: Bidirectional links. If Passage 1 mentions "Apple", they are connected. This allows the retrieval to "hop" from one passage to another via shared concepts.

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

| Method | Accuracy | Token Consumption | Relative Cost (per token to be ingested) |
|--------|----------|-------------------|---------------|
| **Nano-GraphRAG** | 100% baseline | 100% baseline | 80-200x (personal experience)|
| **LightRAG** | ~75% | ~30% | High |
| **MiniRAG** | ~70% | ~25% | High |
| **TERAG** | 80-90% | 3-11% | **3-5x** |

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



## Installation

### From PyPI
```bash
pip install terag
```

### From Source
```bash
git clone https://github.com/rudranaik/terag.git
cd terag
pip install -e .
```

## Environment Variables

TERAG uses the following environment variables for optional LLM-based features:

| Variable | Purpose | Required For | How to Get |
|----------|---------|--------------|------------|
| `GROQ_API_KEY` | Groq LLM API access | LLM-based NER (default provider) | [Get free API key](https://console.groq.com) |
| `OPENAI_API_KEY` | OpenAI API access | LLM-based NER, embeddings | [Get API key](https://platform.openai.com/api-keys) |

**Setup:**

```bash
# Option 1: Export in your shell
export GROQ_API_KEY="your_groq_key_here"
export OPENAI_API_KEY="your_openai_key_here"

# Option 2: Create a .env file in your project root
echo "GROQ_API_KEY=your_groq_key_here" >> .env
echo "OPENAI_API_KEY=your_openai_key_here" >> .env
```

> [!NOTE]
> **LLM-based NER is optional**. TERAG will automatically fall back to regex-based entity extraction if no API key is provided. LLM-based NER provides higher accuracy but incurs API costs.

> [!TIP]
> **Groq is recommended** for LLM-based NER as it offers fast inference and generous free tier limits.

## Quick Start

Get started with TERAG in 3 simple steps:

### 1. Installation
```bash
pip install terag
```

### 2. Basic Usage with Unified Retrieval API

```python
from terag import TERAG, TERAGConfig
from terag.embeddings.manager import EmbeddingManager
import os

# Define sample data
chunks = [
    {"content": "Apple Inc announced strong revenue growth in Q4 2024.", "metadata": {"source": "news"}},
    {"content": "Microsoft Corporation reported significant cloud achievements.", "metadata": {"source": "news"}}
]

# Setup (optional for semantic/hybrid retrieval)
embedding_manager = EmbeddingManager(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize TERAG
config = TERAGConfig(top_k=3)
terag = TERAG.from_chunks(chunks, config=config, embedding_model=embedding_manager)

# Method 1: PPR Retrieval (default, graph-based)
results, metrics = terag.retrieve("What is the revenue growth?", method="ppr")

# Method 2: Semantic Retrieval (embedding-based)
results, metrics = terag.retrieve("What is the revenue growth?", method="semantic")

# Method 3: Hybrid Retrieval (combines both)
results, metrics = terag.retrieve(
    "What is the revenue growth?", 
    method="hybrid",
    ppr_weight=0.6,  # Weight for graph-based scores
    semantic_weight=0.4  # Weight for semantic scores
)

# Inspect results
for result in results:
    print(f"Score: {result.score:.4f} | Content: {result.content}")
```

**Backward Compatibility:**
```python
# This still works (defaults to PPR)
results, metrics = terag.retrieve("What is the revenue growth?")
```

### 3. Visualization & Export
TERAG uses a custom JSON format for storage, but you can easily export to **GraphML** (supported by Gephi, Cytoscape, etc.) using NetworkX:

```python
import networkx as nx

# Convert to NetworkX graph
G = terag.graph.to_networkx()

# Save as GraphML for visualization tools
nx.write_graphml(G, "terag_graph.graphml")
```

## Retrieval Methods

TERAG supports three retrieval methods, each with different strengths:

| Method | Best For | Requires Embeddings | Speed | Accuracy |
|--------|----------|---------------------|-------|----------|
| **PPR** | Entity-centric queries, multi-hop reasoning | No | Fast | High for entity queries |
| **Semantic** | Conceptual queries, paraphrases | Yes | Medium | High for semantic similarity |
| **Hybrid** | General-purpose, best overall performance | Yes | Medium | Highest (combines both) |

### When to Use Each Method

**PPR (Personalized PageRank)**:
- Queries with specific entities: "What did Apple announce?"
- Multi-hop reasoning: "Which companies mentioned by the CEO have partnerships?"
- When you don't have embedding models available

**Semantic**:
- Conceptual queries: "What are the main challenges discussed?"
- Paraphrased queries: "revenue growth" vs "income increase"
- When entity extraction might miss relevant content

**Hybrid** (Recommended):
- General-purpose retrieval
- When you want the best of both approaches
- Production applications where accuracy matters most

### Example Comparison

```python
query = "What are the financial results?"

# PPR: Finds passages with entities like "revenue", "profit", "Q4"
ppr_results, _ = terag.retrieve(query, method="ppr")

# Semantic: Finds passages semantically similar to "financial results"
sem_results, _ = terag.retrieve(query, method="semantic")

# Hybrid: Combines both approaches for best coverage
hyb_results, _ = terag.retrieve(query, method="hybrid", ppr_weight=0.6, semantic_weight=0.4)
```

## Graph Access Convenience Methods

TERAG provides convenient methods to access graph data:

```python
# Get passage by ID
passage = terag.graph.get_passage("passage_0")
print(passage.content)

# Get passage content directly
content = terag.graph.get_passage_content("passage_0")

# Get concept by ID
concept = terag.graph.get_concept("revenue")
print(f"Concept '{concept.concept_text}' appears in {concept.frequency} passages")

# List all passages and concepts
all_passages = terag.graph.list_passages()
all_concepts = terag.graph.list_concepts()

# Search for concepts by text
revenue_concepts = terag.graph.search_concepts("revenue")
for concept in revenue_concepts:
    print(f"Found: {concept.concept_text}")

# Get neighbors (related nodes)
concept_neighbors = terag.graph.get_concept_neighbors("revenue")
passage_neighbors = terag.graph.get_passage_neighbors("passage_0")
```

## Advanced Usage
For more complex scenarios, including custom graph building and hybrid retrieval, check out `terag/examples/example_usage.py`.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to help improve TERAG.

## Configuration Guide

TERAG is highly configurable to suit different use cases. Here's what each setting does:

| Parameter | Default | Developer Explanation | Product/Business Impact |
|-----------|---------|-----------------------|-------------------------|
| `top_k` | `10` | Number of passages to return in the final result set. | **Response Depth**: Higher values give the LLM more context but increase costs and latency. Lower values are faster and cheaper but might miss details. |
| `min_concept_freq` | `2` | Minimum number of times a concept must appear in the corpus to be included in the graph. | **Noise Reduction**: Filters out one-off mentions or typos. Increase this for cleaner graphs from noisy data (e.g., social media). |
| `max_concept_freq_ratio` | `0.5` | Maximum ratio of documents a concept can appear in before being excluded (stopword filtering). | **Relevance**: Prevents common words (like "company" in a business report) from dominating the search results. |
| `ppr_alpha` | `0.15` | Damping factor for Personalized PageRank (teleport probability). | **Exploration vs. Focus**: Lower values explore further away from the query entities (finding indirect connections). Higher values stick closer to direct matches. |
| `semantic_weight` | `0.5` | Weight given to semantic similarity vs. frequency in the initial node scoring. | **Understanding**: Higher values prioritize concepts that *mean* the same thing as the query, even if spelled differently. |
| `use_llm_for_ner` | `False` | Whether to use an LLM (Groq/OpenAI) for Named Entity Recognition during ingestion/querying. | **Accuracy vs. Cost**: `True` gives much better entity extraction but costs money per query. `False` is free and fast but less accurate. |
| `llm_provider` | `"groq"` | Which LLM provider to use ("groq" or "openai") if `use_llm_for_ner` is True. | **Flexibility**: Switch providers based on credit/availability. |
| `auto_save_graph` | `False` | Whether to automatically save the built graph to disk. | **UX**: Prevents losing work after long build times. |
| `graph_save_path` | `terag_graph.json` | Path to save the graph if auto-save is enabled. | **Organization**: Manage multiple graph versions. |

### Example Configuration

**For a high-precision legal search (using OpenAI):**
```python
config = TERAGConfig(
    top_k=20,
    min_concept_freq=1,
    use_llm_for_ner=True,
    llm_provider="openai",
    auto_save_graph=True
)
```

**For a real-time news chatbot:**
```python
config = TERAGConfig(
    top_k=5,                   # Fast response needed
    min_concept_freq=3,        # Ignore noise
    use_llm_for_ner=False,     # Keep costs low
    ppr_alpha=0.2              # Focus on direct matches
)
```

- `terag/graph/builder.py`: Graph construction from chunks
- `terag/ingestion/ner_extractor.py`: Named Entity Recognition for queries/documents
- `terag/retrieval/ppr.py`: Personalized PageRank retrieval algorithm
- `terag/core.py`: Main TERAG retriever interface
- `terag/examples/example_usage.py`: Usage examples and integration tests
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
- Support 100K+ node graphs efficiently

## Future Enhancements

- [ ] Support for dynamic graph updates
- [ ] Advanced concept clustering (embeddings-based)
- [ ] Multi-hop reasoning chains visualization
- [ ] Hybrid retrieval combining TERAG + dense vectors
- [ ] Real-time graph construction for streaming documents

## License

MIT License
