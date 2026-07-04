# TERAG Technical Overview

TERAG is a graph-based retriever designed to keep graph RAG cheaper and simpler than approaches that ask an LLM to summarize or reason over the whole corpus during indexing.

The package is currently focused on retrieval. It builds a graph from chunks, retrieves relevant passages, and leaves answer generation to the application layer.

## Core Idea

Most RAG systems retrieve by comparing the query embedding against chunk embeddings. TERAG adds a graph:

- Passage nodes represent chunks.
- Concept nodes represent extracted named entities and document-level concepts.
- Edges connect a passage to the concepts found inside that passage.

At query time, TERAG extracts important query terms, matches them to concept nodes, runs retrieval, and returns passage results.

## Graph Construction

Input chunks look like this:

```python
chunks = [
    {
        "content": "Apple reported strong revenue growth in Q4 2024.",
        "metadata": {"source": "report"}
    }
]
```

TERAG then:

1. Creates one passage node per chunk.
2. Extracts named entities and concepts from each chunk.
3. Normalizes concept text into stable concept IDs.
4. Filters concepts using frequency thresholds.
5. Connects each passage node to its concept nodes.

The default graph backend is in-memory NetworkX-style graph data. Durable storage and production backends remain roadmap items.

## Entity And Concept Extraction

TERAG supports two extraction styles:

- Regex fallback: free, local, fast, and good enough for smoke tests or simple corpora.
- LLM-backed extraction: higher quality extraction through configured providers such as OpenAI or Groq.

Provider dependencies are optional extras. A base install should not force OpenAI, Groq, SentenceTransformers, or benchmark dependencies onto users who do not need them.

## Semantic Entity Matching

When an embedding model is available, TERAG can match query terms to graph concepts in three ways:

- Exact match: `revenue` matches `revenue`.
- Partial match: `cash` can match `cash flow`.
- Semantic match: `cashflow` can match `cash flow`, and `AI` can match `artificial intelligence`.

Semantic matching is controlled by:

```python
TERAGConfig(
    use_semantic_entity_matching=True,
    semantic_match_threshold=0.7,
)
```

Higher thresholds favor precision. Lower thresholds favor recall.

## Retrieval Modes

### PPR

Personalized PageRank starts from matched concept nodes and spreads relevance through the graph. Passages connected to strong matching concepts score higher, and passages connected through shared concepts can also surface.

This works well for entity-heavy and multi-hop-style questions.

### Semantic

Semantic retrieval compares the query embedding with passage embeddings. It is useful when the query and source text use different wording.

This mode requires an embedding model.

### Hybrid

Hybrid retrieval combines graph/PPR evidence with semantic similarity. It is the most general-purpose mode when embeddings are available.

## Result Shape

TERAG result objects expose both TERAG-native and common retriever-style fields:

```python
result.id
result.passage_id
result.content
result.text
result.score
result.metadata
result.matched_concepts
```

Use `result.to_dict()` for a normalized dictionary and `result.to_document()` for a lightweight Document-like shape.

## Persistence

Graphs can be saved and loaded:

```python
rag.save_graph("terag_graph.json")
rag = TERAG.from_graph_file("terag_graph.json", config=config, embedding_model=embeddings)
```

Current persistence is JSON-based. The adoption roadmap tracks index metadata, schema versions, embedding-model compatibility checks, durable storage, update/delete operations, and backend abstractions.

## Logging

Library calls are quiet by default. TERAG uses Python logging for progress and diagnostics:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

The `verbose` arguments are kept for compatibility, but they now emit log records instead of printing directly to stdout.

## Benchmarks

The repository includes a HotPotQA smoke benchmark for development checks:

```bash
make bench-smoke
```

The smoke tier is intentionally small. Its purpose is not to prove benchmark leadership; it catches obvious regressions while the package is being made easier to install, use, and integrate.

See `benchmarks/hotpotqa/README.md` for benchmark details.
