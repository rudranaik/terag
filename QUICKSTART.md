# TERAG Quickstart

This guide covers the shortest supported path from installation to retrieval,
then adds provider-backed extraction, persistence, and LangChain integration.

## 1. Install TERAG

The core package supports offline graph construction and PPR retrieval:

```bash
pip install terag
```

For a provider or integration, install the corresponding extra:

```bash
pip install "terag[openai]"
pip install "terag[groq]"
pip install "terag[local]"
pip install "terag[langchain]"
```

## 2. Build an Offline Index

Regex entity extraction is enabled by default. Disabling semantic entity
matching keeps this example independent of embedding providers and API keys.

```python
from terag import EmbeddingConfig, GraphConfig, RetrievalConfig, TERAG, TERAGConfig

config = TERAGConfig(
    graph_config=GraphConfig(
        min_concept_freq=1,
        max_concept_freq_ratio=1.0,
    ),
    retrieval_config=RetrievalConfig(top_k=3),
    embedding_config=EmbeddingConfig(
        use_semantic_entity_matching=False,
    ),
)

rag = TERAG.empty(config=config)
rag.insert(
    [
        "Apple released a new iPhone in September.",
        "Apple reported strong services revenue in Q4.",
        "Microsoft announced improvements to Azure AI.",
    ]
)

results = rag.query("What did Apple report?")

for result in results:
    print(f"[{result.score:.3f}] {result.content}")
```

`insert()` accepts:

- strings;
- dictionaries containing `content` or `text` and optional `metadata`;
- document-like objects with `page_content` and optional `metadata`, including
  LangChain documents.

For already chunked data, `TERAG.from_chunks()` remains supported:

```python
chunks = [
    {
        "content": "Apple reported strong services revenue in Q4.",
        "metadata": {"source": "earnings-report", "page": 4},
    }
]

rag = TERAG.from_chunks(chunks, config=config, verbose=False)
```

## 3. Query and Inspect Results

The preferred application API returns results directly:

```python
results = rag.query("What did Apple report?", top_k=2)

for result in results:
    print(result.id)
    print(result.content)
    print(result.score)
    print(result.matched_concepts)
    print(result.metadata)
```

Use `retrieve()` or `return_metrics=True` when you also need retrieval metrics:

```python
results, metrics = rag.retrieve("What did Apple report?", top_k=2)
results, metrics = rag.query(
    "What did Apple report?",
    top_k=2,
    return_metrics=True,
)
```

Available methods are `ppr`, `semantic`, and `hybrid`:

```python
ppr_results = rag.query("Apple revenue", method="ppr")
```

Semantic and hybrid methods require an embedding model; the offline instance
above intentionally supports only PPR retrieval.

## 4. Use OpenAI Entity Extraction and Embeddings

Install the OpenAI extra and provide the key through the environment:

```bash
pip install "terag[openai]"
export OPENAI_API_KEY="your-key"
```

```python
from terag import EmbeddingConfig, NERConfig, RetrievalConfig, TERAG, TERAGConfig
from terag.embeddings.manager import EmbeddingManager

config = TERAGConfig(
    ner_config=NERConfig(
        use_llm_for_ner=True,
        llm_provider="openai",
        llm_model="gpt-5-nano",
    ),
    retrieval_config=RetrievalConfig(
        top_k=5,
        default_retrieval_method="hybrid",
    ),
    embedding_config=EmbeddingConfig(
        use_semantic_entity_matching=True,
        semantic_match_threshold=0.7,
    ),
)

embeddings = EmbeddingManager(model="text-embedding-3-small")
rag = TERAG.from_chunks(
    chunks,
    config=config,
    embedding_model=embeddings,
    verbose=False,
)

results = rag.query(
    "Which company improved financially?",
    method="hybrid",
)
```

Provider calls may send chunk or query content to that provider. Review its
privacy, retention, and rate-limit policies before using sensitive data.

## 5. Use Groq for Entity Extraction

Groq can perform entity extraction while PPR retrieval remains local:

```bash
pip install "terag[groq]"
export GROQ_API_KEY="your-key"
```

```python
from terag import NERConfig, TERAGConfig

config = TERAGConfig(
    ner_config=NERConfig(
        use_llm_for_ner=True,
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
    )
)
```

Model availability can change; select a currently available model supported by
your provider account.

## 6. Save and Reload the Graph

```python
rag.save_graph("terag_graph.json")

loaded = TERAG.from_graph_file(
    "terag_graph.json",
    config=rag.config,
    embedding_model=rag.embedding_model,
    verbose=False,
)

results = loaded.query("What did Apple report?")
```

Keep the same embedding configuration when reloading a graph for semantic or
hybrid retrieval. Graph files include original passage content and metadata, so
store them appropriately.

## 7. Use TERAG as a LangChain Retriever

```bash
pip install "terag[langchain]"
```

```python
retriever = rag.as_langchain_retriever(method="ppr", top_k=2)
documents = retriever.invoke("What did Apple report?")

for document in documents:
    print(document.page_content)
    print(document.metadata["score"])
```

Original metadata is retained and TERAG adds identifiers, scores, and matched
concepts. See [docs/langchain-integration.md](docs/langchain-integration.md) for
the complete metadata contract.

## 8. Update an Existing Index

```python
rag.insert(["Apple announced another services expansion."])
```

The current implementation rebuilds the in-memory graph from the existing and
new documents. This is correct but can be expensive for large collections; a
true incremental indexing backend is not yet available.

## Troubleshooting

### Provider package is missing

Install the matching extra, such as `terag[openai]` or `terag[groq]`. The base
installation deliberately excludes provider SDKs.

### An API key is not found

Set `OPENAI_API_KEY` or `GROQ_API_KEY`, or supply `llm_api_key` in `NERConfig`.
For an offline workflow, leave `use_llm_for_ner=False`.

### Semantic or hybrid retrieval fails

These modes need a compatible embedding model. Use PPR for the dependency-free
path, or install and configure an embedding provider.

### Results are empty or weak

- Set `min_concept_freq=1` for small collections.
- Confirm that the documents and query contain recognizable concepts.
- Try semantic or hybrid retrieval for paraphrased queries.
- Inspect `matched_concepts` and retrieval metrics before changing thresholds.

## Next Steps

- Read the [technical overview](docs/technical-overview.md).
- Follow the [retrieval flow](docs/retrieval-flow.md).
- Review the [API stability policy](docs/api-stability.md).
- Explore the graph visually with [examples/README.md](examples/README.md).
