# TERAG

[![CI](https://github.com/rudranaik/terag/actions/workflows/ci.yml/badge.svg)](https://github.com/rudranaik/terag/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/terag.svg)](https://pypi.org/project/terag/)
[![Python](https://img.shields.io/pypi/pyversions/terag.svg)](https://pypi.org/project/terag/)
[![License](https://img.shields.io/pypi/l/terag.svg)](LICENSE)

Token-efficient graph retrieval for retrieval-augmented generation (RAG).

TERAG builds a lightweight graph that connects passages through extracted
entities and concepts. At query time, it can use graph-based Personalized
PageRank (PPR), semantic similarity, or a hybrid of both to retrieve relevant
passages without sending the entire knowledge base to an LLM.

## Why TERAG?

- **Graph-aware retrieval:** connect evidence across passages through shared
  people, organizations, products, dates, and other concepts.
- **Offline core path:** build and query a graph with regex-based entity
  extraction and no API key.
- **Flexible retrieval:** choose PPR, semantic, or hybrid retrieval per query.
- **Provider choice:** use OpenAI, Groq, or local SentenceTransformers through
  optional dependencies.
- **Persistent indexes:** save a graph once and reload it for later queries.
- **RAG integration:** expose a LangChain-compatible retriever when needed.

TERAG is a good fit when entity relationships and multi-passage connections
matter. Pure semantic search may be a better fit for collections with few
identifiable concepts or queries based mainly on broad thematic similarity.

## Installation

Install the offline core package:

```bash
pip install terag
```

Install only the optional capabilities you need:

| Extra | Capability |
| --- | --- |
| `terag[openai]` | OpenAI entity extraction and embeddings |
| `terag[groq]` | Groq entity extraction |
| `terag[local]` | Local SentenceTransformers embeddings |
| `terag[langchain]` | LangChain retriever adapter |
| `terag[visual]` | Streamlit visual pipeline explorer and PDF input |
| `terag[bench]` | HotPotQA benchmark provider dependencies |
| `terag[dev]` | Test and development dependencies |
| `terag[all]` | All optional runtime capabilities |

Extras can be combined, for example:

```bash
pip install "terag[langchain,openai]"
```

## Five-Minute Quickstart

This example runs without an LLM, an embedding model, or an API key:

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
        "Apple reported strong revenue growth in Q4 2024.",
        "Microsoft expanded Azure cloud services.",
        "Apple plans to invest the additional revenue in research.",
    ]
)

results = rag.query("What happened to Apple revenue?")

for result in results:
    print(f"{result.score:.3f}  {result.content}")
```

`query()` is the simple application-facing API and returns a list of
`RetrievalResult` objects. The older `retrieve()` API remains supported when
metrics are needed:

```python
results, metrics = rag.retrieve("What happened to Apple revenue?")
```

See [QUICKSTART.md](QUICKSTART.md) for provider setup, document inputs,
persistence, and LangChain usage.

## Retrieval Modes

| Mode | Best suited to | Embeddings required |
| --- | --- | --- |
| `ppr` | Queries containing concrete entities or concepts | No |
| `semantic` | Paraphrases and thematic similarity | Yes |
| `hybrid` | Combining graph evidence with semantic similarity | Yes |

```python
graph_results = rag.query("Apple revenue", method="ppr")
semantic_results = rag.query("income growth", method="semantic")
hybrid_results = rag.query(
    "Which company improved financially?",
    method="hybrid",
)
```

Semantic and hybrid retrieval require a compatible embedding model. Installing
an embedding extra does not automatically enable every provider; configure the
embedding model as shown in the quickstart guide.

## Save and Load

```python
rag.save_graph("terag_graph.json")

loaded = TERAG.from_graph_file(
    "terag_graph.json",
    config=rag.config,
    embedding_model=rag.embedding_model,
    verbose=False,
)
```

Saved graphs contain the indexed passages and graph structure. Protect them as
you would protect the source documents.

## LangChain

```python
retriever = rag.as_langchain_retriever(top_k=2)
documents = retriever.invoke("What happened to Apple revenue?")
```

Install `terag[langchain]` first. See the
[LangChain integration guide](docs/langchain-integration.md) for result
metadata and advanced configuration.

## Visual Pipeline Explorer

The Streamlit explorer can ingest text, Markdown, JSON, and PDF inputs, render
the TERAG graph, and show the PPR query trace:

```bash
pip install "terag[visual]"
python examples/visual_terag_pipeline.py
```

See [examples/README.md](examples/README.md) for offline CLI usage and input
formats.

## Project Status and Limitations

TERAG is a pre-1.0 package with a stable core public API. It is suitable for
evaluation and controlled production use, with these current limitations:

- `insert()` currently rebuilds the in-memory graph; it is not a constant-time
  incremental indexing operation.
- The built-in persistence format is a local JSON graph, not a transactional or
  distributed storage backend.
- Semantic and hybrid retrieval require optional embedding dependencies and may
  incur provider cost or local model overhead.
- Provider-backed entity extraction is subject to the selected provider's
  availability, rate limits, and data-handling terms.
- The package does not yet provide a first-party LlamaIndex adapter or async
  indexing API.

Patch releases preserve the documented top-level imports and construction
paths. See [API stability](docs/api-stability.md) and
[migration notes](docs/migration-0.8.md) before upgrading production systems.

## Benchmarks

The repository includes a reproducible HotPotQA harness and a small smoke
configuration for development regressions:

```bash
make bench-smoke
```

See the [benchmark guide](benchmarks/hotpotqa/README.md) for sampling,
evaluation, comparison, and result inspection. Benchmark results should be
interpreted in the context of the selected entity extractor, embedding model,
and dataset split.

## Documentation

- [Quickstart](QUICKSTART.md)
- [Technical overview](docs/technical-overview.md)
- [Retrieval flow](docs/retrieval-flow.md)
- [API stability](docs/api-stability.md)
- [LangChain integration](docs/langchain-integration.md)
- [HotPotQA benchmark harness](benchmarks/hotpotqa/README.md)
- [Research paper](TERAG.pdf)
- [Adoption-readiness roadmap](TERAG_USABILITY_EVALUATION_AND_PRIORITIZED_CHANGES.md)

## Contributing

Bug reports, focused feature proposals, documentation improvements, and test
cases are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull
request.

## License

TERAG is available under the [MIT License](LICENSE).
