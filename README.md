# TERAG

Token-efficient graph retrieval for RAG systems.

TERAG builds a lightweight graph from your chunks, links passages to extracted entities and concepts, and retrieves relevant passages with graph search, semantic search, or a hybrid of both.

## Install

```bash
pip install terag
```

Optional provider and development extras:

```bash
pip install "terag[openai]"   # OpenAI NER/embeddings
pip install "terag[groq]"     # Groq NER
pip install "terag[local]"    # local SentenceTransformers embeddings
pip install "terag[bench]"    # benchmark tooling
pip install "terag[dev]"      # tests and development tools
```

## Quickstart

```python
from terag import EmbeddingConfig, GraphConfig, RetrievalConfig, TERAG, TERAGConfig

rag = TERAG.empty(
    config=TERAGConfig(
        graph_config=GraphConfig(min_concept_freq=1, max_concept_freq_ratio=1.0),
        retrieval_config=RetrievalConfig(top_k=3),
        embedding_config=EmbeddingConfig(use_semantic_entity_matching=False),
    )
)

rag.insert([
    "Apple reported strong revenue growth in Q4 2024.",
    "Microsoft expanded Azure cloud services.",
])

results = rag.query("What happened to Apple revenue?")

for result in results:
    print(result.content, result.score)
```

`query()` returns a list of result objects. Use `retrieve()` when you need the backward-compatible `(results, metrics)` tuple:

```python
results, metrics = rag.retrieve("What happened to Apple revenue?")
```

## OpenAI Setup

TERAG can use OpenAI for LLM-based entity extraction and embeddings. Install the OpenAI extra and set `OPENAI_API_KEY`.

```bash
pip install "terag[openai]"
export OPENAI_API_KEY="..."
```

```python
from terag import TERAG, TERAGConfig
from terag.embeddings.manager import EmbeddingManager

chunks = [
    {"content": "Apple reported strong revenue growth in Q4 2024.", "metadata": {"source": "report"}},
    {"content": "Microsoft expanded Azure cloud services.", "metadata": {"source": "report"}},
]

config = TERAGConfig(
    top_k=3,
    min_concept_freq=1,
    max_concept_freq_ratio=1.0,
    use_llm_for_ner=True,
    llm_provider="openai",
    llm_model="gpt-5-nano",
    use_semantic_entity_matching=True,
)

embeddings = EmbeddingManager(model="text-embedding-3-small")
rag = TERAG.from_chunks(chunks, config=config, embedding_model=embeddings, verbose=False)

results = rag.query("Which company had revenue growth?", method="hybrid")
```

## Retrieval Modes

| Mode | Use it when | Embeddings required |
| --- | --- | --- |
| `ppr` | Your queries mention entities, names, dates, products, or other concrete concepts. | No |
| `semantic` | Your queries are phrased differently from the source text. | Yes |
| `hybrid` | You want both graph evidence and semantic similarity. | Yes |

```python
graph_results = rag.query("Apple revenue", method="ppr")
semantic_results = rag.query("income growth", method="semantic")
hybrid_results = rag.query("Which company improved financially?", method="hybrid")
```

## Save And Load

```python
rag.save_graph("terag_graph.json")

loaded = TERAG.from_graph_file(
    "terag_graph.json",
    config=rag.config,
    embedding_model=rag.embedding_model,
    verbose=False,
)
```

## Benchmarks

The repository includes a small HotPotQA smoke benchmark for development regressions.

```bash
make bench-smoke
```

The benchmark keeps generated data and caches outside git so repeated runs are much faster after the first run.

## Further Reading

- [Technical overview](/docs/technical-overview.md)
- [How TERAG retrieval works](/docs/retrieval-flow.md)
- [API stability](/docs/api-stability.md)
- [Migration notes from 0.8.x](/docs/migration-0.8.md)
- [HotPotQA benchmark harness](/benchmarks/hotpotqa/README.md)
- [Adoption readiness checklist](/TERAG_USABILITY_EVALUATION_AND_PRIORITIZED_CHANGES.md)

## Project Status

TERAG is moving from prototype toward an adoption-ready SDK. The current focus is a stable Python package, clean install paths, quiet library behavior, simple `insert()` / `query()` usage, reproducible smoke benchmarks, and better documentation.

See the [adoption readiness checklist](/TERAG_USABILITY_EVALUATION_AND_PRIORITIZED_CHANGES.md) for the tracked roadmap.

## License

MIT
