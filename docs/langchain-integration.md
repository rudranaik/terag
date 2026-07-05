# LangChain Integration

TERAG can be used as a LangChain retriever through the optional `langchain` extra.

## Install

```bash
pip install "terag[langchain]"
```

Provider extras are still separate. For OpenAI-backed NER or embeddings, install both:

```bash
pip install "terag[langchain,openai]"
```

## Basic Usage

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

retriever = rag.as_langchain_retriever(top_k=2)
docs = retriever.invoke("What happened to Apple revenue?")
```

`docs` is a list of LangChain `Document` objects.

## Result Metadata

Each LangChain document includes TERAG metadata:

```python
doc.page_content
doc.metadata["id"]
doc.metadata["passage_id"]
doc.metadata["score"]
doc.metadata["matched_concepts"]
```

Original chunk metadata is preserved alongside these TERAG fields.

## Advanced Import

You can also instantiate the adapter directly:

```python
from terag.integrations.langchain import TERAGRetriever

retriever = TERAGRetriever(terag=rag, method="ppr", top_k=5)
docs = retriever.invoke("What happened to Apple revenue?")
```

LangChain remains optional. Importing TERAG itself does not require LangChain.
