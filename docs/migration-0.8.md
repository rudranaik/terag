# Migration Notes From 0.8.x Prototype Usage

This guide documents the preferred API shape while preserving compatibility with older TERAG usage.

## Keep Working Code Working

Existing code like this is still supported:

```python
from terag import TERAG, TERAGConfig

rag = TERAG.from_chunks(chunks, config=TERAGConfig(min_concept_freq=1))
results, metrics = rag.retrieve("What happened to Apple revenue?")
```

`retrieve()` continues to return `(results, metrics)`.

## Prefer Query For New Code

For new application code, use `query()` when you only need retrieval results:

```python
results = rag.query("What happened to Apple revenue?")
```

Use `return_metrics=True` when you want the metrics tuple:

```python
results, metrics = rag.query("What happened to Apple revenue?", return_metrics=True)
```

## Prefer Insert For Build-Later Workflows

Older code often built directly from chunk dictionaries:

```python
rag = TERAG.from_chunks(chunks, config=config)
```

That remains supported. For simpler workflows, create an empty index and insert plain documents:

```python
rag = TERAG.empty(config=config)
rag.insert([
    "Apple reported strong revenue growth in Q4 2024.",
    "Microsoft expanded Azure cloud services.",
])
```

`insert()` currently rebuilds the in-memory graph. True incremental updates are tracked separately in the adoption checklist.

## Result Objects

Old field names remain:

```python
result.passage_id
result.content
```

New compatibility aliases are also available:

```python
result.id
result.text
```

Use `to_dict()` for a normalized dictionary and `to_document()` for a lightweight Document-like object.

## Logging

TERAG library calls are quiet by default. The old `verbose` arguments are kept, but they now emit Python log records rather than printing directly to stdout.

```python
import logging

logging.basicConfig(level=logging.INFO)
```
