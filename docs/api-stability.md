# API Stability

TERAG is still pre-1.0, but the package now has a small public API surface that should remain stable across patch releases.

## Stable Public Imports

Prefer importing from `terag`:

```python
from terag import (
    EmbeddingConfig,
    GraphConfig,
    NERConfig,
    RetrievalConfig,
    RetrievalMetrics,
    RetrievalResult,
    StorageConfig,
    TERAG,
    TERAGConfig,
    __version__,
)
```

These names are part of the supported public API:

- `TERAG`
- `TERAGConfig`
- `GraphConfig`
- `RetrievalConfig`
- `NERConfig`
- `EmbeddingConfig`
- `StorageConfig`
- `RetrievalResult`
- `RetrievalMetrics`
- `__version__`

README quickstarts should use only these public imports unless a feature specifically requires an advanced provider object such as `EmbeddingManager`.

## Advanced Imports

Modules under these namespaces are available for advanced users, but they are not yet as stable as the top-level API:

- `terag.graph`
- `terag.ingestion`
- `terag.retrieval`
- `terag.embeddings`
- `terag.utils`

They may change during the `0.x` line as the package moves toward a stable SDK.

## Compatibility Policy

Until `1.0.0`:

- Patch releases should not intentionally break top-level public imports.
- Existing construction paths such as `TERAG.from_chunks()` and `TERAG.from_graph_file()` remain supported.
- Existing `retrieve()` tuple behavior remains supported.
- New user-facing examples should prefer `TERAG.empty()`, `insert()`, and `query()`.
- Breaking changes should be documented in a migration guide before release.

After `1.0.0`, TERAG should follow semantic versioning:

- Patch: bug fixes, no public API breaks.
- Minor: backward-compatible features.
- Major: intentional breaking changes with migration notes.

## Deprecation Policy

When a public API is replaced, TERAG should:

1. Keep the old API working for at least one minor release.
2. Document the replacement in the migration guide.
3. Add a deprecation warning only when there is a clear replacement path.
4. Remove deprecated behavior only in a planned breaking release.
