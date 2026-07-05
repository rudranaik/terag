# TERAG Usability, Clarity & Mainstream Adoption Checklist

**Goal**: Upgrade TERAG from a research/prototype package into a mainstream-adoption-ready graph RAG library that is easy to install, easy to swap into existing stacks, and credible next to LightRAG, nanoGraphRAG, LlamaIndex, LangChain retrievers, Haystack, and related libraries.

**Research Paper Reference**: arXiv:2509.18667 (Token-Efficient Graph RAG)

**Evaluation Date**: 2026-07-04

## Executive Assessment

To compete with packages such as LightRAG and nanoGraphRAG, TERAG also needs release engineering, reproducible benchmarks, stronger testing, optional dependency boundaries, storage/index lifecycle management, security posture, async/batch operations, provider abstractions, API stability guarantees, and deployment/integration options.

Current repository observations:

- The package uses a nested install layout (`terag/terag`) plus a stale root-level `__init__.py` that references older modules. This is a real packaging/import risk even though it is not literally triple-nested in the current checkout.
- `pyproject.toml` installs from `where = ["terag"]`, has mandatory OpenAI/Groq/tiktoken dependencies, minimal classifier metadata, no type-check/lint config, and Python `>=3.8`.
- Public imports expose internal modules and mix relative imports with absolute `terag.*` imports.
- The main retrieval API returns `(results, metrics)` tuples and prints progress/status in library code.
- The config dataclass mixes graph build, retrieval, NER, provider, embedding, and persistence settings.
- Tests exist, but root-level test files, generated caches, evaluation data, `dist/`, and package artifacts are mixed into the repository root.
- CI currently appears focused on publishing, not multi-version tests, linting, typing, package validation, security checks, or benchmark regression.

Competitive baseline checked on 2026-07-04:

- LightRAG positions itself as a scalable knowledge-graph RAG framework with SDK/API server install modes, Web UI, Docker deployment, multiple storage types, local/global/hybrid/naive/mix query modes, reranking, document deletion, multimodal parsing, concurrency controls, caching, and security setup guidance: <https://github.com/HKUDS/LightRAG>
- nanoGraphRAG positions itself around a tiny quick-start API, PyPI install, `insert()` / `query()`, incremental insert, async methods, typing, optional providers, and storage portability: <https://github.com/gusye1234/nano-graphrag>

## Adoption-Readiness Definition

TERAG should be considered mainstream-adoption-ready only when these gates are true:

- [ ] A new user can `pip install terag` in a clean environment and run a 5-minute quickstart without editing package internals.
- [ ] Core usage is simple: create an index, insert documents/chunks, query, persist, reload, and update.
- [ ] Public APIs are documented, typed, semver-managed, and backward-compatible across patch/minor releases.
- [ ] Optional providers and integrations do not force heavyweight dependencies on all users.
- [ ] LangChain and LlamaIndex users can swap in TERAG without rewriting their whole pipeline.
- [ ] Benchmarks show where TERAG wins, loses, and trades accuracy for token/cost efficiency.
- [ ] CI validates installability, package contents, tests, linting, typing, docs examples, and build artifacts.
- [ ] Failure modes are clear: missing API keys, incompatible embedding dimensions, stale indexes, memory pressure, bad input schemas, and unsupported storage are handled with actionable errors.
- [ ] The project has contribution, release, security, and support hygiene expected of a serious open-source package.

## Prioritized Checklist

### P0 - Critical: Basic Installability, API Coherence, and Trust

- [ ] **Fix package directory structure and imports.**
  - Current: install source is `terag/terag`; root `__init__.py` appears stale and references older module names.
  - Target: standard `src/terag/` layout, or a clean single `terag/` package at repository root.
  - Remove stale root package code and generated package artifacts from the repository.
  - Ensure internal imports are consistently relative inside the package.
  - Acceptance: `python -m venv /tmp/terag-test && pip install . && python -c "import terag; print(terag.__version__)"` works from outside the repo.

- [x] **Define a small, stable public API.**
  - Export only high-level objects from `terag.__init__`: `TERAG`, config models, result models, provider protocols, and `__version__`.
  - Move lower-level graph/retrieval/ingestion modules behind documented advanced imports or private namespaces.
  - Add `__all__` and an API stability note.
  - Public top-level exports are now limited to `TERAG`, `TERAGConfig`, `RetrievalResult`, `RetrievalMetrics`, and `__version__`.
  - Added focused public config models: `GraphConfig`, `RetrievalConfig`, `NERConfig`, `EmbeddingConfig`, and `StorageConfig`.
  - `__version__` now resolves from package metadata with a local fallback to avoid drift from `pyproject.toml`.
  - Added `docs/api-stability.md` documenting stable public imports, advanced imports, compatibility policy, and deprecation policy.
  - Added tests for public `__all__` and `__version__`.
  - Validation:
    - `.venv/bin/python -m pytest tests` passes.
    - Built wheel imports from a fresh temporary venv outside the repo and exposes `terag.__version__ == "0.8.0"`.
  - Acceptance: README quickstart uses only public imports.

- [x] **Standardize the core user API around insert/query/retrieve.**
  - Add `TERAG.insert()`, `TERAG.add_documents()`, `TERAG.add_chunks()`, and `TERAG.query()`.
  - Keep `retrieve()` for retriever-style use and backward compatibility.
  - Make the default return value a list of result objects, not `(results, metrics)`.
  - Expose metrics through `return_metrics=True`, `last_metrics`, or `get_metrics()`.
  - Implemented `TERAG.empty()` for build-later workflows.
  - Implemented `query()` returning results by default and `(results, metrics)` with `return_metrics=True`.
  - Implemented `insert` / `add_documents()` for strings, dicts with `content` or `text`, and document-like objects with `page_content`.
  - Implemented `add_chunks()` as a rebuild-backed compatibility shim. True incremental graph updates remain tracked in the P1 incremental-update item.
  - Edge cases validated: metrics compatibility, string documents, dict text documents, document-like objects, add-chunks rebuild, unsupported `rebuild=False`, and invalid document shapes.
  - Acceptance: the simplest usage is comparable to:
    ```python
    from terag import TERAG

    rag = TERAG()
    rag.insert(["Apple revenue grew in Q4.", "Microsoft expanded cloud revenue."])
    results = rag.query("What happened to Apple revenue?", top_k=3)
    ```

- [x] **Add backward-compatibility shims and deprecation policy.**
  - Preserve `from_chunks()`, `from_chunks_file()`, and tuple-return behavior behind opt-in flags or deprecation warnings for at least one minor release.
  - Add a migration guide from current `0.8.x` usage to the new API.
  - `from_chunks()`, `from_chunks_file()`, `from_graph_file()`, and `retrieve()` tuple behavior remain supported.
  - Added `docs/migration-0.8.md` showing old `retrieve()` usage, preferred `query()` usage, build-later `insert()` usage, result-object aliases, and logging changes.
  - Added deprecation policy in `docs/api-stability.md`; no warnings are emitted yet because the old APIs remain supported rather than deprecated.
  - Validation:
    - `.venv/bin/python -m pytest tests` passes, including old tuple-return and new query/insert API tests.
  - Acceptance: existing tests for old API pass while new API tests are added.

- [x] **Split configuration into focused validated models.**
  - Introduce `GraphConfig`, `RetrievalConfig`, `NERConfig`, `EmbeddingConfig`, `StorageConfig`, and optional `RuntimeConfig`.
  - Use Pydantic v2 or a lightweight validation layer with clear serialization.
  - Validate ranges, enum values, path settings, embedding dimensions, and provider names.
  - Implemented lightweight dataclass config sections:
    - `GraphConfig`
    - `RetrievalConfig`
    - `NERConfig`
    - `EmbeddingConfig`
    - `StorageConfig`
  - `TERAGConfig` remains backward-compatible with existing flat keyword arguments while also accepting focused config objects or dictionaries.
  - Added `TERAGConfig.to_dict()` for grouped serialization.
  - Validation now rejects invalid concept-frequency settings, invalid frequency ratios, invalid retrieval modes, invalid `top_k`, invalid PPR settings, unsupported LLM providers, empty extraction cache paths, invalid semantic thresholds, and missing save paths when auto-save is enabled.
  - Updated README, technical overview, API stability docs, and migration notes with grouped config examples.
  - Validation:
    - `.venv/bin/python -m pytest tests` passes with config validation and compatibility coverage.
    - `.venv/bin/python -m benchmarks.hotpotqa.scripts.compare --config benchmarks/hotpotqa/configs/smoke.json` passes against the accepted smoke baseline.
    - `PYTHON=.venv/bin/python make bench-inspect-smoke` passes and regenerates the qualitative retrieval report.
    - Built wheel contains `terag/config.py` and no `examples/`, `tests/`, `benchmarks/`, or `hotpotqa_data/` entries.
    - Fresh temporary venv import validates focused config imports and rejects invalid `top_k=0` with a clear `ValueError`.
  - Acceptance: invalid config fails before index construction with actionable errors.

- [x] **Make dependencies optional by capability.**
  - Move OpenAI, Groq, tiktoken, sentence-transformers, visualization, document loaders, LangChain, LlamaIndex, Neo4j, and API server dependencies into extras.
  - Implemented extras: `terag[openai]`, `terag[groq]`, `terag[local]`, `terag[bench]`, `terag[dev]`, `terag[fuzzy]`, and `terag[all]`.
  - Base install now keeps graph/retrieval/core behavior lightweight: `numpy`, `networkx`, and `tqdm`.
  - Replaced `scikit-learn` cosine similarity usage with an internal NumPy helper, so `scikit-learn` is no longer required for semantic/query/deduplication imports.
  - Validation:
    - `pip install .` in a clean venv did not install `openai`, `groq`, `python-dotenv`, `scikit-learn`, `tiktoken`, or `sentence-transformers`.
    - Minimal install supported `import terag`, advanced imports (`terag.graph`, `terag.retrieval.semantic`, `terag.retrieval.query_processor`, `terag.graph.deduplication`), and no-LLM regex retrieval.
    - `pip install ".[openai]"` installed `openai` and `python-dotenv` but not `groq`, `scikit-learn`, `tiktoken`, or `sentence-transformers`; missing API key raises a clear `ValueError`.
    - `pip install ".[groq]"` installed `groq` and `python-dotenv` but not `openai`, `scikit-learn`, `tiktoken`, or `sentence-transformers`.
    - Existing tests pass: `.venv/bin/python -m pytest tests`.
    - HotPotQA smoke comparison passes: `.venv/bin/python -m benchmarks.hotpotqa.scripts.compare --config benchmarks/hotpotqa/configs/smoke.json`.

- [x] **Normalize result objects.**
  - Define a stable `Passage` or `RetrievalResult` model with `id`, `content`, `score`, `metadata`, `matched_concepts`, `source`, and optional `explanation`.
  - Add adapters to/from LangChain `Document` and LlamaIndex node types.
  - Added non-breaking `RetrievalResult.id` and `RetrievalResult.text` aliases while preserving `passage_id` and `content`.
  - Added `RetrievalResult.to_dict()` with normalized keys and `RetrievalResult.to_document()` as a lightweight Document-like adapter.
  - LangChain/LlamaIndex-specific typed adapters remain tracked under their integration checklist items.
  - Validation:
    - Existing tuple-return retrieval behavior remains compatible.
    - Public API exports `RetrievalResult` and `RetrievalMetrics`.
    - Result shape tests pass via `.venv/bin/python -m pytest tests`.

- [x] **Remove print-based library behavior.**
  - Replace `print()` with Python logging and progress callbacks.
  - Default library calls should be quiet unless logging is configured.
  - Converted public TERAG construction/query/save flows, graph building, PPR retrieval, NER extraction progress, and entity deduplication progress from direct `print()` calls to Python logging.
  - Kept `verbose` arguments for compatibility; they now emit log records when application logging is configured instead of writing directly to stdout.
  - Added a regression test asserting `TERAG.from_chunks(..., verbose=False)` plus `query()` produces no stdout/stderr.
  - Remaining `print()` calls are limited to demo/script entry points or standalone CLI-style modules and should be cleaned up during repository cleanup/CLI work.
  - Validation:
    - `.venv/bin/python -m pytest tests` passes.
    - `.venv/bin/python -m benchmarks.hotpotqa.scripts.compare --config benchmarks/hotpotqa/configs/smoke.json` passes against the accepted smoke baseline.
  - Acceptance: quickstart output is controlled by the caller, not the library.

- [x] **Clean repository and distribution contents.**
  - Move root-level test scripts into `tests/` or `benchmarks/`.
  - Move HotpotQA and generated graph files into `benchmarks/data/`, `examples/data/`, or external download scripts.
  - Remove `dist/`, egg-info, caches, local virtualenvs, and generated extraction/embedding caches from version control.
  - Update `.gitignore`, `MANIFEST.in`, and package-data settings.
  - Moved legacy root-level check scripts into `tests/manual/` with non-pytest filenames and a README explaining they are manual checks.
  - Moved sample `chunks.json` into `examples/data/chunks.json` and updated manual checks to reference the new path.
  - Moved the package-adjacent `terag/examples/example_usage.py` into repository-level `examples/example_usage.py` so it is not packaged as a top-level `examples` module.
  - Added `hotpotqa_data/` to `.gitignore` so local/generated HotPotQA source data stays outside git.
  - Updated `MANIFEST.in` and package discovery to keep manual tests, benchmark data, caches, and examples out of the wheel.
  - Validation:
    - `.venv/bin/python -m pytest tests` passes and collects only the active 19-test suite.
    - `.venv/bin/python -m py_compile tests/manual/*.py` passes for moved manual checks.
    - `.venv/bin/python -m pip wheel . -w /tmp/terag-wheel-check-final --no-deps --no-cache-dir` builds successfully.
    - Wheel inspection shows no `examples/`, `tests/`, `benchmarks/`, or `hotpotqa_data/` entries.
    - `.venv/bin/python -m benchmarks.hotpotqa.scripts.compare --config benchmarks/hotpotqa/configs/smoke.json` passes against the accepted smoke baseline.
  - Acceptance: built wheel contains only package code, license/readme metadata, and intentional package assets.

### P1 - High: Swappability, Index Lifecycle, and Developer Experience

- [x] **Add LangChain integration.**
  - Implement `terag.integrations.langchain.TERAGRetriever`.
  - Support modern LangChain retriever methods and `Document` conversion.
  - Keep LangChain dependency optional.
  - Added optional `terag[langchain]` extra using `langchain-core`.
  - Added `terag.integrations.langchain.TERAGRetriever` implementing LangChain's modern `BaseRetriever` interface.
  - Added `TERAG.as_langchain_retriever()` convenience method.
  - Added `results_to_documents()` conversion preserving TERAG result content, score, id, passage id, matched concepts, and original chunk metadata.
  - Added `docs/langchain-integration.md` and linked it from README.
  - Added integration tests for Document conversion and retriever `.invoke()`.
  - Validation:
    - `.venv/bin/python -m pytest tests` passes with LangChain integration tests.
    - Local-only `local_playground/langchain_e2e_demo.py` runs end to end and prints retrieved LangChain Documents.
    - Built wheel contains `terag/integrations/langchain.py` and no `examples/`, `tests/`, `benchmarks/`, `hotpotqa_data/`, or `local_playground/` entries.
    - Fresh base install imports TERAG without LangChain and raises a clear optional-extra error if the adapter is instantiated.
    - Fresh `terag[langchain]` install returns real LangChain `Document` objects from `rag.as_langchain_retriever().invoke(...)`.
    - HotPotQA smoke comparison and qualitative inspection still pass.
  - Acceptance: documented example can replace a vectorstore retriever with TERAG in a LangChain chain.

- [ ] **Add LlamaIndex integration.**
  - Implement `terag.integrations.llama_index.TERAGRetriever`.
  - Support node/result conversion and retriever configuration.
  - Keep LlamaIndex dependency optional.
  - Acceptance: documented example can plug TERAG into a LlamaIndex query engine.

- [ ] **Add storage abstraction and durable index lifecycle.**
  - Define `StorageBackend` protocol for graph, chunks, embeddings, metadata, and metrics.
  - Keep JSON as the simple default, but add at least SQLite/DuckDB or local file-backed storage as a practical durable backend.
  - Track index metadata: TERAG version, embedding model, embedding dimension, config hash, schema version, created/updated timestamps.
  - Acceptance: users can persist, reload, inspect, and validate an index before querying.

- [ ] **Support incremental updates as a first-class feature.**
  - Implement idempotent `insert()` / `add_chunks()` using stable content IDs or caller-provided IDs.
  - Add update and delete operations for documents/chunks.
  - Recompute only affected graph structures when practical; document when full rebuild is required.
  - Acceptance: repeated insertion of identical content does not duplicate graph nodes or embeddings.

- [ ] **Add embedding model lifecycle management.**
  - Define an `EmbeddingProvider` protocol (`embed_documents`, `embed_query`, dimension metadata, batching).
  - Support local SentenceTransformers, OpenAI-compatible APIs, and caller-provided embedding functions.
  - Add explicit re-embedding workflows when the embedding model changes.
  - Acceptance: loading an index with a mismatched embedding model raises a clear error and offers a reindex path.

- [ ] **Add provider abstraction for LLM/NER.**
  - Define `LLMProvider` and `EntityExtractor` protocols.
  - Support regex/no-LLM mode, OpenAI-compatible APIs, Groq, Anthropic/Together-compatible adapters, and local model hooks through extras.
  - Acceptance: users can bring their own extractor without patching internals.

- [ ] **Improve ingestion beyond pre-chunked JSON.**
  - Add document loaders for plain text and Markdown in the base or a small extra.
  - Add optional loaders for PDF, HTML, DOCX, CSV, and directory ingestion.
  - Add a `Chunker` protocol and default token-aware chunker.
  - Acceptance: quickstart can index a text file without hand-building chunk dictionaries.

- [ ] **Improve docs structure.**
  - Keep README short: install, 5-minute quickstart, feature matrix, links.
  - Add docs pages for concepts, configuration, persistence, integrations, provider setup, index lifecycle, evaluation, and troubleshooting.
  - Generate API reference from docstrings.
  - Progress: README has been rewritten into a short install/quickstart/modes/benchmarks page.
  - Progress: moved deeper algorithm and configuration explanation into `docs/technical-overview.md`.
  - Progress: added `docs/retrieval-flow.md` with Mermaid diagrams explaining how PPR, semantic, and hybrid retrieval work after graph construction.
  - Progress: added `docs/api-stability.md` and `docs/migration-0.8.md` and linked them from README.
  - Still pending: configuration, persistence, integrations, provider setup, index lifecycle, evaluation, troubleshooting, and generated API reference docs.
  - Validation:
    - `.venv/bin/python -m pytest tests` passes.
    - `.venv/bin/python -m benchmarks.hotpotqa.scripts.compare --config benchmarks/hotpotqa/configs/smoke.json` passes against the accepted smoke baseline.
  - Acceptance: docs have one canonical path for common tasks and no conflicting examples.

- [ ] **Add notebooks and realistic examples.**
  - Provide examples for local-only usage, OpenAI-compatible usage, LangChain, LlamaIndex, persistence/reload, incremental updates, and evaluation.
  - Ensure examples are tested or at least smoke-tested in CI where practical.
  - Acceptance: examples work from a clean install with documented extras.

- [ ] **Add CLI for common workflows.**
  - Provide commands such as `terag init`, `terag index`, `terag query`, `terag stats`, `terag validate-index`, and `terag reembed`.
  - Acceptance: a user can index and query a local directory without writing Python.

- [ ] **Add structured errors and troubleshooting.**
  - Introduce custom exceptions for config errors, provider errors, embedding errors, storage errors, graph errors, and index compatibility errors.
  - Include remediation in error messages.
  - Acceptance: missing API keys, invalid embedding objects, empty indexes, unsupported query modes, and corrupt graph files have dedicated tests.

### P2 - Medium: Production Quality, Evaluation, Performance, and Release Discipline

- [ ] **Add CI for quality gates.**
  - Run tests across supported Python versions.
  - Add linting, formatting checks, import checks, type checks, package build validation, and minimal install smoke tests.
  - Add separate optional-dependency CI jobs for integrations.
  - Acceptance: PRs cannot pass with broken imports, broken wheel metadata, or failing quickstart tests.

- [ ] **Modernize Python support policy.**
  - Decide whether to support Python 3.9/3.10+ instead of `>=3.8`.
  - Match dependency realities and competitor/user expectations.
  - Acceptance: support matrix is documented and enforced in CI.

- [ ] **Add typing discipline.**
  - Add `py.typed`.
  - Add mypy or pyright configuration.
  - Type public APIs strictly; type internals pragmatically.
  - Acceptance: downstream users get useful type hints from the installed package.

- [ ] **Add unit, integration, and contract tests.**
  - Unit-test graph construction, entity matching, retrieval methods, config validation, storage, errors, and adapters.
  - Add integration tests with fake providers to avoid paid API calls.
  - Add contract tests for embedding and LLM provider protocols.
  - Acceptance: tests cover both no-key local mode and provider-backed mode through mocks.

- [ ] **Add reproducible benchmark suite.**
  - Benchmark TERAG against vector-only retrieval, LightRAG/nanoGraphRAG where feasible, and internal ablations.
  - Track retrieval quality, latency, memory, token usage, index build cost, query cost, and update cost.
  - Include HotpotQA or other public datasets with download/build scripts rather than committed generated artifacts.
  - Set up HotPotQA smoke/dev-5pct/full tiers with deterministic sampling manifests, cached/generated artifacts outside git, baseline metrics, and regression comparison.
  - Added qualitative smoke inspection: `make bench-inspect-smoke` writes `latest_smoke_inspection.md/json` showing each real question, expected supporting chunks, retrieved chunks, ranks, scores, matched concepts, and snippets.
  - Validation:
    - `PYTHON=.venv/bin/python make bench-inspect-smoke` passes and writes `benchmarks/hotpotqa/results/latest_smoke_inspection.md`.
  - Acceptance: benchmark results can be reproduced by a new contributor.

- [ ] **Add evaluation hooks.**
  - Support context precision/recall, MRR/nDCG, faithfulness hooks, answer generation evaluation, and custom evaluator callbacks.
  - Provide RAGAS or equivalent integration as optional extra.
  - Acceptance: users can evaluate retriever quality without writing bespoke scripts.

- [ ] **Add reranking support.**
  - Add optional reranker protocol for cross-encoders, Cohere/Jina/Voyage-style rerankers, and local rerankers.
  - Document latency/quality tradeoffs.
  - Acceptance: reranking can be enabled at query time without rebuilding the index.

- [ ] **Improve retrieval modes and naming.**
  - Clarify and document TERAG modes: graph/PPR, semantic, hybrid, and any future local/global/mix-style modes.
  - Add query parameter objects for repeatable configuration.
  - Acceptance: mode names are stable, documented, and consistently implemented.

- [ ] **Improve scalability and performance engineering.**
  - Add batching for embeddings and extraction.
  - Cache extraction and embedding operations with invalidation keyed by content/config/provider.
  - Profile PPR on large graphs and consider sparse matrix implementations.
  - Add memory estimates and large-index warnings.
  - Acceptance: performance limits are documented with measured data, not guesses.

- [ ] **Add async and batch APIs.**
  - Add `aquery`, `ainsert`, and batch query/insert methods where providers/storage support it.
  - Avoid forcing async on simple synchronous users.
  - Acceptance: async methods are tested with fake providers.

- [ ] **Add graph/database backend roadmap with one practical backend.**
  - Keep NetworkX as default.
  - Add one production-useful backend first, likely SQLite/DuckDB/Kuzu for local durability or Neo4j for graph-native deployments.
  - Define backend capability flags.
  - Acceptance: storage/backend interfaces do not leak backend-specific objects into the core API.

- [ ] **Add API server only after SDK stabilizes.**
  - Optional FastAPI server for index, query, stats, reload, and health endpoints.
  - Include authentication guidance if any server binds beyond localhost.
  - Acceptance: server is an optional extra and does not complicate the base SDK.

- [ ] **Add security and privacy hygiene.**
  - Never log secrets or raw provider credentials.
  - Document local-only mode and provider data-flow implications.
  - Add `SECURITY.md`.
  - Add dependency vulnerability checks in CI.
  - Acceptance: users can understand what content leaves their machine for each provider mode.

- [ ] **Improve release process.**
  - Use PyPI trusted publishing instead of raw token upload where possible.
  - Add changelog, versioning policy, release checklist, and GitHub release notes template.
  - Validate wheels and sdists before publish.
  - Acceptance: releases are reproducible and have documented migration notes.

- [ ] **Add project governance hygiene.**
  - Keep `CONTRIBUTING.md` current.
  - Add issue templates, PR template, support policy, code of conduct if desired, and roadmap.
  - Acceptance: external contributors can submit useful PRs without private context.

### P3 - Future Differentiators

- [ ] **Add visual graph inspection helpers.**
  - Export graph summaries, concept neighborhoods, and retrieval traces.
  - Keep visualization dependencies optional.
  - Progress: added a documentation-only retrieval visualization in `docs/retrieval-flow.md`.
  - Still pending: actual package APIs for graph summaries, neighborhoods, trace data, and optional rendered visualizations.

- [ ] **Add retrieval explainability.**
  - Show matched query entities, matched concepts, graph paths/neighborhoods, PPR contribution, semantic contribution, and reranker contribution.
  - Useful for enterprise debugging and evaluation.
  - Progress: documented the target trace shape in `docs/retrieval-flow.md`.
  - Still pending: structured trace objects returned by the package.

- [ ] **Add multimodal/document-structure roadmap.**
  - Consider table-aware and PDF-aware parsing after text/Markdown/PDF ingestion is stable.
  - Do not let multimodal scope delay the core SDK.

- [ ] **Add Docker/API deployment package.**
  - Optional only after base package, storage, auth, and API stability are mature.

- [ ] **Add hosted-demo or docs demo.**
  - A small public example can help adoption, but should not become a maintenance burden before package fundamentals are fixed.

## Recommended Target Public API

```python
from terag import TERAG, GraphConfig, RetrievalConfig

rag = TERAG(
    graph_config=GraphConfig(min_concept_freq=2),
    retrieval_config=RetrievalConfig(top_k=5, mode="hybrid"),
)

rag.insert([
    "Apple reported strong revenue growth in Q4.",
    "Microsoft expanded its cloud business.",
])

results = rag.query("What happened to Apple revenue?")
for result in results:
    print(result.content, result.score)
```

## Recommended Integration API

```python
from terag import TERAG

rag = TERAG.from_index("./terag-index")
retriever = rag.as_langchain_retriever(top_k=5)
docs = retriever.invoke("What happened to Apple revenue?")
```

```python
from terag import TERAG

rag = TERAG.from_index("./terag-index")
retriever = rag.as_llama_index_retriever(top_k=5)
nodes = retriever.retrieve("What happened to Apple revenue?")
```

## Suggested Implementation Sequence

1. Finish P0 package structure, public API, dependency extras, result model, no-print logging, repository cleanup, and backward-compatible API shims.
2. Add P1 LangChain/LlamaIndex adapters, storage/index metadata, incremental insert/update/delete, embedding lifecycle validation, provider protocols, and focused docs.
3. Add P2 CI gates, typing, test matrix, reproducible benchmarks, evaluation hooks, reranking, async/batch operations, security/release hygiene, and one practical durable backend.
4. Defer P3 server/web UI, multimodal parsing, advanced graph databases, and hosted demo until the SDK is stable.

## Versioning Recommendation

- `0.9.0`: Packaging cleanup, public API, compatibility shims, docs quickstart, CI basics.
- `0.10.0`: Integrations, storage abstraction, incremental update/delete, provider protocols.
- `0.11.0`: Benchmarks, evaluation hooks, reranking, stronger typing/tests.
- `1.0.0`: Stable API contract, migration guide, security/release process, adoption-ready docs, and reproducible benchmark claims.

## Completion Tracking

Use the checkboxes above as the source of truth. When an item is implemented, update the checkbox and add a short note with the validation commands and any follow-up constraints.
