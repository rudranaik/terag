# TERAG Package Evaluation & Refactoring Plan

**Author:** AI code review  
**Date:** July 4, 2026  
**Version reviewed:** 0.8.0  
**Codebase:** `/Users/rudranaik/Documents/terag/`

---

## Executive Summary

TERAG is a genuinely interesting RAG package — its core insight (token-efficient graph construction using lightweight NER + PPR) is novel and well-supported by the paper. The API surface works in practice, but it has several rough edges that would limit adoption by other developers. This document identifies those issues and proposes a prioritized migration path.

**Target state:** A developer should be able to drop TERAG into an existing RAG pipeline by replacing ~3 lines of code. The API should feel familiar to anyone who has used LangChain retrievers, LlamaIndex, or Haystack.

---

## Priority Legend

| Priority | Label | Meaning |
|----------|-------|---------|
| P0 | **Breaking** | Changes the public API signature; must be done before a non-breaking release |
| P1 | **High** | Affects usability, discoverability, or first-impression quality |
| P2 | **Medium** | Improves code quality, consistency, or documentation |
| P3 | **Low** | Nice-to-have polish or future-proofing |

---

## P0 — Breaking API Changes

### P0.1 — Namespacing: Rename classes from `TERAG*` to `Terag*`

**Problem:** Python convention (PEP 8) says class names use `CapWords` for regular classes and `Capwords_Caps` only when acronyms are the entire name (e.g., `HTTPError`). `TERAG`, `TERAGConfig`, `TERAGGraph`, `TERAGRetriever` look like constants or enums. Mixed-case acronyms like this are jarring in import statements and type hints.

**Fix:**
```python
# Before
from terag import TERAG, TERAGConfig
from terag.graph.builder import TERAGGraph
from terag.retrieval.ppr import TERAGRetriever

# After
from terag import Terag, TeragConfig
from terag.graph.builder import TeragGraph
from terag.retrieval.ppr import TeragRetriever
```

Add backwards-compatible aliases in `__init__.py` for one release:
```python
TERAG = Terag  # Deprecated — use Terag
TERAGConfig = TeragConfig
```

**Impact:** All imports, all internal references, documentation, examples, tests.  
**Effort:** 1-2 hours (grep + replace + verify with test suite).

---

### P0.2 — Split `TERAGConfig` into focused configuration objects

**Problem:** `TERAGConfig` currently holds 16 parameters covering:
- Graph construction (`min_concept_freq`, `max_concept_freq_ratio`, `enable_concept_clustering`)
- PPR algorithm (`ppr_alpha`, `ppr_max_iterations`)
- Weighting (`semantic_weight`, `frequency_weight`)
- Retrieval (`top_k`, `default_retrieval_method`)
- Semantic matching (`use_semantic_entity_matching`, `semantic_match_threshold`)
- LLM configuration (`use_llm_for_ner`, `llm_provider`, `llm_api_key`)
- Persistence (`auto_save_graph`, `graph_save_path`)

This violates the **Single Responsibility Principle**. A developer who wants to change graph building params shouldn't need to know about LLM provider config.

**Fix:** Introduce focused config dataclasses:

```python
@dataclass
class GraphConfig:
    min_concept_freq: int = 2
    max_concept_freq_ratio: float = 0.5
    enable_concept_clustering: bool = False

@dataclass  
class RetrievalConfig:
    top_k: int = 10
    default_method: str = "ppr"
    ppr_alpha: float = 0.15
    ppr_max_iterations: int = 100
    semantic_weight: float = 0.5
    frequency_weight: float = 0.5

@dataclass
class NerConfig:
    use_llm: bool = False
    provider: str = "groq"
    api_key: Optional[str] = None

@dataclass
class SemanticConfig:
    enabled: bool = True
    match_threshold: float = 0.7

# Keep TeragConfig as a facade that accepts *Config kwargs:
@dataclass
class TeragConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ner: NerConfig = field(default_factory=NerConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    auto_save_graph: bool = False
    graph_save_path: Optional[str] = "terag_graph.json"
```

**Impact:** Medium — existing code that passes flat params to `TERAGConfig()` breaks. But this is a clean break worth taking once.  
**Effort:** 2-3 hours.

---

### P0.3 — Standardize the chunk input format

**Problem:** The `from_chunks()` method expects `List[Dict]` with `'content'` and `'metadata'` keys. This is a de facto convention but not enforced or documented as a formal type. Other RAG libraries use Pydantic models or dataclasses for this.

**Fix:** Define a `Document` dataclass that aligns with the emerging standard:

```python
@dataclass
class Document:
    """Standardized document/passage for TERAG."""
    content: str  # The actual text
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None  # Auto-generated if not provided
```

Then add community-standard classmethod names:
```python
@classmethod
def from_documents(cls, documents: List[Document], ...) -> Terag:
    """Standard name used by LangChain, LlamaIndex, etc."""

@classmethod  
def from_texts(cls, texts: List[str], metadatas: Optional[List[Dict]] = None, ...) -> Terag:
    """Convenience — wraps texts into Document objects."""
```

Keep `from_chunks` as a deprecated alias for one release.

**Impact:** Medium — internal code and tests need updating. External users get a smoother migration path.  
**Effort:** 1-2 hours.

---

## P1 — High-Priority Improvements

### P1.1 — Replace `print()` with structured logging throughout

**Problem:** Library code uses `print()` statements everywhere (`builder.py`, `core.py`, `pipeline.py`, `ner_extractor.py`). This is acceptable for CLI tools but inappropriate for a library. It pollutes stdout, can't be suppressed properly, and interferes with Jupyter notebooks and production logging pipelines.

**Severity:** Every `print()` call in library code is a bug waiting to happen.

**Fix:** Audit every file for `print()` and replace with `logger.info()`, `logger.debug()`, or `logger.warning()`. The `verbose` parameter pattern in methods is fine — it can control log level:
```python
if verbose:
    logger.info("Building graph from %d chunks...", len(chunks))
```

**Affected files:** `builder.py`, `core.py`, `pipeline.py`, `ppr.py`, `ner_extractor.py`, `deduplication.py`  
**Effort:** 2-3 hours.

---

### P1.2 — Eliminate lazy imports

**Problem:** Several modules import dependencies inside methods instead of at the top of the file:
- `core.py` line 188: `from terag.embeddings.manager import EmbeddingManager`
- `core.py` line 470: `from terag.retrieval.semantic import SemanticRetriever`
- `core.py` line 539: `from terag.retrieval.hybrid import HybridRetriever`
- `pipeline.py` line 308: `from terag.embeddings.manager import EmbeddingManager`
- `hybrid.py` line 109: `from terag.ingestion.query_ner import ImprovedQueryNER`
- Various other places

**Why this matters:**
1. Breaks static analysis (IDE can't find usages, can't do auto-import)
2. Hides import errors until runtime in specific code paths
3. Makes it harder to reason about dependencies
4. Creates subtle circular import risks
5. "Performance" justification is almost always premature — Python imports are cached after first load

**Fix:** Move all imports to the top of each file. If there's a genuine circular import, refactor the module structure instead of hiding it.

**Effort:** 30 minutes to 1 hour.

---

### P1.3 — Follow standard retriever interface patterns

**Problem:** The `retrieve()` method returns `Tuple[List[RetrievalResult], RetrievalMetrics]`. While this works, it doesn't match what developers expect from RAG packages. The standard pattern in the ecosystem is:
- **LangChain:** `retriever.get_relevant_documents(query) -> List[Document]`
- **LlamaIndex:** `retriever.retrieve(str_or_query_bundle) -> List[NodeWithScore]`
- **Haystack:** `retriever.run(query) -> List[Document]`

All return just the results (metrics are accessed separately).

**Fix:** 
```python
def retrieve(
    self, 
    query: str, 
    top_k: Optional[int] = None,
    method: Optional[str] = None,
    **kwargs
) -> List[RetrievalResult]:
```

Move `RetrievalMetrics` to a separate property or method:
```python
results = terag.retrieve("What is revenue?")
metrics = terag.last_metrics  # or terag.get_metrics()
```

Or return them separately but make it obvious:
```python
terag = Terag.from_documents(docs)
results = terag.retrieve("What is revenue?")
# If they need metrics:
metrics = terag.get_last_retrieval_metrics()
```

**Impact:** Breaking change, but worth it for ecosystem compatibility.  
**Effort:** 1 hour.

---

### P1.4 — Fix the `ImprovedQueryNER` naming and the NER abstraction layer

**Problem:** 
1. "ImprovedQueryNER" vs "QueryNER" — the "Improved" prefix is a code smell that means there was never a pruning step. Rename to `QueryNER`.
2. `NERExtractor.__init__` takes both `groq_api_key` (old) and `api_key` (new) parameters. It also aliases `self.groq_client = self.llm_provider` which is confusing when the provider is OpenAI.
3. `groq_client.py` and `llm_providers.py` coexist — the old Groq-specific client should be removed or folded into the provider abstraction.

**Fix:**
- `ImprovedQueryNER` → `QueryNER` (add alias for backward compat)
- Remove `groq_api_key` parameter from NERExtractor (keep only `api_key`)
- Remove the `self.groq_client` alias — use `self.llm_provider` consistently
- Either delete `groq_client.py` or make it re-export from `llm_providers.py`

**Impact:** Internal consistency. Zero external API change if aliases are kept.  
**Effort:** 1-2 hours.

---

### P1.5 — Make `EmbeddingManager` provider-agnostic

**Problem:** `EmbeddingManager` is hardcoded to use OpenAI (`self.client = OpenAI(api_key=api_key)`) with no fallback or alternative provider support. The `encode()` method is provided as a SentenceTransformers-compatible shim, but the class itself can't work with local models without an OpenAI API key.

**Fix:** 
```python
@dataclass
class EmbeddingConfig:
    provider: str = "openai"  # or "sentence_transformers" or "custom"
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None

class EmbeddingManager:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        config = config or EmbeddingConfig()
        if config.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.model)
        elif config.provider == "openai":
            self._model = OpenAIEmbedder(config.api_key, config.model)
        # ...
    
    def encode(self, texts, **kwargs):
        return self._model.encode(texts, **kwargs)
```

This also solves the problem where `from_chunks()` does `isinstance(self.embedding_model, EmbeddingManager)` — that check exists precisely because the system currently accepts both `EmbeddingManager` and `SentenceTransformer` objects. A proper provider abstraction eliminates the need for such runtime type checks.

**Impact:** Breaking for anyone passing raw `SentenceTransformer` objects, but makes the API much cleaner.  
**Effort:** 2-3 hours.

---

### P1.6 — Clean up the `pyproject.toml` / `requirements.txt` split

**Problem:** There are TWO dependency files with different content:
- `pyproject.toml`: `groq`, `openai` as hard deps; no `sentence-transformers`
- `requirements.txt`: `sentence-transformers`, `matplotlib`, `plotly`, `pyvis`, `requests`; has `groq` and `openai`

This means `pip install terag` installs `groq` and `openai` even if the user only wants regex-based graph building. And `sentence-transformers` is in requirements.txt but not pyproject.toml.

**Fix:** 
- `groq`, `openai` → `[llm]` extra
- `sentence-transformers` → `[semantic]` extra  
- All visualization deps → `[viz]` extra
- Core deps (in `[project.dependencies]`): `numpy`, `networkx`, `scikit-learn`, `python-dotenv`
- Delete `requirements.txt` or make it auto-generated

```toml
[project.optional-dependencies]
llm = ["groq>=0.4.0", "openai>=1.0.0"]
semantic = ["sentence-transformers>=2.2.0"]
viz = ["matplotlib>=3.5.0", "plotly>=5.0.0", "pyvis>=0.3.0"]
dev = ["pytest", "black", "isort"]
all = ["terag[llm,semantic,viz]"]
```

Then: `pip install terag[all]` or `pip install terag[llm,semantic]`.

**Impact:** Low for current users (they already have these installed). Makes pip install much leaner for new users.  
**Effort:** 30 minutes.

---

## P2 — Medium-Priority Improvements

### P2.1 — Add a LangChain-compatible adapter

**Problem:** Developers who want to swap their current RAG package for TERAG need to rewrite their pipeline code. A LangChain `BaseRetriever` subclass would let them use TERAG with zero change to their existing chain code.

**Fix:** Create `terag/integrations/langchain.py`:

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LCDocument

class TeragLangChainRetriever(BaseRetriever):
    """Wraps Terag as a LangChain retriever for drop-in replacement."""
    
    def __init__(self, terag: Terag, **kwargs):
        super().__init__()
        self.terag = terag
        self.top_k = kwargs.get("top_k", 10)
    
    def _get_relevant_documents(self, query: str) -> List[LCDocument]:
        results, _ = self.terag.retrieve(query, top_k=self.top_k)
        return [
            LCDocument(
                page_content=r.content,
                metadata={"score": r.score, "id": r.passage_id, **r.metadata}
            )
            for r in results
        ]
```

LangChain is an optional dependency — use `try/except ImportError` at the import site.

**Effort:** 1-2 hours.

---

### P2.2 — Add proper type stubs and Pydantic models

**Problem:** The code uses `Dict`, `List`, `Optional` from `typing` but doesn't use Pydantic for data validation. `TERAGConfig` is a plain `@dataclass` — no validation on field values. For example, `ppr_alpha=1.5` (invalid — must be 0-1) would silently produce wrong results.

**Fix:**
- Either add Pydantic models for config validation:
  ```python
  from pydantic import BaseModel, Field, field_validator
  
  class TeragConfig(BaseModel):
      ppr_alpha: float = Field(default=0.15, ge=0.0, le=1.0)
  ```
- Or add `__post_init__` validation to existing dataclasses
- Add `py.typed` marker file for PEP 561 compliance

**Effort:** 2-3 hours for Pydantic, 1 hour for basic validation.

---

### P2.3 — Remove module-level `dotenv.load_dotenv()` call

**Problem:** `deduplication.py` line 20 has `dotenv.load_dotenv()` at module level. This means merely importing this module triggers side effects (reading `.env` file, modifying `os.environ`). This is surprising behavior that can interfere with other libraries and the user's environment.

**Fix:** Move `load_dotenv()` calls to the entry points (`main()` functions, CLI handlers, etc.). In library code, just read from `os.getenv()` — it's the user's responsibility to load their env.

**Impact:** None, except imports won't have side effects anymore (which is correct behavior).  
**Effort:** 5 minutes.

---

### P2.4 — Add `RetrievalResult` to public API

**Problem:** `RetrievalResult` is defined in `terag.retrieval.ppr` but is returned by the main `TERAG.retrieve()` method. Users doing type annotations on their own code can't easily import the type:
```python
from terag.retrieval.ppr import RetrievalResult  # Deep import
```

**Fix:** Re-export in `__init__.py` and `retrieval/__init__.py`:
```python
from terag.retrieval.ppr import RetrievalResult, RetrievalMetrics

__all__ = ["Terag", "TeragConfig", "RetrievalResult", "RetrievalMetrics"]
```

**Effort:** 5 minutes.

---

### P2.5 — Fix the duplicate import in `ner_extractor.py`

**Problem:** Line 21 and 22:
```python
from terag.embeddings.cache import ExtractionCache
from terag.embeddings.cache import ExtractionCache
```
Same import, twice.

**Effort:** 30 seconds. But it indicates that linting/formatting wasn't run.

---

### P2.6 — Add CI configuration

**Problem:** No `pytest.ini`, `tox.ini`, `.github/workflows/`, or any CI config. Without this, contributors can't run tests, and the project can't gate PRs on test passing.

**Fix:** Add minimum:
- `.github/workflows/ci.yml` — runs `pytest` on push/PR
- `pytest.ini` or `pyproject.toml` `[tool.pytest.ini_options]`
- Coverage target

**Effort:** 30 minutes for basic CI.

---

### P2.7 — Standardize edge-case handling

**Problem:** Several edge cases are handled inconsistently:
- Empty query → PPR returns `([], metrics_with_zeros)` 
- No matching concepts → same
- Single-passage graph → works, but undefined behavior for PPR's normalization
- What happens when all concepts are filtered by `min_concept_freq`?
- What happens when `top_k > num_passages`? (Seems fine but not explicitly tested)

**Fix:** Document edge-case behavior in docstrings and add explicit guard clauses with meaningful error messages.

---

## P3 — Polish & Future-Proofing

### P3.1 — Emoji-free library code

**Problem:** The pipeline module uses emoji extensively (`🎉`, `🔍`, `🏗️`, `✅`, `❌`, `📊`) in log messages. While charming, this breaks in non-UTF-8 terminals, logging frameworks, and doesn't match the tone of mainstream Python libraries (scikit-learn, NumPy, LangChain).

**Action:** Remove emoji from library log messages. Keep them in docstring examples and CLI output (if any).

**Effort:** 15 minutes.

---

### P3.2 — Add `__main__.py` for CLI entry point

**Problem:** The package has several standalone scripts (`pipeline.py` has `if __name__ == "__main__"` and a `main()` function) but no CLI entry point registered in `pyproject.toml`.

**Fix:**
```python
# terag/__main__.py
from terag.ingestion.pipeline import main
main()
```

Or register a console_scripts entry point in `pyproject.toml`:
```toml
[project.scripts]
terag = "terag.cli:main"
```

**Effort:** 30 minutes.

---

### P3.3 — Write integration tests with HotPotQA data

**Problem:** The `hotpotqa_data/` directory has evaluation scripts and data but no automated tests that verify retrieval accuracy against a known baseline. The paper claims specific EM/F1 scores — these should be automated regression tests.

```python
def test_hotpotqa_baseline():
    """Verify TERAG achieves Paper's claimed accuracy on HotPotQA."""
    terag = Terag.from_graph_file("hotpotqa_data/terag_graph_small.json")
    em, f1 = evaluate_on_hotpotqa(terag, "hotpotqa_data/eval_qa_small.json")
    assert em >= 48.0  # Paper claims 51.2
    assert f1 >= 55.0  # Paper claims 57.8
```

**Effort:** 1-2 hours. Pick just 2-3 benchmark queries for a quick sanity test.

---

### P3.4 — Consolidate documentation into a single entry point

**Problem:** There are **5 markdown files** in the project root:
- `README.md` (629 lines — very comprehensive but overwhelming)
- `QUICKSTART.md` (99 lines — mostly overlaps with README)
- `CONTRIBUTING.md`
- `PUBLISHING_TO_PYPI.md`
- `TERAG_SEMANTIC_ENTITY_MATCHING_PROPOSAL.md`

The semantic entity matching documentation is the most detailed and useful, but it's embedded in `README.md` starting at line 38 and also in its own proposal file.

**Fix:**
1. README.md → concise (200 lines max), focused on 30-second "what is this" + install + minimal example
2. Full docs → Sphinx/GitHub Pages or MkDocs deploy
3. Keep QUICKSTART.md as the "I want to start now" doc

**Effort:** 2-3 hours.

---

### P3.5 — Refactor `GraphBuilder.build_graph_from_chunks()` to reduce method size

**Problem:** `build_graph_from_chunks()` is ~400 lines (lines 467-857). It handles:
- Creating passage nodes
- Concept extraction  
- Frequency tracking
- Edge creation
- Weight calculation
- Clustering
- Deduplication

This is too much for one method. Break into:
- `_create_passage_nodes()`
- `_extract_and_filter_concepts()`
- `_compute_edge_weights()`
- `_cluster_concepts()`

**Effort:** 1-2 hours.

---

### P3.6 — Vectorize the PPR implementation

**Problem:** The PPR implementation uses scalar operations in a Python loop:
```python
for node_idx in range(self.num_nodes):
    for neighbor_idx, weight in self.transitions[node_idx]:
        pi_new[neighbor_idx] += (1 - self.alpha) * weight * pi[node_idx]
```

For graphs with 100k+ nodes, this will be slow. A sparse matrix operation would be much faster:
```python
from scipy.sparse import csr_matrix
pi_new = self.alpha * R + (1 - self.alpha) * self.transition_matrix.T @ pi
```

**Note:** This doesn't need to be P0 because TERAG's paper targets small-medium graphs, but it's worth noting.

**Effort:** 2-3 hours (depends on adding `scipy` dependency).

---

## Implementation Roadmap

### Phase 1 (pre-release: do first, ship next version)
| Item | Effort | Dependencies |
|------|--------|-------------|
| P0.1 — Rename classes | 2h | None |
| P1.6 — Clean up deps | 30m | None |
| P1.4 — Fix NER naming | 1h | None |
| P1.2 — Eliminate lazy imports | 1h | None |
| P1.1 — print() → logging | 3h | None |
| P2.3 — Remove module-level load_dotenv | 5m | None |
| P2.5 — Fix duplicate import | 30s | None |

### Phase 2 (quality: ship second)
| Item | Effort | Dependencies |
|------|--------|-------------|
| P0.2 — Split config | 3h | P0.1 |
| P0.3 — Standardize input format | 2h | P0.1 |
| P1.5 — Provider-agnostic embeddings | 3h | None |
| P1.3 — Standard retriever interface | 1h | P0.1 |
| P2.4 — Public API types | 5m | P0.1 |
| P2.7 — Edge-case handling | 2h | None |

### Phase 3 (ecosystem: ship third)
| Item | Effort | Dependencies |
|------|--------|-------------|
| P2.1 — LangChain adapter | 2h | P0.1, P1.3 |
| P2.2 — Pydantic validation | 3h | P0.2 |
| P2.6 — CI setup | 30m | None |
| P3.3 — Integration tests | 2h | None |

### Phase 4 (polish: ongoing)
| Item | Effort | Dependencies |
|------|--------|-------------|
| P3.1 — Emoji removal | 15m | None |
| P3.4 — Doc consolidation | 3h | None |
| P3.5 — Refactor builder | 2h | None |
| P3.2 — CLI entry point | 30m | None |
| P3.6 — Vectorize PPR | 3h | None |

---

## Summary of Key Principles

1. **Minimize surprise.** A user coming from LangChain/LlamaIndex should feel at home.
2. **Library code should be silent by default.** `print()` is a CLI affordance, not a library affordance.
3. **One way to do it.** Avoid parallel APIs (e.g., `from_chunks` + `from_documents` during transition is OK, but remove the old one eventually).
4. **Fail early and clearly.** Validate config values at construction time, not after hours of graph building.
5. **Optional deps should be optional.** Don't force users to install `groq` and `openai` if they only want regex-based NER.
