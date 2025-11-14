# TERAG Repository Cleanup Plan

## Current State Analysis

### Core Production Files (KEEP)
These are the essential files for the TERAG system:

**Main Pipeline**:
- `ingest_chunks.py` - Main ingestion entry point
- `json_ner_demo.py` - NER extraction
- `graph_builder.py` - Graph construction
- `hybrid_retriever.py` - Dual-layer retrieval
- `retrieval_demo.py` - Query interface

**Core Components**:
- `embedding_manager.py` - OpenAI embedding management
- `query_processor.py` - Query processing
- `semantic_retriever.py` - Semantic retrieval
- `ppr_retriever.py` - PPR retrieval
- `entity_merger.py` - Graph merging after deduplication
- `query_explainer.py` - Query explanation

**Utilities**:
- `edge_weight_calculator.py` - Edge scoring
- `json_ingestion.py` - JSON parsing
- `extraction_cache.py` - Caching utilities

### Experimental/Duplicate Files (CLEAN UP)

**Deduplication Variants** (need consolidation):
- ❌ `entity_deduplicator.py` - Original (slow for large datasets)
- ✅ `openai_smart_dedup.py` - Optimized version
- ❌ `smart_dedup.py` - Experimental variant
- ❌ `resume_dedup.py` - Utility script
- ❌ `deduplicate_graph_demo.py` - Demo script

**Demo/Utility Scripts** (evaluate for removal):
- ❌ `merge_graphs.py` - Standalone merging (functionality in ingest_chunks.py)
- ❌ `fast_retrieval_demo.py` - Progress-tracked demo (merge into main demo)
- ❌ `build_graph_demo.py` - Graph building demo (redundant)

**Documentation/Analysis**:
- ✅ `TERAG_ARCHITECTURE_AND_DECISIONS.md` - Keep
- ❌ `TERAG_CODE_PATHS.md` - Outdated, remove
- ❌ `README.md` - Update with current info
- ❌ `INTEGRATION_GUIDE.md` - Outdated, remove

---

## Cleanup Actions

### Phase 1: Integrate Optimized Deduplication

**Goal**: Replace slow deduplication with OpenAI-optimized version in main pipeline

**Actions**:
1. **Modify `ingest_chunks.py`**:
   - Replace `entity_deduplicator` import with `openai_smart_dedup`
   - Use `OpenAIEntityDeduplicator` instead of original
   - Update deduplication step to use OpenAI embeddings

2. **Update imports and dependencies**:
   - Ensure consistent embedding manager usage
   - Remove references to old deduplicator

**Files to modify**:
- `ingest_chunks.py` (main changes)
- Update any other files importing old deduplicator

### Phase 2: Remove Redundant Files

**Goal**: Eliminate experimental and duplicate files

**Files to DELETE**:
```
entity_deduplicator.py           # Replaced by openai_smart_dedup
smart_dedup.py                   # Experimental
resume_dedup.py                  # Utility script
deduplicate_graph_demo.py        # Demo script
merge_graphs.py                  # Redundant with ingest_chunks
fast_retrieval_demo.py          # Merge features into main demo
build_graph_demo.py             # Redundant
TERAG_CODE_PATHS.md             # Outdated
INTEGRATION_GUIDE.md            # Outdated
```

**Files to CONSOLIDATE**:
- Merge useful features from `fast_retrieval_demo.py` into `retrieval_demo.py`
- Update `README.md` with current architecture

### Phase 3: Reorganize Directory Structure

**Goal**: Clean directory structure for production

**Proposed Structure**:
```
terag/
├── terag/                      # Main package
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── json_ingestion.py
│   │   ├── ner_extraction.py   # Rename from json_ner_demo.py
│   │   └── pipeline.py         # Rename from ingest_chunks.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py          # From graph_builder.py
│   │   ├── deduplication.py    # From openai_smart_dedup.py
│   │   └── merger.py           # From entity_merger.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid.py           # From hybrid_retriever.py
│   │   ├── semantic.py         # From semantic_retriever.py
│   │   ├── ppr.py              # From ppr_retriever.py
│   │   ├── query_processor.py
│   │   └── explainer.py        # From query_explainer.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── manager.py          # From embedding_manager.py
│   │   └── cache.py            # From extraction_cache.py
│   └── utils/
│       ├── __init__.py
│       └── edge_weights.py     # From edge_weight_calculator.py
├── examples/                   # Demo scripts
│   ├── basic_ingestion.py
│   ├── interactive_retrieval.py  # From retrieval_demo.py
│   └── query_explanation.py
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── setup.py
├── requirements.txt
├── README.md
└── TERAG_ARCHITECTURE_AND_DECISIONS.md
```

### Phase 4: Clean Data and Cache Directories

**Goal**: Organize generated files

**Actions**:
```bash
# Create organized data structure
mkdir -p data/{raw,processed,cache,graphs}

# Move cache files
mv embeddings_cache/ data/cache/embeddings/
mv json_ner_cache/ data/cache/ner/

# Move generated graphs
mv terag_data/ data/graphs/
mv dedup_results/ data/graphs/dedup/
mv graph_results/ data/graphs/raw/

# Clean temporary files
rm -rf ner_results/ graph_results/ merged_graph_results/
```

---

## Immediate Action Items

### Step 1: Backup Current State
```bash
# Create backup of current working state
tar -czf terag_backup_$(date +%Y%m%d).tar.gz *.py *.md data/ embeddings_cache/
```

### Step 2: Integration Fix
**Priority**: Fix `ingest_chunks.py` to use optimized deduplication

**Changes needed**:
```python
# Replace in ingest_chunks.py
from entity_deduplicator import deduplicate_graph_entities  # OLD
from openai_smart_dedup import OpenAIEntityDeduplicator     # NEW

# Update deduplicate_graph function to use OpenAI deduplicator
```

### Step 3: Remove Obvious Junk
**Safe to delete immediately**:
- `smart_dedup.py`
- `resume_dedup.py` 
- `TERAG_CODE_PATHS.md`
- `INTEGRATION_GUIDE.md`
- Any `*.pyc` files or `__pycache__` directories

### Step 4: Feature Consolidation
**Merge progress tracking** from `fast_retrieval_demo.py` into `retrieval_demo.py`:
- Better initialization feedback
- Real-time embedding progress
- System statistics display

---

## Testing Strategy

### Before Cleanup
1. **Test current pipeline**:
   ```bash
   python ingest_chunks.py --chunks test_data.json
   python retrieval_demo.py --graph-file terag_data/terag_graph.json --interactive
   ```

2. **Document current behavior**:
   - Processing times
   - Output file sizes
   - Cache utilization

### After Each Phase
1. **Regression testing**:
   - Same input → same output
   - Performance maintained or improved
   - All features still accessible

2. **Integration testing**:
   - End-to-end pipeline works
   - Embedding cache properly shared
   - Error handling preserved

---

## Risk Mitigation

### Backup Strategy
- **Full backup** before starting cleanup
- **Incremental backups** after each phase
- **Git branches** for experimental changes

### Rollback Plan
- Keep backup of working state
- Document all changes made
- Test rollback procedure before starting

### Validation Checkpoints
- **Functionality**: Core features work
- **Performance**: No significant regression  
- **Usability**: Clear entry points and documentation

---

## Success Criteria

### Technical
- ✅ Single entry point for ingestion (`ingest_chunks.py` or renamed equivalent)
- ✅ Optimized deduplication integrated into main pipeline
- ✅ Consistent OpenAI embedding usage throughout
- ✅ Clean directory structure without duplicates

### Operational  
- ✅ Reduced confusion for new developers
- ✅ Clear distinction between production and example code
- ✅ Maintainable codebase with single source of truth for each feature

### Strategic
- ✅ Ready for packaging and PyPI publication
- ✅ Clear upgrade path for existing users
- ✅ Professional open-source project structure

---

## Timeline

**Week 1**: Integration fixes and immediate cleanup
- Fix `ingest_chunks.py` deduplication
- Remove obvious junk files
- Test core functionality

**Week 2**: Directory restructuring
- Reorganize into package structure
- Update imports and references
- Create examples directory

**Week 3**: Polish and documentation
- Update README with current state
- Add setup.py for packaging
- Final testing and validation

This cleanup plan prioritizes maintaining functionality while eliminating technical debt and preparing for production use.