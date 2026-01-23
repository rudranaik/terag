# TERAG Enhancement Proposal: Semantic Entity Matching for PPR Retrieval

**Date**: 2026-01-23  
**Author**: Rudra Naik  
**Package**: terag v0.6.0  
**Repository**: https://github.com/rudranaik/terag

---

## Executive Summary

This document proposes adding **semantic similarity-based entity matching** to TERAG's Personalized PageRank (PPR) retrieval system. Currently, entity-to-concept matching relies solely on fuzzy text matching, which fails to handle spelling variations, synonyms, and semantically equivalent terms.

**Key Benefits**:
- Better recall for semantically similar queries (e.g., "cash flow" vs "cashflow")
- Robust handling of synonyms (e.g., "revenue" vs "income")
- Consistent with existing hybrid retrieval approach
- Minimal performance impact (embeddings already computed)

---

## Problem Statement

### Current Behavior

The PPR retrieval's entity matching (`_match_entities_to_concepts` in `terag/retrieval/ppr.py`) uses only text-based matching:

1. **Exact match**: Normalized entity text matches concept ID exactly
2. **Partial match**: Entity is substring of concept OR concept is substring of entity

### Real-World Issue

**Example from user testing**:
- Query: "What is their cash flow strategy?"
  - Entity extracted: "cashflow" (single word)
  - Graph concepts: "cash flow" (two words)
  - **Result**: No match found, retrieval fails

**Other problematic cases**:
- "AI" vs "artificial intelligence"
- "Q4" vs "fourth quarter"
- "revenue growth" vs "income expansion"
- "CEO" vs "chief executive officer"

### Current Implementation

```python
def _match_entities_to_concepts(self, query_entities: List[str]) -> Set[str]:
    """
    Match query entities to graph concept IDs
    Uses fuzzy matching to handle variations
    """
    matched = set()

    for entity in query_entities:
        # Normalize entity
        entity_normalized = " ".join(entity.lower().strip().split())

        # Exact match
        if entity_normalized in self.graph.concepts:
            matched.add(entity_normalized)
            continue

        # Partial match (entity contained in concept or vice versa)
        for concept_id, concept in self.graph.concepts.items():
            # Check if entity is substring of concept or vice versa
            if entity_normalized in concept_id or concept_id in entity_normalized:
                matched.add(concept_id)

    return matched
```

**Limitation**: Purely lexical matching - no semantic understanding.

---

## Current Architecture Analysis

### How PPR Retrieval Works

1. **Extract entities** from query using NER
2. **Match entities to concepts** ← **THIS IS WHERE WE NEED IMPROVEMENT**
3. **Build restart vector** with frequency + semantic weights
4. **Run PageRank** to propagate scores
5. **Return ranked passages**

### Semantic Similarity Already Used

Interestingly, TERAG already uses semantic similarity in the **restart vector construction** (`_build_restart_vector`):

```python
# Semantic weight calculation (already implemented)
if query_embedding is not None and concept_id in self.concept_embeddings:
    concept_emb = self.concept_embeddings[concept_id]
    similarity = np.dot(query_embedding, concept_emb)
    sem_score = max(0, similarity)  # Cosine similarity

# Combined weight
weight = frequency_weight * freq_score + semantic_weight * sem_score
```

**Key insight**: We already compute concept embeddings and use semantic similarity for weighting. We should also use it for **matching**.

---

## Proposed Solution

### Enhanced Entity Matching Algorithm

Add semantic similarity as a **third matching strategy** alongside exact and partial text matching:

```python
def _match_entities_to_concepts(
    self, 
    query_entities: List[str],
    semantic_threshold: float = 0.7
) -> Set[str]:
    """
    Match query entities to graph concept IDs
    
    Uses three strategies:
    1. Exact text match
    2. Partial text match (substring)
    3. Semantic similarity match (NEW)
    """
    matched = set()
    
    for entity in query_entities:
        # Normalize entity
        entity_normalized = " ".join(entity.lower().strip().split())
        
        # Strategy 1: Exact match
        if entity_normalized in self.graph.concepts:
            matched.add(entity_normalized)
            continue
        
        # Strategy 2: Partial text match
        text_matched = False
        for concept_id in self.graph.concepts:
            if entity_normalized in concept_id or concept_id in entity_normalized:
                matched.add(concept_id)
                text_matched = True
        
        # Strategy 3: Semantic match (if no text match and embeddings available)
        if not text_matched and self.embedding_model and self.concept_embeddings:
            entity_emb = self.embedding_model.encode([entity])[0]
            
            for concept_id, concept_emb in self.concept_embeddings.items():
                # Cosine similarity
                similarity = np.dot(entity_emb, concept_emb)
                
                if similarity >= semantic_threshold:
                    matched.add(concept_id)
    
    return matched
```

### Configuration Options

Add to `TERAGConfig`:

```python
@dataclass
class TERAGConfig:
    # ... existing fields ...
    
    # Semantic entity matching
    use_semantic_entity_matching: bool = True
    semantic_match_threshold: float = 0.7  # Cosine similarity threshold
```

### Threshold Selection

Recommended thresholds based on use case:

| Use Case | Threshold | Rationale |
|----------|-----------|-----------|
| **High precision** (legal, medical) | 0.85 | Only very similar concepts |
| **Balanced** (general Q&A) | 0.70 | Good balance |
| **High recall** (exploratory search) | 0.60 | Cast wider net |

---

## Implementation Plan

### Phase 1: Core Implementation

**File**: `terag/retrieval/ppr.py`

1. Add `semantic_threshold` parameter to `_match_entities_to_concepts`
2. Implement semantic matching logic
3. Add fallback behavior when embeddings unavailable

**Estimated effort**: 2-3 hours

### Phase 2: Configuration

**File**: `terag/core.py`

1. Add config fields to `TERAGConfig`
2. Pass config to `TERAGRetriever`
3. Update `retrieve()` method signature

**Estimated effort**: 1 hour

### Phase 3: Testing

**File**: `tests/test_semantic_entity_matching.py` (new)

Test cases:
- Spelling variations ("cash flow" vs "cashflow")
- Synonyms ("revenue" vs "income")
- Abbreviations ("AI" vs "artificial intelligence")
- Threshold sensitivity
- Fallback when no embeddings

**Estimated effort**: 2-3 hours

### Phase 4: Documentation

**Files**: `README.md`, docstrings

1. Update README with new feature
2. Add usage examples
3. Document threshold recommendations

**Estimated effort**: 1 hour

---

## Example Usage

### Before (Current)

```python
from terag import TERAG, TERAGConfig

config = TERAGConfig(top_k=10)
terag = TERAG.from_chunks(chunks, config=config)

# Query: "What is their cashflow strategy?"
# Entity extracted: "cashflow"
# Graph has: "cash flow"
# Result: NO MATCH ❌
results, metrics = terag.retrieve("What is their cashflow strategy?")
# Returns: 0 results
```

### After (Proposed)

```python
from terag import TERAG, TERAGConfig
from terag.embeddings.manager import EmbeddingManager

# Create embedding manager
embedding_manager = EmbeddingManager(api_key=openai_key)

# Enable semantic entity matching
config = TERAGConfig(
    top_k=10,
    use_semantic_entity_matching=True,
    semantic_match_threshold=0.7
)

terag = TERAG.from_chunks(
    chunks, 
    config=config,
    embedding_model=embedding_manager
)

# Query: "What is their cashflow strategy?"
# Entity extracted: "cashflow"
# Graph has: "cash flow"
# Semantic similarity: 0.95 (very high!)
# Result: MATCH FOUND ✅
results, metrics = terag.retrieve("What is their cashflow strategy?")
# Returns: relevant passages about cash flow
```

---

## Performance Considerations

### Computational Cost

**Minimal impact** because:
1. Concept embeddings already pre-computed in `_precompute_concept_embeddings()`
2. Entity embeddings: Only N entities per query (typically 2-5)
3. Similarity computation: Fast dot product (O(d) where d = embedding dimension)

**Estimated overhead**: < 50ms per query for typical use cases

### Memory

No additional memory required - concept embeddings already stored.

---

## Backward Compatibility

### Default Behavior

Set `use_semantic_entity_matching=False` by default to maintain backward compatibility:

```python
@dataclass
class TERAGConfig:
    use_semantic_entity_matching: bool = False  # Opt-in for now
```

### Migration Path

Users can opt-in by:
1. Providing an embedding model
2. Setting `use_semantic_entity_matching=True`

---

## Alternative Approaches Considered

### 1. Fuzzy String Matching (e.g., Levenshtein distance)

**Pros**: No embeddings needed  
**Cons**: 
- Doesn't handle synonyms
- "cash flow" vs "cashflow" still problematic (distance = 1)
- Requires tuning distance thresholds

**Decision**: Not sufficient for semantic understanding

### 2. Expand Query with Synonyms

**Pros**: Simple to implement  
**Cons**:
- Requires external synonym database
- Language-specific
- Doesn't handle domain-specific terms

**Decision**: Embeddings more flexible

### 3. Use Semantic Matching Only (Remove Text Matching)

**Pros**: Simpler code  
**Cons**:
- Slower (must compute all similarities)
- May miss exact matches
- Less interpretable

**Decision**: Hybrid approach (text + semantic) is best

---

## Success Metrics

### Quantitative

- **Recall improvement**: Measure on test queries with spelling variations
- **Precision**: Ensure semantic matches are relevant (threshold tuning)
- **Latency**: Query time increase < 10%

### Qualitative

- User feedback on query result quality
- Reduction in "no results found" cases
- Better handling of natural language variations

---

## Related Work

### Hybrid Retriever

TERAG already has a `HybridRetriever` that combines:
- PPR (entity-based graph traversal)
- Semantic retrieval (direct passage similarity)

This proposal makes PPR retrieval **internally hybrid** by using semantic similarity for entity matching, not just passage retrieval.

### Consistency

This change aligns PPR retrieval with the existing hybrid approach philosophy: **combine structural (graph) and semantic (embeddings) signals**.

---

## Next Steps

1. **Create GitHub Issue**: Document this proposal at https://github.com/rudranaik/terag/issues
2. **Create Feature Branch**: `feature/semantic-entity-matching`
3. **Implement Core Logic**: Update `_match_entities_to_concepts()`
4. **Add Tests**: Comprehensive test coverage
5. **Update Documentation**: README and docstrings
6. **Create PR**: For review and merge

---

## Questions for Discussion

1. Should semantic matching be **enabled by default** or opt-in?
2. What should the default threshold be? (Recommend: 0.7)
3. Should we log/expose which matching strategy was used for debugging?
4. Should we support multiple embedding models (not just OpenAI)?

---

## Appendix: Code Locations

### Files to Modify

1. **`terag/retrieval/ppr.py`**
   - Line 330-355: `_match_entities_to_concepts()` method
   
2. **`terag/core.py`**
   - Line 22-50: `TERAGConfig` dataclass
   - Line 193-211: `TERAGRetriever.__init__()`

3. **`README.md`**
   - Add semantic entity matching to features list
   - Update configuration guide

### New Files to Create

1. **`tests/test_semantic_entity_matching.py`**
   - Unit tests for new functionality

---

## Contact

For questions or feedback on this proposal:
- **GitHub Issues**: https://github.com/rudranaik/terag/issues
- **Email**: rudra91@gmail.com

---

**End of Proposal**
