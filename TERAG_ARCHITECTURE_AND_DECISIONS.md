# TERAG: Architecture and Design Decisions Documentation

## Overview
This document captures all architectural decisions, design choices, and implementation details for the TERAG (Token-Efficient Graph-Based RAG) dual-layer retrieval system. This serves as the foundation for productionization and potential PyPI publication.

## Table of Contents
1. [Core Architecture](#core-architecture)
2. [Pipeline Components](#pipeline-components)
3. [Key Design Decisions](#key-design-decisions)
4. [Technical Choices](#technical-choices)
5. [Performance Optimizations](#performance-optimizations)
6. [File Structure and Responsibilities](#file-structure-and-responsibilities)
7. [Production Considerations](#production-considerations)

---

## Core Architecture

### High-Level System Design
TERAG implements a **dual-layer retrieval architecture** combining:
1. **Entity-based PPR (Personalized PageRank)** - Graph traversal from query entities
2. **Direct semantic similarity** - Embedding-based passage matching
3. **Hybrid fusion** - Intelligent score combination

### Graph Structure
- **Bipartite graph**: Passages ↔ Concepts (named entities + document concepts)
- **Weighted edges**: TF-IDF + positional + contextual scoring
- **JSON serialization**: Custom format for persistence
- **NetworkX compatibility**: Optional export for visualization

---

## Pipeline Components

### 1. Document Ingestion (`ingest_chunks.py`)
**Purpose**: Single-entry point for complete document processing

**Pipeline Steps**:
1. **NER Extraction** → Extract entities and concepts from JSON documents
2. **Graph Building** → Create bipartite graph with weighted edges
3. **Graph Merging** → Combine with existing graph (incremental ingestion)
4. **Entity Deduplication** → Remove duplicate entities across documents
5. **Final Output** → Production-ready graph file

**Key Design**: 
- **Interruption-proof**: Each step saves intermediate results
- **Auto-resume**: Detects existing graphs and merges intelligently
- **Progress tracking**: Real-time feedback for long-running operations

### 2. Named Entity Recognition (`json_ner_demo.py`)
**Purpose**: Extract structured entities and concepts from document chunks

**Architecture**:
- **Dual extraction**: Named entities (spaCy) + document concepts (TF-IDF)
- **JSON input**: Handles complex nested document structures
- **Caching system**: File-based caching for expensive NER operations
- **Batch processing**: Efficient handling of large document sets

### 3. Graph Construction (`graph_builder.py`)
**Purpose**: Transform NER extractions into weighted bipartite graph

**Core Components**:
- **PassageNode**: Document chunks with metadata
- **ConceptNode**: Entities and concepts with frequency tracking
- **EdgeWeightCalculator**: Hybrid scoring (TF-IDF + position + type + context)
- **GraphBuilder**: Main orchestration class

**Edge Weighting Strategy**:
```
Final Weight = α×TF-IDF + β×Position + γ×Type + δ×Context
```

### 4. Entity Deduplication
**Two-tier approach** addressing computational complexity:

#### Original Deduplicator (`entity_deduplicator.py`)
- **3-phase process**: String → Embedding → Graph validation
- **SentenceTransformer**: Local embedding model
- **Issue**: Slow for large entity sets (O(n²) comparisons)

#### OpenAI-Enhanced Deduplicator (`openai_smart_dedup.py`)
- **Same 3-phase process** but with OpenAI embeddings
- **Sampling strategy**: Limits embedding comparisons to 1000 entities
- **Two-phase execution**: Fast string matching → Semantic embedding matching
- **Why created**: Original couldn't handle 10K+ entities efficiently

### 5. Retrieval System (`hybrid_retriever.py`)
**Purpose**: Implement dual-layer retrieval with score fusion

**Components**:
- **QueryProcessor**: Entity extraction from queries using embeddings
- **PPRRetriever**: Graph-based retrieval via Personalized PageRank
- **SemanticRetriever**: Direct embedding similarity matching
- **HybridRetriever**: Score fusion and final ranking

**Fusion Methods**:
- `weighted_sum`: Linear combination with configurable weights
- `max`: Maximum of PPR and semantic scores
- `harmonic_mean`: Balanced approach favoring dual matches

---

## Key Design Decisions

### 1. OpenAI Embeddings Choice
**Decision**: Use OpenAI `text-embedding-3-small` throughout system

**Rationale**:
- **Quality**: Superior semantic understanding vs local models
- **Consistency**: Same embeddings for deduplication and retrieval
- **Cost-effective**: $0.00002 per 1K tokens
- **API stability**: Production-grade service with reliability

**Implementation**:
- **Persistent caching**: 144MB+ cache prevents re-computation
- **Batch processing**: 500 entities per API call for efficiency
- **Cost tracking**: Built-in usage monitoring

### 2. Dual-Layer Architecture
**Decision**: Combine graph-based and semantic retrieval

**Rationale**:
- **Graph limitations**: May miss semantically similar but unconnected content
- **Embedding limitations**: May miss structured entity relationships
- **Hybrid approach**: Captures both structural and semantic relevance
- **Configurable fusion**: Allows domain-specific tuning

### 3. Bipartite Graph Structure
**Decision**: Passages ↔ Concepts (not passage-to-passage connections)

**Rationale**:
- **Scalability**: O(P×C) vs O(P²) for passage connections
- **Interpretability**: Clear entity-based reasoning paths
- **Memory efficiency**: Sparse representation for large corpora
- **Query optimization**: Direct entity → passage traversal

### 4. Incremental Ingestion Design
**Decision**: Support adding new documents to existing graphs

**Rationale**:
- **Production reality**: Documents arrive continuously
- **Cost efficiency**: Avoid full re-processing
- **User experience**: Near real-time content updates
- **Graph preservation**: Maintain existing entity relationships

### 5. Three-Phase Deduplication
**Decision**: String → Embedding → Graph validation

**Rationale**:
- **Accuracy layers**: Catch obvious duplicates early (fast)
- **Semantic matching**: Find conceptually similar entities
- **Graph validation**: Use co-occurrence for confidence
- **Performance**: Most duplicates found in fast string phase

---

## Technical Choices

### 1. JSON-Based Graph Serialization
**Choice**: Custom JSON format over NetworkX pickle

**Benefits**:
- **Human readable**: Easy debugging and inspection
- **Cross-platform**: No Python version dependencies
- **Lightweight**: Smaller files than pickled NetworkX graphs
- **Portable**: Easy integration with other systems

### 2. File-Based Caching Strategy
**Choice**: Pickle-based embedding cache with metadata

**Benefits**:
- **Persistence**: Survives across sessions
- **Atomic operations**: Safe concurrent access
- **Metadata tracking**: Cost and usage monitoring
- **Automatic management**: No manual cache maintenance

### 3. Subprocess-Based Pipeline
**Choice**: Shell out to separate scripts vs in-process execution

**Benefits**:
- **Memory isolation**: Large NER operations don't affect main process
- **Progress tracking**: Real-time feedback via stdout monitoring
- **Interruptibility**: Can cancel long-running operations
- **Modularity**: Each component testable independently

### 4. Configuration Through CLI Arguments
**Choice**: Argparse-based configuration vs config files

**Benefits**:
- **Immediate feedback**: No config file management
- **Scriptable**: Easy automation and testing
- **Documentation**: Built-in help and validation
- **Override-friendly**: Easy parameter tuning

---

## Performance Optimizations

### 1. Embedding Cache Architecture
**Problem**: 10K+ entities × multiple components = expensive re-computation
**Solution**: Shared persistent cache with intelligent key generation

**Implementation**:
```python
cache_key = hashlib.md5(f"{model_name}:{text_content}".encode()).hexdigest()
```

### 2. Batch API Operations
**Problem**: Individual API calls for each embedding
**Solution**: Batch processing with configurable sizes

**Optimization**: 500 entities/batch balances API limits with latency

### 3. Two-Phase Deduplication
**Problem**: O(n²) embedding comparisons for large entity sets
**Solution**: String matching first, embeddings on reduced set

**Result**: 10K entities → ~2K after string dedup → manageable embedding phase

### 4. Sampling Strategy for Embeddings
**Problem**: Still too many pairs even after string dedup
**Solution**: Sample 1000 entities for embedding comparison

**Trade-off**: May miss some semantic duplicates but maintains feasible runtime

### 5. Progress Indicators
**Problem**: Long-running operations appear frozen
**Solution**: Multi-threaded progress tracking with real-time updates

---

## File Structure and Responsibilities

### Core Pipeline Files
- `ingest_chunks.py` - **Main entry point** for document ingestion
- `json_ner_demo.py` - NER extraction from JSON documents
- `graph_builder.py` - Graph construction and edge weighting
- `hybrid_retriever.py` - Dual-layer retrieval orchestration
- `retrieval_demo.py` - Interactive query interface

### Specialized Components
- `embedding_manager.py` - OpenAI embedding management with caching
- `entity_deduplicator.py` - Original 3-phase deduplication
- `openai_smart_dedup.py` - **OpenAI-optimized deduplication**
- `entity_merger.py` - Graph restructuring after deduplication
- `query_processor.py` - Query entity extraction and processing
- `semantic_retriever.py` - Direct embedding similarity retrieval
- `ppr_retriever.py` - Graph-based PPR retrieval
- `query_explainer.py` - Step-by-step retrieval explanation

### Support Files
- `edge_weight_calculator.py` - Hybrid edge scoring algorithms
- `json_ingestion.py` - JSON parsing and structure analysis
- `extraction_cache.py` - File-based caching utilities

### Utility Scripts
- `merge_graphs.py` - Manual graph merging utility
- `fast_retrieval_demo.py` - Progress-tracked retrieval demo
- `smart_dedup.py` - Alternative deduplication approach
- `resume_dedup.py` - Resume from partial deduplication

---

## Production Considerations

### 1. Why `openai_smart_dedup.py` Was Necessary
**Original Problem**: `entity_deduplicator.py` couldn't handle large-scale deduplication

**Issues**:
- **Computational complexity**: O(n²) for 10K+ entities = hours
- **Memory usage**: All embeddings loaded simultaneously
- **Different embedding model**: SentenceTransformer vs OpenAI consistency
- **No sampling strategy**: Attempted full pairwise comparison

**Solution**: Specialized OpenAI-based deduplicator with:
- **Phased approach**: String filtering → Embedding matching
- **Sampling strategy**: Limit embedding pairs to manageable size
- **OpenAI integration**: Consistent embeddings across pipeline
- **Progress tracking**: Real-time feedback for long operations

### 2. Pipeline Integration Challenges
**Current State**: Multiple scripts with overlapping functionality

**Issues**:
- `ingest_chunks.py` uses original `entity_deduplicator.py`
- `openai_smart_dedup.py` runs separately with better performance
- Embedding caching not perfectly shared across all components
- No unified configuration management

### 3. Codebase Cleanup Requirements
**Problem**: Accumulated experimental and duplicate files

**Impact**:
- **Developer confusion**: Multiple similar scripts
- **Maintenance burden**: Changes need multiple updates
- **Production readiness**: Unclear which components are "official"

---

## Future Productionization Roadmap

### Phase 1: Consolidation
1. **Integrate optimized deduplication** into main pipeline
2. **Unify embedding management** across all components
3. **Remove experimental scripts** and consolidate functionality
4. **Standardize configuration** approach

### Phase 2: API Design
1. **Python package structure** with clear entry points
2. **Configuration classes** instead of CLI arguments
3. **Callback system** for progress monitoring
4. **Exception handling** and error recovery

### Phase 3: PyPI Preparation
1. **Setup.py and packaging** configuration
2. **Comprehensive documentation** and examples
3. **Unit test suite** for core functionality
4. **Performance benchmarks** and optimization guidelines

### Phase 4: Production Features
1. **Distributed processing** for large corpora
2. **Streaming ingestion** for real-time updates
3. **Monitoring and observability** integration
4. **Multi-model support** for different embedding providers

---

## Cost Analysis and Optimization

### Current Costs (per 10K entities)
- **Initial embedding**: ~$0.20 for full entity set
- **Incremental updates**: ~$0.02-0.05 for new entities only
- **Query processing**: ~$0.0001-0.0002 per query
- **Total monthly** (1K queries/day): ~$5-10

### Optimization Strategies
1. **Aggressive caching**: 95%+ cache hit rates in production
2. **Batch processing**: Reduce API overhead
3. **Incremental updates**: Only embed new content
4. **Model alternatives**: Consider local models for cost-sensitive use cases

---

## Conclusion

TERAG represents a sophisticated approach to retrieval-augmented generation, balancing performance, accuracy, and cost. The dual-layer architecture addresses limitations of purely graph-based or embedding-based approaches, while the OpenAI integration ensures high-quality semantic understanding.

The current implementation successfully handles large-scale document ingestion and retrieval, but requires consolidation and cleanup for production deployment. The specialized deduplication component (`openai_smart_dedup.py`) demonstrates the need for performance-optimized variants of core algorithms when scaling to real-world data sizes.

For PyPI publication, the primary focus should be on consolidating the proven components, eliminating experimental code, and providing a clean, documented interface for the complete TERAG pipeline.