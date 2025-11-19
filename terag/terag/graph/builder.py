"""
TERAG Graph Builder

Constructs a directed, unweighted graph from document chunks:
- Passage nodes: Document chunks/passages
- Concept nodes: Named entities and key concepts
- Edges: Bidirectional connections between passages and their concepts
"""

import json
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

# Import our edge weight calculator
from terag.utils.edge_weights import EdgeWeightCalculator


@dataclass
class ConceptNode:
    """Represents a concept (named entity or key term) in the graph"""
    concept_id: str
    concept_text: str
    concept_type: str  # "named_entity" or "document_concept"
    frequency: int = 0  # Number of passages containing this concept
    passage_ids: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash(self.concept_id)


@dataclass
class PassageNode:
    """Represents a passage (chunk) in the graph"""
    passage_id: str
    content: str
    chunk_index: int
    concepts: Set[str] = field(default_factory=set)  # Concept IDs
    metadata: Dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.passage_id)


@dataclass
class TERAGGraph:
    """
    Directed, unweighted graph for TERAG
    G = (V, E) where V = passages ∪ concepts
    """
    passages: Dict[str, PassageNode] = field(default_factory=dict)
    concepts: Dict[str, ConceptNode] = field(default_factory=dict)

    # Adjacency lists for efficient traversal
    # passage -> concepts it contains with weights
    passage_to_concepts: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    # concept -> passages that contain it with weights
    concept_to_passages: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))

    def add_passage(self, passage: PassageNode):
        """Add a passage node to the graph"""
        self.passages[passage.passage_id] = passage

    def add_concept(self, concept: ConceptNode):
        """Add a concept node to the graph"""
        if concept.concept_id in self.concepts:
            # Merge with existing concept
            existing = self.concepts[concept.concept_id]
            existing.frequency += concept.frequency
            existing.passage_ids.update(concept.passage_ids)
        else:
            self.concepts[concept.concept_id] = concept

    def add_edge(self, passage_id: str, concept_id: str, weight: float = 1.0):
        """Add weighted bidirectional edge between passage and concept"""
        self.passage_to_concepts[passage_id][concept_id] = weight
        self.concept_to_passages[concept_id][passage_id] = weight

        # Update concept frequency
        if concept_id in self.concepts:
            self.concepts[concept_id].passage_ids.add(passage_id)
            self.concepts[concept_id].frequency = len(self.concepts[concept_id].passage_ids)

    def get_passage_neighbors(self, passage_id: str) -> Dict[str, float]:
        """Get all concept neighbors of a passage with weights"""
        return self.passage_to_concepts.get(passage_id, {})

    def get_concept_neighbors(self, concept_id: str) -> Dict[str, float]:
        """Get all passage neighbors of a concept with weights"""
        return self.concept_to_passages.get(concept_id, {})

    def get_statistics(self) -> Dict:
        """Return graph statistics"""
        total_edges = sum(len(concepts) for concepts in self.passage_to_concepts.values())

        return {
            "num_passages": len(self.passages),
            "num_concepts": len(self.concepts),
            "num_edges": total_edges,
            "avg_concepts_per_passage": total_edges / len(self.passages) if self.passages else 0,
            "avg_passages_per_concept": total_edges / len(self.concepts) if self.concepts else 0,
            "concept_frequency_distribution": self._get_frequency_distribution()
        }

    def _get_frequency_distribution(self) -> Dict[str, int]:
        """Get distribution of concept frequencies"""
        frequencies = [c.frequency for c in self.concepts.values()]
        if not frequencies:
            return {}

        return {
            "min": min(frequencies),
            "max": max(frequencies),
            "mean": np.mean(frequencies),
            "median": np.median(frequencies),
            "std": np.std(frequencies)
        }

    def save_to_file(self, filepath: str):
        """Save graph to JSON file"""
        data = {
            "passages": [
                {
                    "passage_id": p.passage_id,
                    "content": p.content,
                    "chunk_index": p.chunk_index,
                    "concepts": list(p.concepts),
                    "metadata": p.metadata
                }
                for p in self.passages.values()
            ],
            "concepts": [
                {
                    "concept_id": c.concept_id,
                    "concept_text": c.concept_text,
                    "concept_type": c.concept_type,
                    "frequency": c.frequency,
                    "passage_ids": list(c.passage_ids)
                }
                for c in self.concepts.values()
            ],
            "edges": {
                passage_id: list(concepts)
                for passage_id, concepts in self.passage_to_concepts.items()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'TERAGGraph':
        """Load graph from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = cls()

        # Load passages
        for p_data in data["passages"]:
            passage = PassageNode(
                passage_id=p_data["passage_id"],
                content=p_data["content"],
                chunk_index=p_data["chunk_index"],
                concepts=set(p_data["concepts"]),
                metadata=p_data["metadata"]
            )
            graph.add_passage(passage)

        # Load concepts
        for c_data in data["concepts"]:
            concept = ConceptNode(
                concept_id=c_data["concept_id"],
                concept_text=c_data["concept_text"],
                concept_type=c_data["concept_type"],
                frequency=c_data["frequency"],
                passage_ids=set(c_data["passage_ids"])
            )
            graph.add_concept(concept)

        # Load edges
        for passage_id, concept_ids in data["edges"].items():
            for concept_id in concept_ids:
                graph.add_edge(passage_id, concept_id)

        return graph

    def to_networkx(self, include_node_attributes: bool = True, include_edge_weights: bool = False, flatten_metadata: bool = False) -> 'nx.Graph':
        """
        Convert TERAG graph to NetworkX graph
        
        Args:
            include_node_attributes: Include node metadata as attributes
            include_edge_weights: Add edge weights based on concept frequency
            flatten_metadata: Flatten complex metadata for GraphML compatibility
            
        Returns:
            nx.Graph: NetworkX representation of the TERAG graph
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is not available. Install with: pip install networkx")
            
        # Create undirected graph (bipartite)
        G = nx.Graph()
        
        # Add passage nodes
        for passage_id, passage in self.passages.items():
            node_attrs = {
                'node_type': 'passage',
                'bipartite': 0  # Bipartite node set 0
            }
            
            if include_node_attributes:
                node_attrs.update({
                    'content': passage.content[:200] + "..." if len(passage.content) > 200 else passage.content,
                    'chunk_index': passage.chunk_index,
                    'num_concepts': len(passage.concepts)
                })
                
                if not flatten_metadata:
                    # Include full metadata as complex objects
                    node_attrs.update({
                        'full_content': passage.content,
                        'metadata': passage.metadata
                    })
                else:
                    # Flatten metadata for GraphML compatibility
                    node_attrs.update(self._flatten_metadata_for_export(passage.metadata, 'passage_'))
            
            G.add_node(passage_id, **node_attrs)
        
        # Add concept nodes
        for concept_id, concept in self.concepts.items():
            node_attrs = {
                'node_type': 'concept',
                'bipartite': 1  # Bipartite node set 1
            }
            
            if include_node_attributes:
                node_attrs.update({
                    'concept_text': concept.concept_text,
                    'concept_type': concept.concept_type,
                    'frequency': concept.frequency,
                    'num_passages': len(concept.passage_ids)
                })
            
            G.add_node(concept_id, **node_attrs)
        
        # Add edges
        for passage_id, concept_weights in self.passage_to_concepts.items():
            for concept_id, weight in concept_weights.items():
                if concept_id in self.concepts:
                    edge_attrs = {}
                    
                    if include_edge_weights:
                        edge_attrs['weight'] = weight
                        edge_attrs['concept_frequency'] = self.concepts[concept_id].frequency
                    
                    G.add_edge(passage_id, concept_id, **edge_attrs)
        
        return G
    
    def _flatten_metadata_for_export(self, metadata: Dict, prefix: str = '') -> Dict[str, any]:
        """
        Flatten nested metadata dictionaries for GraphML export compatibility
        
        Args:
            metadata: Nested metadata dictionary
            prefix: Prefix for flattened keys
            
        Returns:
            Flattened dictionary with only simple data types
        """
        flattened = {}
        
        def flatten_dict(d, parent_key=''):
            for key, value in d.items():
                new_key = f"{parent_key}{key}" if parent_key else key
                
                if isinstance(value, dict):
                    flatten_dict(value, f"{new_key}_")
                elif isinstance(value, list):
                    # Convert lists to strings
                    if all(isinstance(item, str) for item in value):
                        flattened[new_key] = ', '.join(value)
                    else:
                        flattened[new_key] = str(value)
                elif isinstance(value, (str, int, float, bool)):
                    flattened[new_key] = value
                elif value is None:
                    flattened[new_key] = ""  # Convert None to empty string for GraphML
                else:
                    # Convert complex types to strings
                    flattened[new_key] = str(value)
        
        flatten_dict(metadata, prefix)
        return flattened

    def get_networkx_statistics(self) -> Dict:
        """Get NetworkX graph statistics"""
        if not NETWORKX_AVAILABLE:
            return {"error": "NetworkX not available"}
            
        G = self.to_networkx()
        
        # Basic stats
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'num_connected_components': nx.number_connected_components(G)
        }
        
        # Bipartite stats
        passage_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'passage']
        concept_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'concept']
        
        stats.update({
            'num_passage_nodes': len(passage_nodes),
            'num_concept_nodes': len(concept_nodes),
            'is_bipartite': nx.is_bipartite(G)
        })
        
        # Degree statistics
        degrees = [G.degree(n) for n in G.nodes()]
        if degrees:
            stats.update({
                'avg_degree': np.mean(degrees),
                'max_degree': max(degrees),
                'min_degree': min(degrees)
            })
        
        # Central nodes (top 5 by degree)
        if degrees:
            degree_dict = dict(G.degree())
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['top_nodes_by_degree'] = top_nodes
        
        return stats


class GraphBuilder:
    """
    Builds TERAG graph from document chunks with hybrid edge weighting
    """

    def __init__(
        self,
        min_concept_freq: int = 1,
        max_concept_freq_ratio: float = 0.5,  # Filter concepts in > 50% of passages
        enable_concept_clustering: bool = False,
        enable_edge_weights: bool = True,  # Enable hybrid edge weighting
        separate_entity_concept_types: bool = True  # Keep entities and concepts separate
    ):
        self.min_concept_freq = min_concept_freq
        self.max_concept_freq_ratio = max_concept_freq_ratio
        self.enable_concept_clustering = enable_concept_clustering
        self.enable_edge_weights = enable_edge_weights
        self.separate_entity_concept_types = separate_entity_concept_types
        
        # Initialize edge weight calculator
        if self.enable_edge_weights:
            self.weight_calculator = EdgeWeightCalculator()

    def build_graph_from_chunks(
        self,
        chunks: List[Dict],
        extract_concepts_fn: callable
    ) -> TERAGGraph:
        """
        Build TERAG graph from chunks

        Args:
            chunks: List of chunk dictionaries with 'content' and metadata
            extract_concepts_fn: Function to extract concepts from text
                                Returns (named_entities, document_concepts)

        Returns:
            TERAGGraph: Constructed graph
        """
        graph = TERAGGraph()

        print(f"Building graph from {len(chunks)} chunks...")

        # Phase 1: Create passage nodes and extract concepts
        all_concepts = []
        for i, chunk in enumerate(chunks):
            passage_id = f"passage_{i}"
            content = chunk.get('content', '')

            # Create passage node
            passage = PassageNode(
                passage_id=passage_id,
                content=content,
                chunk_index=i,
                metadata=chunk.get('metadata', {})
            )

            # Extract concepts from passage
            named_entities, doc_concepts = extract_concepts_fn(content)

            # Create concept nodes
            for entity in named_entities:
                concept_id = self._normalize_concept(entity)
                concept = ConceptNode(
                    concept_id=concept_id,
                    concept_text=entity,
                    concept_type="named_entity",
                    frequency=1,
                    passage_ids={passage_id}
                )
                all_concepts.append(concept)
                passage.concepts.add(concept_id)

            for concept_text in doc_concepts:
                concept_id = self._normalize_concept(concept_text)
                concept = ConceptNode(
                    concept_id=concept_id,
                    concept_text=concept_text,
                    concept_type="document_concept",
                    frequency=1,
                    passage_ids={passage_id}
                )
                all_concepts.append(concept)
                passage.concepts.add(concept_id)

            graph.add_passage(passage)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks...")

        # Phase 2: Merge and filter concepts
        print("Merging and filtering concepts...")
        concept_counter = Counter(c.concept_id for c in all_concepts)

        total_passages = len(chunks)
        max_freq = int(total_passages * self.max_concept_freq_ratio)

        for concept in all_concepts:
            freq = concept_counter[concept.concept_id]

            # Filter by frequency
            if freq < self.min_concept_freq or freq > max_freq:
                continue

            graph.add_concept(concept)

        # Phase 3: Build edges
        print("Building edges...")
        for passage_id, passage in graph.passages.items():
            for concept_id in passage.concepts:
                if concept_id in graph.concepts:
                    graph.add_edge(passage_id, concept_id)

        # Phase 4: Optional concept clustering
        if self.enable_concept_clustering:
            print("Clustering similar concepts...")
            graph = self._cluster_concepts(graph)

        # Print statistics
        stats = graph.get_statistics()
        print(f"\nGraph construction complete!")
        print(f"  Passages: {stats['num_passages']}")
        print(f"  Concepts: {stats['num_concepts']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Avg concepts/passage: {stats['avg_concepts_per_passage']:.2f}")
        print(f"  Avg passages/concept: {stats['avg_passages_per_concept']:.2f}")

        return graph

    def _normalize_concept(self, text: str) -> str:
        """Normalize concept text to create concept ID"""
        # Convert to lowercase, remove extra spaces
        normalized = " ".join(text.lower().strip().split())
        return normalized

    def _cluster_concepts(self, graph: TERAGGraph) -> TERAGGraph:
        """
        Cluster similar concepts to reduce graph size
        (Placeholder - can be enhanced with embedding-based clustering)
        """
        # TODO: Implement embedding-based concept clustering
        # For now, return graph as-is
        return graph

    def build_graph_from_ner_extractions(
        self,
        ner_extractions_file: str
    ) -> TERAGGraph:
        """
        Build TERAG graph from NER extractions JSON file with hybrid edge weights
        
        Args:
            ner_extractions_file: Path to ner_extractions.json file
            
        Returns:
            TERAGGraph: Constructed weighted graph
        """
        print(f"Building graph from NER extractions: {ner_extractions_file}")
        
        # Load extractions
        with open(ner_extractions_file, 'r', encoding='utf-8') as f:
            extractions = json.load(f)
        
        print(f"Loaded {len(extractions)} extraction records")
        
        graph = TERAGGraph()
        all_passage_texts = []  # For IDF calculation
        passage_data = []  # Store for weight calculation
        
        # Phase 1: Collect all passages and extract basic info
        for extraction in extractions:
            content = extraction.get('content', '')
            if len(content.strip()) < 10:  # Skip very short passages
                continue
                
            all_passage_texts.append(content)
            passage_data.append(extraction)
        
        print(f"Processing {len(passage_data)} valid passages...")
        
        # Phase 2: Create passage and concept nodes
        all_concepts = []
        passage_weights_data = {}  # Store for edge weight calculation
        
        for i, extraction in enumerate(passage_data):
            passage_hash = extraction.get('passage_hash', f'passage_{i}')
            content = extraction.get('content', '')
            entities = extraction.get('entities', [])
            concepts = extraction.get('concepts', [])
            
            # Create passage node
            passage = PassageNode(
                passage_id=passage_hash,
                content=content,
                chunk_index=extraction.get('passage_metadata', {}).get('original_index', i),
                metadata={
                    'document_metadata': extraction.get('document_metadata', {}),
                    'passage_metadata': extraction.get('passage_metadata', {}),
                    'timestamp': extraction.get('timestamp', ''),
                    'model_used': extraction.get('model_used', ''),
                    'processing_time': extraction.get('processing_time', 0)
                }
            )
            
            # Calculate edge weights if enabled
            if self.enable_edge_weights:
                # Create entity type mapping (simple heuristics for now)
                entity_types = self._infer_entity_types(entities, concepts)
                
                edge_weights = self.weight_calculator.calculate_edge_weights(
                    content, entities, concepts, all_passage_texts, entity_types
                )
                passage_weights_data[passage_hash] = edge_weights
            
            # Create concept nodes for entities
            if self.separate_entity_concept_types:
                concept_type_entities = "named_entity"
                concept_type_concepts = "document_concept" 
            else:
                concept_type_entities = concept_type_concepts = "concept"
            
            for entity in entities:
                concept_id = self._normalize_concept(entity)
                concept = ConceptNode(
                    concept_id=concept_id,
                    concept_text=entity,
                    concept_type=concept_type_entities,
                    frequency=1,
                    passage_ids={passage_hash}
                )
                all_concepts.append(concept)
                passage.concepts.add(concept_id)
            
            for concept_text in concepts:
                concept_id = self._normalize_concept(concept_text)
                concept = ConceptNode(
                    concept_id=concept_id,
                    concept_text=concept_text,
                    concept_type=concept_type_concepts,
                    frequency=1,
                    passage_ids={passage_hash}
                )
                all_concepts.append(concept)
                passage.concepts.add(concept_id)
            
            graph.add_passage(passage)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(passage_data)} passages...")
        
        # Phase 3: Merge and filter concepts by frequency
        print("Merging and filtering concepts...")
        concept_counter = Counter(c.concept_id for c in all_concepts)
        
        total_passages = len(passage_data)
        max_freq = int(total_passages * self.max_concept_freq_ratio)
        
        filtered_concepts = 0
        for concept in all_concepts:
            freq = concept_counter[concept.concept_id]
            
            # Filter by frequency
            if freq < self.min_concept_freq or freq > max_freq:
                filtered_concepts += 1
                continue
            
            graph.add_concept(concept)
        
        print(f"  Filtered {filtered_concepts} concepts by frequency")
        print(f"  Kept {len(graph.concepts)} concepts")
        
        # Phase 4: Build weighted edges
        print("Building weighted edges...")
        total_edges = 0
        
        for passage_id, passage in graph.passages.items():
            for concept_id in passage.concepts:
                if concept_id in graph.concepts:
                    # Get edge weight
                    weight = 1.0  # Default weight
                    if self.enable_edge_weights and passage_id in passage_weights_data:
                        # Find the original entity/concept text for this concept_id
                        concept_text = graph.concepts[concept_id].concept_text
                        weight = passage_weights_data[passage_id].get(concept_text, 1.0)
                    
                    graph.add_edge(passage_id, concept_id, weight)
                    total_edges += 1
        
        print(f"  Created {total_edges} weighted edges")
        
        # Phase 5: Optional concept clustering
        if self.enable_concept_clustering:
            print("Clustering similar concepts...")
            graph = self._cluster_concepts(graph)
        
        # Print final statistics
        stats = graph.get_statistics()
        print(f"\nGraph construction complete!")
        print(f"  Passages: {stats['num_passages']}")
        print(f"  Concepts: {stats['num_concepts']}")
        print(f"    - Entities: {sum(1 for c in graph.concepts.values() if c.concept_type == 'named_entity')}")
        print(f"    - Concepts: {sum(1 for c in graph.concepts.values() if c.concept_type == 'document_concept')}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Avg concepts/passage: {stats['avg_concepts_per_passage']:.2f}")
        print(f"  Avg passages/concept: {stats['avg_passages_per_concept']:.2f}")
        
        if self.enable_edge_weights:
            # Calculate weight statistics
            all_weights = []
            for passage_concepts in graph.passage_to_concepts.values():
                all_weights.extend(passage_concepts.values())
            
            if all_weights:
                print(f"  Edge weights - Min: {min(all_weights):.3f}, Max: {max(all_weights):.3f}, Avg: {sum(all_weights)/len(all_weights):.3f}")
        
        return graph
    
    def _infer_entity_types(self, entities: List[str], concepts: List[str]) -> Dict[str, str]:
        """
        Simple heuristics to infer entity types for edge weight calculation
        
        TODO: Could be enhanced by storing entity types in NER extraction
        """
        entity_types = {}
        
        for entity in entities:
            # Simple heuristics based on entity text patterns
            if any(word in entity.lower() for word in ['inc', 'corp', 'ltd', 'llc', 'company', 'group']):
                entity_types[entity] = "ORG"
            elif entity.replace(' ', '').replace(',', '').replace('%', '').isdigit():
                entity_types[entity] = "DATE" if len(entity) == 4 else "MONEY"
            elif '%' in entity:
                entity_types[entity] = "PERCENT"
            elif '$' in entity or 'billion' in entity.lower() or 'million' in entity.lower():
                entity_types[entity] = "MONEY"
            elif any(month in entity.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']) or 'q1' in entity.lower() or 'q2' in entity.lower() or 'q3' in entity.lower() or 'q4' in entity.lower():
                entity_types[entity] = "DATE"
            elif entity[0].isupper() and len(entity.split()) <= 3:  # Capitalized short phrases
                entity_types[entity] = "PERSON"  # Could be person or organization
            else:
                entity_types[entity] = "UNKNOWN"
        
        # All concepts are tagged as concepts
        for concept in concepts:
            entity_types[concept] = "CONCEPT"
            
        return entity_types


def build_graph_from_chunks_file(
    chunks_file: str,
    extract_concepts_fn: callable,
    output_file: Optional[str] = None,
    **builder_kwargs
) -> TERAGGraph:
    """
    Build TERAG graph from chunks JSON file

    Args:
        chunks_file: Path to chunks JSON file
        extract_concepts_fn: Function to extract concepts
        output_file: Optional path to save graph
        **builder_kwargs: Additional arguments for GraphBuilder

    Returns:
        TERAGGraph: Constructed graph
    """
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Build graph
    builder = GraphBuilder(**builder_kwargs)
    graph = builder.build_graph_from_chunks(chunks, extract_concepts_fn)

    # Save if requested
    if output_file:
        print(f"\nSaving graph to {output_file}...")
        graph.save_to_file(output_file)

    return graph


if __name__ == "__main__":
    # Test with a simple example
    def dummy_extract_concepts(text: str) -> Tuple[List[str], List[str]]:
        """Dummy concept extractor for testing"""
        # Extract capitalized words as named entities
        words = text.split()
        named_entities = [w for w in words if w[0].isupper() and len(w) > 2]
        # Extract long words as document concepts
        doc_concepts = [w.lower() for w in words if len(w) > 8]
        return named_entities, doc_concepts

    # Test chunks
    test_chunks = [
        {"content": "Apple Inc announced revenue growth in California market."},
        {"content": "Microsoft Corporation expanded operations in Washington."},
        {"content": "Apple and Microsoft compete in technology sector."}
    ]

    # Build graph
    builder = GraphBuilder(min_concept_freq=1)
    graph = builder.build_graph_from_chunks(test_chunks, dummy_extract_concepts)

    # Print results
    print("\n" + "="*50)
    print("Test Graph:")
    for passage_id, concepts in graph.passage_to_concepts.items():
        print(f"{passage_id}: {concepts}")

    print("\nConcepts:")
    for concept_id, concept in graph.concepts.items():
        print(f"{concept_id} ({concept.concept_type}): freq={concept.frequency}")
