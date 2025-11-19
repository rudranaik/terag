from dataclasses import dataclass
from typing import Set

@dataclass
class DuplicateCandidate:
    """Potential duplicate entity pair"""
    entity1: str
    entity2: str
    string_similarity: float
    embedding_similarity: float = 0.0
    graph_similarity: float = 0.0
    confidence_score: float = 0.0
    phase_detected: str = ""


@dataclass
class EntityCluster:
    """Cluster of duplicate entities"""
    canonical_entity: str  # The "main" entity name to use
    duplicate_entities: Set[str]
    confidence_score: float
    detection_method: str
