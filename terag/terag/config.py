"""Configuration models for TERAG."""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Type, TypeVar, Union


RETRIEVAL_METHODS = {"ppr", "semantic", "hybrid"}
LLM_PROVIDERS = {"groq", "openai"}
ConfigT = TypeVar("ConfigT")


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_probability(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _validate_positive_probability(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1.")


def _validate_non_negative(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_non_empty_string(name: str, value: Optional[str]) -> None:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _coerce_section(value: Optional[Union[ConfigT, Dict]], cls: Type[ConfigT]) -> Optional[ConfigT]:
    if value is None:
        return None
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"{cls.__name__} must be a {cls.__name__} instance or a dict.")


@dataclass
class GraphConfig:
    """Graph construction settings."""

    min_concept_freq: int = 2
    max_concept_freq_ratio: float = 0.5
    enable_concept_clustering: bool = False

    def __post_init__(self) -> None:
        _validate_positive_int("min_concept_freq", self.min_concept_freq)
        _validate_positive_probability("max_concept_freq_ratio", self.max_concept_freq_ratio)


@dataclass
class RetrievalConfig:
    """Retrieval and ranking settings."""

    top_k: int = 10
    default_retrieval_method: str = "ppr"
    ppr_alpha: float = 0.15
    ppr_max_iterations: int = 100
    semantic_weight: float = 0.5
    frequency_weight: float = 0.5

    def __post_init__(self) -> None:
        _validate_positive_int("top_k", self.top_k)
        _validate_positive_int("ppr_max_iterations", self.ppr_max_iterations)
        _validate_probability("ppr_alpha", self.ppr_alpha)
        _validate_non_negative("semantic_weight", self.semantic_weight)
        _validate_non_negative("frequency_weight", self.frequency_weight)
        if self.semantic_weight == 0 and self.frequency_weight == 0:
            raise ValueError("semantic_weight and frequency_weight cannot both be 0.")
        if self.default_retrieval_method not in RETRIEVAL_METHODS:
            allowed = ", ".join(sorted(RETRIEVAL_METHODS))
            raise ValueError(f"default_retrieval_method must be one of: {allowed}.")


@dataclass
class NERConfig:
    """Named-entity extraction provider settings."""

    use_llm_for_ner: bool = False
    llm_provider: str = "groq"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    extraction_cache_dir: str = "extraction_cache"

    def __post_init__(self) -> None:
        if self.llm_provider not in LLM_PROVIDERS:
            allowed = ", ".join(sorted(LLM_PROVIDERS))
            raise ValueError(f"llm_provider must be one of: {allowed}.")
        _validate_non_empty_string("extraction_cache_dir", self.extraction_cache_dir)


@dataclass
class EmbeddingConfig:
    """Semantic entity matching settings."""

    use_semantic_entity_matching: bool = True
    semantic_match_threshold: float = 0.7

    def __post_init__(self) -> None:
        _validate_probability("semantic_match_threshold", self.semantic_match_threshold)


@dataclass
class StorageConfig:
    """Graph persistence settings."""

    auto_save_graph: bool = False
    graph_save_path: Optional[str] = "terag_graph.json"

    def __post_init__(self) -> None:
        if self.auto_save_graph:
            _validate_non_empty_string("graph_save_path", self.graph_save_path)


@dataclass
class TERAGConfig:
    """
    Backward-compatible TERAG configuration.

    New code can pass focused config sections. Existing flat keyword arguments
    continue to work and are mirrored into the focused sections.
    """

    # Focused config sections
    graph_config: Optional[Union[GraphConfig, Dict]] = None
    retrieval_config: Optional[Union[RetrievalConfig, Dict]] = None
    ner_config: Optional[Union[NERConfig, Dict]] = None
    embedding_config: Optional[Union[EmbeddingConfig, Dict]] = None
    storage_config: Optional[Union[StorageConfig, Dict]] = None

    # Backward-compatible flat graph construction settings
    min_concept_freq: int = 2
    max_concept_freq_ratio: float = 0.5
    enable_concept_clustering: bool = False

    # Backward-compatible flat PPR parameters
    ppr_alpha: float = 0.15
    ppr_max_iterations: int = 100

    # Backward-compatible flat weighting settings
    semantic_weight: float = 0.5
    frequency_weight: float = 0.5

    # Backward-compatible flat retrieval settings
    top_k: int = 10
    default_retrieval_method: str = "ppr"

    # Backward-compatible flat semantic matching settings
    use_semantic_entity_matching: bool = True
    semantic_match_threshold: float = 0.7

    # Backward-compatible flat NER settings
    use_llm_for_ner: bool = False
    llm_provider: str = "groq"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    extraction_cache_dir: str = "extraction_cache"

    # Backward-compatible flat storage settings
    auto_save_graph: bool = False
    graph_save_path: Optional[str] = "terag_graph.json"

    def __post_init__(self) -> None:
        self.graph_config = _coerce_section(self.graph_config, GraphConfig)
        self.retrieval_config = _coerce_section(self.retrieval_config, RetrievalConfig)
        self.ner_config = _coerce_section(self.ner_config, NERConfig)
        self.embedding_config = _coerce_section(self.embedding_config, EmbeddingConfig)
        self.storage_config = _coerce_section(self.storage_config, StorageConfig)

        self.graph_config = self.graph_config or GraphConfig(
            min_concept_freq=self.min_concept_freq,
            max_concept_freq_ratio=self.max_concept_freq_ratio,
            enable_concept_clustering=self.enable_concept_clustering,
        )
        self.retrieval_config = self.retrieval_config or RetrievalConfig(
            top_k=self.top_k,
            default_retrieval_method=self.default_retrieval_method,
            ppr_alpha=self.ppr_alpha,
            ppr_max_iterations=self.ppr_max_iterations,
            semantic_weight=self.semantic_weight,
            frequency_weight=self.frequency_weight,
        )
        self.ner_config = self.ner_config or NERConfig(
            use_llm_for_ner=self.use_llm_for_ner,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            llm_api_key=self.llm_api_key,
            extraction_cache_dir=self.extraction_cache_dir,
        )
        self.embedding_config = self.embedding_config or EmbeddingConfig(
            use_semantic_entity_matching=self.use_semantic_entity_matching,
            semantic_match_threshold=self.semantic_match_threshold,
        )
        self.storage_config = self.storage_config or StorageConfig(
            auto_save_graph=self.auto_save_graph,
            graph_save_path=self.graph_save_path,
        )

        self._sync_flat_fields()

    def _sync_flat_fields(self) -> None:
        self.min_concept_freq = self.graph_config.min_concept_freq
        self.max_concept_freq_ratio = self.graph_config.max_concept_freq_ratio
        self.enable_concept_clustering = self.graph_config.enable_concept_clustering

        self.top_k = self.retrieval_config.top_k
        self.default_retrieval_method = self.retrieval_config.default_retrieval_method
        self.ppr_alpha = self.retrieval_config.ppr_alpha
        self.ppr_max_iterations = self.retrieval_config.ppr_max_iterations
        self.semantic_weight = self.retrieval_config.semantic_weight
        self.frequency_weight = self.retrieval_config.frequency_weight

        self.use_llm_for_ner = self.ner_config.use_llm_for_ner
        self.llm_provider = self.ner_config.llm_provider
        self.llm_model = self.ner_config.llm_model
        self.llm_api_key = self.ner_config.llm_api_key
        self.extraction_cache_dir = self.ner_config.extraction_cache_dir

        self.use_semantic_entity_matching = self.embedding_config.use_semantic_entity_matching
        self.semantic_match_threshold = self.embedding_config.semantic_match_threshold

        self.auto_save_graph = self.storage_config.auto_save_graph
        self.graph_save_path = self.storage_config.graph_save_path

    def to_dict(self) -> Dict[str, Dict]:
        """Serialize config grouped by focused config sections."""
        return {
            "graph_config": asdict(self.graph_config),
            "retrieval_config": asdict(self.retrieval_config),
            "ner_config": asdict(self.ner_config),
            "embedding_config": asdict(self.embedding_config),
            "storage_config": asdict(self.storage_config),
        }
