"""Public API for TERAG."""

from importlib.metadata import PackageNotFoundError, version

from .config import (
    EmbeddingConfig,
    GraphConfig,
    NERConfig,
    RetrievalConfig,
    StorageConfig,
    TERAGConfig,
)
from .core import TERAG
from .retrieval.ppr import RetrievalMetrics, RetrievalResult

try:
    __version__ = version("terag")
except PackageNotFoundError:
    __version__ = "0.8.0"

__all__ = [
    "TERAG",
    "TERAGConfig",
    "GraphConfig",
    "RetrievalConfig",
    "NERConfig",
    "EmbeddingConfig",
    "StorageConfig",
    "RetrievalMetrics",
    "RetrievalResult",
    "__version__",
]
