"""Public API for TERAG."""

from importlib.metadata import PackageNotFoundError, version

from .core import TERAG, TERAGConfig
from .retrieval.ppr import RetrievalMetrics, RetrievalResult

try:
    __version__ = version("terag")
except PackageNotFoundError:
    __version__ = "0.8.0"

__all__ = [
    "TERAG",
    "TERAGConfig",
    "RetrievalMetrics",
    "RetrievalResult",
    "__version__",
]
