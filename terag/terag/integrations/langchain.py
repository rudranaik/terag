"""LangChain integration for TERAG."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:  # pragma: no cover - exercised only without optional extra
    Document = None
    BaseRetriever = object
    _LANGCHAIN_IMPORT_ERROR = exc
else:
    _LANGCHAIN_IMPORT_ERROR = None


def _missing_langchain_error() -> ImportError:
    return ImportError(
        "LangChain integration requires langchain-core. "
        'Install it with `pip install "terag[langchain]"`.'
    )


def _result_to_document(result: Any) -> Any:
    if Document is None:
        raise _missing_langchain_error() from _LANGCHAIN_IMPORT_ERROR

    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata.update(
        {
            "id": getattr(result, "id", getattr(result, "passage_id", None)),
            "passage_id": getattr(result, "passage_id", None),
            "score": getattr(result, "score", None),
            "matched_concepts": list(getattr(result, "matched_concepts", []) or []),
        }
    )
    return Document(
        page_content=getattr(result, "content", getattr(result, "text", "")),
        metadata=metadata,
    )


def results_to_documents(results: Iterable[Any]) -> List[Any]:
    """Convert TERAG retrieval results into LangChain Documents."""
    return [_result_to_document(result) for result in results]


if _LANGCHAIN_IMPORT_ERROR is not None:

    class TERAGRetriever:  # pragma: no cover - exercised only without optional extra
        """Placeholder that raises a clear error when LangChain is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise _missing_langchain_error() from _LANGCHAIN_IMPORT_ERROR

else:
    from pydantic import ConfigDict

    class TERAGRetriever(BaseRetriever):
        """LangChain retriever adapter for TERAG."""

        terag: Any
        method: Optional[str] = None
        top_k: Optional[int] = None
        ppr_weight: float = 0.6
        semantic_weight: float = 0.4
        min_score_threshold: Optional[float] = None

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Any]:
            results = self.terag.query(
                query=query,
                method=self.method,
                top_k=self.top_k,
                ppr_weight=self.ppr_weight,
                semantic_weight=self.semantic_weight,
                min_score_threshold=self.min_score_threshold,
            )
            return results_to_documents(results)


__all__ = ["TERAGRetriever", "results_to_documents"]
