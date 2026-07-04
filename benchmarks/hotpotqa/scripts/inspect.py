"""Write a qualitative HotPotQA retrieval inspection report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import load_config, read_json, repo_relative, repo_root, resolve_repo_path, sample_dir, write_json
from .evaluate import _install_query_ner_cache, _prewarm_extraction_cache, _result_passage_id


def _load_terag(config: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Any:
    root = repo_root()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    for import_path in (root, root / "terag"):
        path_str = str(import_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from terag import TERAG, TERAGConfig
    from terag.embeddings.manager import EmbeddingManager

    config_kwargs = dict(config.get("terag_config", {}))
    performance_config = dict(config.get("performance", {}))
    embedding_model = None
    embedding_config = config.get("embedding_model")

    if embedding_config:
        if embedding_config.get("provider") != "openai":
            raise ValueError(f"Unsupported embedding provider: {embedding_config.get('provider')}")
        embedding_model = EmbeddingManager(
            model=embedding_config.get("model", "text-embedding-3-small"),
            cache_dir=str(resolve_repo_path(embedding_config.get("cache_dir", ".benchmark_cache/embeddings"))),
        )

    if performance_config.get("prewarm_extraction_cache"):
        _prewarm_extraction_cache(
            chunks,
            config_kwargs,
            workers=int(performance_config.get("extraction_workers", 4)),
        )

    terag = TERAG.from_chunks(
        chunks,
        config=TERAGConfig(**config_kwargs),
        embedding_model=embedding_model,
        verbose=False,
    )

    query_cache_path = performance_config.get("query_cache_path")
    if query_cache_path:
        _install_query_ner_cache(terag, resolve_repo_path(query_cache_path))

    return terag


def _chunk_record(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata", {}) or {}
    return {
        "passage_id": int(metadata.get("passage_id", -1)),
        "title": metadata.get("title"),
        "source": metadata.get("source"),
        "original_passage_id": metadata.get("original_passage_id"),
        "content": chunk.get("content", ""),
    }


def _retrieved_record(result: Any, chunks_by_id: Dict[int, Dict[str, Any]], supporting_ids: set[int]) -> Dict[str, Any]:
    passage_id = _result_passage_id(result)
    chunk = chunks_by_id.get(passage_id or -1, {})
    metadata = getattr(result, "metadata", {}) or {}
    return {
        "rank": None,
        "passage_id": passage_id,
        "title": metadata.get("title") or chunk.get("metadata", {}).get("title"),
        "score": round(float(getattr(result, "score", 0.0)), 8),
        "is_expected_support": passage_id in supporting_ids if passage_id is not None else False,
        "matched_concepts": list(getattr(result, "matched_concepts", []) or []),
        "content": getattr(result, "content", ""),
    }


def _snippet(text: str, limit: int = 650) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _format_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# HotPotQA Retrieval Inspection",
        "",
        f"**Date:** {report['run']['finished_at']}",
        f"**Sample:** {report['sample']['sample_name']}",
        f"**Method:** {report['run']['method']}",
        f"**Top K:** {report['run']['top_k']}",
        f"**Questions inspected:** {len(report['questions'])}",
        "",
        "This report shows the real question, the expected supporting chunks, and the chunks TERAG retrieved.",
        "",
    ]

    for idx, item in enumerate(report["questions"], 1):
        lines.extend(
            [
                f"## {idx}. {item['question']}",
                "",
                f"- Answer: `{item['answer']}`",
                f"- Type/level: `{item['type']}` / `{item['level']}`",
                f"- Query entities: {', '.join(item['query_entities']) or '(none)'}",
                f"- Expected support found in top {report['run']['top_k']}: {item['support_found_count']}/{item['support_expected_count']}",
                "",
                "### Expected Supporting Chunks",
                "",
            ]
        )

        for support in item["expected_support"]:
            lines.extend(
                [
                    f"**Passage {support['passage_id']} - {support.get('title') or 'Untitled'}**",
                    "",
                    f"> {_snippet(support['content'], 900)}",
                    "",
                ]
            )

        lines.extend(["### Retrieved Chunks", ""])
        for retrieved in item["retrieved"]:
            marker = "EXPECTED SUPPORT" if retrieved["is_expected_support"] else "distractor"
            concepts = ", ".join(retrieved["matched_concepts"][:8]) or "(none)"
            lines.extend(
                [
                    f"**Rank {retrieved['rank']} - Passage {retrieved['passage_id']} - {retrieved.get('title') or 'Untitled'}**",
                    "",
                    f"- Score: `{retrieved['score']}`",
                    f"- Label: `{marker}`",
                    f"- Matched concepts: {concepts}",
                    "",
                    f"> {_snippet(retrieved['content'])}",
                    "",
                ]
            )

    return "\n".join(lines)


def inspect(config: Dict[str, Any], limit: Optional[int] = None, top_k_override: Optional[int] = None) -> Dict[str, Any]:
    prepared_dir = sample_dir(config)
    chunks_path = prepared_dir / "chunks.json"
    eval_path = prepared_dir / "eval_qa.json"
    manifest_path = prepared_dir / "manifest.json"
    if not chunks_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Prepared sample not found in {prepared_dir}. "
            "Run `python -m benchmarks.hotpotqa.scripts.sample --config ...` first."
        )

    chunks = read_json(chunks_path)
    eval_qas = read_json(eval_path)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if limit is not None:
        eval_qas = eval_qas[:limit]

    chunks_by_id = {
        int((chunk.get("metadata", {}) or {}).get("passage_id")): chunk
        for chunk in chunks
        if "passage_id" in (chunk.get("metadata", {}) or {})
    }

    terag = _load_terag(config, chunks)

    retrieval_config = config["retrieval"]
    method = retrieval_config.get("method", "ppr")
    top_k = top_k_override or int(retrieval_config.get("top_k", 10))

    questions = []
    for eq in eval_qas:
        started = time.time()
        query_entities = terag.query_ner.extract_query_entities(eq["question"], verbose=False)
        retrieved, metrics = terag.retrieve(eq["question"], method=method, top_k=top_k, verbose=False)
        retrieval_time_s = round(time.time() - started, 6)

        supporting_ids = set(int(pid) for pid in eq.get("supporting_passage_ids", []))
        expected_support = [
            _chunk_record(chunks_by_id[pid])
            for pid in eq.get("supporting_passage_ids", [])
            if int(pid) in chunks_by_id
        ]

        retrieved_records = [
            _retrieved_record(result, chunks_by_id, supporting_ids)
            for result in retrieved
        ]
        for rank, record in enumerate(retrieved_records, 1):
            record["rank"] = rank

        found_ids = {
            record["passage_id"]
            for record in retrieved_records
            if record["is_expected_support"] and record["passage_id"] is not None
        }

        questions.append(
            {
                "id": eq.get("id"),
                "question": eq["question"],
                "answer": eq.get("answer"),
                "type": eq.get("type"),
                "level": eq.get("level"),
                "query_entities": query_entities,
                "retrieval_time_s": retrieval_time_s,
                "metrics": {
                    "num_query_entities": getattr(metrics, "num_query_entities", None),
                    "num_matched_concepts": getattr(metrics, "num_matched_concepts", None),
                    "num_results": getattr(metrics, "num_results", None),
                },
                "support_expected_count": len(supporting_ids),
                "support_found_count": len(found_ids),
                "expected_support": expected_support,
                "retrieved": retrieved_records,
            }
        )

    return {
        "run": {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "top_k": top_k,
            "config_path": config["_config_path"],
        },
        "sample": {
            "sample_name": config["sample_name"],
            "manifest_sha256": manifest.get("manifest_sha256"),
            "passage_count": len(chunks),
            "qa_count": len(eval_qas),
            "chunks_path": repo_relative(chunks_path),
            "eval_path": repo_relative(eval_path),
        },
        "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to benchmark config JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to inspect.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    report = inspect(config, limit=args.limit, top_k_override=args.top_k)

    results_dir = resolve_repo_path(config["results_dir"])
    sample_name = config["sample_name"]
    json_path = results_dir / f"latest_{sample_name}_inspection.json"
    md_path = results_dir / f"latest_{sample_name}_inspection.md"
    write_json(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_format_markdown(report), encoding="utf-8")

    print("HotPotQA retrieval inspection complete")
    print(f"  sample: {sample_name}")
    print(f"  questions: {len(report['questions'])}")
    print(f"  markdown: {repo_relative(md_path)}")
    print(f"  json: {repo_relative(json_path)}")


if __name__ == "__main__":
    main()
