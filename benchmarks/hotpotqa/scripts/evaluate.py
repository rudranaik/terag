"""Evaluate TERAG retrieval on a prepared HotPotQA sample."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import load_config, read_json, repo_root, resolve_repo_path, sample_dir, write_json


def _result_passage_id(result: Any) -> Optional[int]:
    metadata = getattr(result, "metadata", {}) or {}
    if "passage_id" in metadata:
        return int(metadata["passage_id"])

    passage_id = str(getattr(result, "passage_id", ""))
    if passage_id.startswith("passage_"):
        return int(passage_id.replace("passage_", "", 1))
    try:
        return int(passage_id)
    except ValueError:
        return None


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((percentile / 100) * (len(ordered) - 1))))
    return ordered[idx]


def _format_markdown(results: Dict[str, Any]) -> str:
    hit_rate = results["metrics"]["hit_rate"]
    hit_cols = " | ".join(f"@{k}" for k in sorted(hit_rate, key=int))
    hit_vals = " | ".join(f"{hit_rate[k]:.4f}" for k in sorted(hit_rate, key=int))
    return f"""# HotPotQA TERAG Benchmark

**Date:** {results['run']['finished_at']}
**Sample:** {results['sample']['sample_name']}
**Method:** {results['run']['method']}
**Queries:** {results['sample']['qa_count']}
**Passages:** {results['sample']['passage_count']}

## Retrieval Metrics

| {hit_cols} | MRR | Full Support @10 |
| {' | '.join(['---'] * (len(hit_rate) + 2))} |
| {hit_vals} | {results['metrics']['mrr']:.4f} | {results['metrics']['full_support_rate_at_10']:.4f} |

## Runtime

- Build time: {results['runtime']['build_time_s']:.3f}s
- Avg query time: {results['runtime']['avg_query_time_s']:.4f}s
- P50 query time: {results['runtime']['p50_query_time_s']:.4f}s
- P95 query time: {results['runtime']['p95_query_time_s']:.4f}s
- Errors: {results['metrics']['errors']}
- No-entity queries: {results['metrics']['no_entity_queries']}

## Graph

- Passages: {results['graph']['passages']}
- Concepts: {results['graph']['concepts']}
- Edges: {results['graph']['edges']}
"""


def _prewarm_extraction_cache(chunks: List[Dict[str, Any]], config_kwargs: Dict[str, Any], workers: int) -> Dict[str, Any]:
    if workers <= 0:
        workers = 1

    from terag.ingestion.ner_extractor import NERExtractor

    extractor = NERExtractor(
        cache_dir=config_kwargs.get("extraction_cache_dir", "extraction_cache"),
        use_llm=config_kwargs.get("use_llm_for_ner", False),
        provider=config_kwargs.get("llm_provider", "groq"),
        api_key=config_kwargs.get("llm_api_key"),
        model=config_kwargs.get("llm_model"),
        enable_progress_reporting=False,
    )

    uncached = [
        chunk.get("content", "")
        for chunk in chunks
        if chunk.get("content", "") and not extractor.cache.is_passage_cached(chunk.get("content", ""))
    ]
    if not uncached:
        return {"requested": len(chunks), "uncached": 0, "workers": workers, "time_s": 0.0}

    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extractor.extract_entities_and_concepts, content) for content in uncached]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0:
                print(f"  prewarmed {completed}/{len(uncached)} extractions", flush=True)

    return {
        "requested": len(chunks),
        "uncached": len(uncached),
        "workers": workers,
        "time_s": round(time.time() - started, 6),
    }


def _install_query_ner_cache(terag: Any, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    original = terag.query_ner.extract_query_entities

    def cached_extract(query: str, verbose: bool = False) -> List[str]:
        if query in cache:
            return cache[query]
        entities = original(query, verbose=verbose)
        cache[query] = entities
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return entities

    terag.query_ner.extract_query_entities = cached_extract


def evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
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

    prewarm_stats = None
    if performance_config.get("prewarm_extraction_cache"):
        prewarm_stats = _prewarm_extraction_cache(
            chunks,
            config_kwargs,
            workers=int(performance_config.get("extraction_workers", 4)),
        )

    t0 = time.time()
    terag = TERAG.from_chunks(
        chunks,
        config=TERAGConfig(**config_kwargs),
        embedding_model=embedding_model,
        verbose=False,
    )
    build_time_s = time.time() - t0
    stats = terag.graph.get_statistics()

    query_cache_path = performance_config.get("query_cache_path")
    if query_cache_path:
        _install_query_ner_cache(terag, resolve_repo_path(query_cache_path))

    retrieval_config = config["retrieval"]
    method = retrieval_config.get("method", "ppr")
    top_k = int(retrieval_config.get("top_k", 10))
    hit_ks = [int(k) for k in retrieval_config.get("hit_ks", [1, 3, 5, 10])]

    hits = {k: 0 for k in hit_ks}
    full_support_hits = {k: 0 for k in hit_ks}
    reciprocal_ranks = []
    support_coverage = {k: [] for k in hit_ks}
    by_type = defaultdict(lambda: {k: [0, 0] for k in hit_ks})
    query_times = []
    no_entity = 0
    errors = 0

    for eq in eval_qas:
        try:
            t0 = time.time()
            retrieved, metrics = terag.retrieve(eq["question"], method=method, top_k=top_k, verbose=False)
            query_times.append(time.time() - t0)
            if getattr(metrics, "num_query_entities", 0) == 0:
                no_entity += 1

            supporting = set(int(pid) for pid in eq.get("supporting_passage_ids", []))
            ranked_ids = [_result_passage_id(result) for result in retrieved]
            ranked_ids = [pid for pid in ranked_ids if pid is not None]

            best_rank = None
            for rank, pid in enumerate(ranked_ids, 1):
                if pid in supporting:
                    best_rank = rank
                    break
            if best_rank is not None:
                reciprocal_ranks.append(1.0 / best_rank)

            for k in hit_ks:
                top_ids = set(ranked_ids[:k])
                found = top_ids & supporting
                if found:
                    hits[k] += 1
                    by_type[eq.get("type", "unknown")][k][0] += 1
                if supporting and supporting.issubset(top_ids):
                    full_support_hits[k] += 1
                support_coverage[k].append(len(found) / max(len(supporting), 1))
                by_type[eq.get("type", "unknown")][k][1] += 1
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"Error on {eq.get('id')}: {exc}")

    total = len(eval_qas)
    metrics = {
        "total_queries": total,
        "errors": errors,
        "no_entity_queries": no_entity,
        "hit_rate": {str(k): round(hits[k] / total, 6) for k in hit_ks},
        "full_support_rate": {str(k): round(full_support_hits[k] / total, 6) for k in hit_ks},
        "support_coverage": {
            str(k): round(sum(values) / len(values), 6) if values else 0.0
            for k, values in support_coverage.items()
        },
        "mrr": round(sum(reciprocal_ranks) / total, 6) if total else 0.0,
        "by_type": {
            qtype: {str(k): round(vals[k][0] / max(vals[k][1], 1), 6) for k in hit_ks}
            for qtype, vals in by_type.items()
        },
        "full_support_rate_at_10": round(full_support_hits.get(10, 0) / total, 6) if total else 0.0,
    }

    results = {
        "run": {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "top_k": top_k,
            "config_path": config["_config_path"],
            "llm_provider": config_kwargs.get("llm_provider"),
            "llm_model": config_kwargs.get("llm_model"),
            "embedding_model": (embedding_config or {}).get("model"),
            "prewarm_extraction_cache": prewarm_stats,
        },
        "sample": {
            "sample_name": config["sample_name"],
            "manifest_sha256": manifest.get("manifest_sha256"),
            "passage_count": len(chunks),
            "qa_count": len(eval_qas),
        },
        "graph": {
            "passages": stats["num_passages"],
            "concepts": stats["num_concepts"],
            "edges": stats["num_edges"],
        },
        "runtime": {
            "build_time_s": round(build_time_s, 6),
            "avg_query_time_s": round(sum(query_times) / len(query_times), 6) if query_times else 0.0,
            "p50_query_time_s": round(_percentile(query_times, 50), 6),
            "p95_query_time_s": round(_percentile(query_times, 95), 6),
        },
        "metrics": metrics,
    }

    results_dir = resolve_repo_path(config["results_dir"])
    latest_json = results_dir / f"latest_{config['sample_name']}.json"
    latest_md = results_dir / f"latest_{config['sample_name']}.md"
    write_json(latest_json, results)
    latest_md.parent.mkdir(parents=True, exist_ok=True)
    latest_md.write_text(_format_markdown(results), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to benchmark config JSON.")
    args = parser.parse_args()

    results = evaluate(load_config(Path(args.config)))
    print("HotPotQA evaluation complete")
    print(f"  sample: {results['sample']['sample_name']}")
    print(f"  passages: {results['sample']['passage_count']}")
    print(f"  queries: {results['sample']['qa_count']}")
    print(f"  hit@10: {results['metrics']['hit_rate'].get('10', 0):.4f}")
    print(f"  mrr: {results['metrics']['mrr']:.4f}")
    print(f"  avg_query_time_s: {results['runtime']['avg_query_time_s']:.4f}")


if __name__ == "__main__":
    main()
