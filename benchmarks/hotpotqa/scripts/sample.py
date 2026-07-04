"""Create deterministic HotPotQA benchmark subsets."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import (
    load_config,
    read_json,
    repo_relative,
    resolve_repo_path,
    sample_dir,
    sha256_file,
    sha256_json,
    write_json,
)


def _stratified_sample(
    eval_qas: List[Dict[str, Any]],
    target_qa_count: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if target_qa_count is None or target_qa_count >= len(eval_qas):
        return list(eval_qas)

    rng = random.Random(seed)
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for eq in eval_qas:
        groups[(eq.get("type", "unknown"), eq.get("level", "unknown"))].append(eq)

    total = len(eval_qas)
    allocations = {}
    fractional = []
    for key, items in groups.items():
        raw = target_qa_count * (len(items) / total)
        base = min(len(items), int(raw))
        allocations[key] = base
        fractional.append((raw - base, key))

    remaining = target_qa_count - sum(allocations.values())
    for _, key in sorted(fractional, reverse=True):
        if remaining <= 0:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            remaining -= 1

    sampled = []
    for key in sorted(groups):
        items = list(groups[key])
        n = allocations[key]
        sampled.extend(rng.sample(items, n))

    rng.shuffle(sampled)
    return sampled


def build_sample(config: Dict[str, Any]) -> Dict[str, Any]:
    source_dir = resolve_repo_path(config["source_data_dir"])
    chunks_path = source_dir / "chunks.json"
    eval_path = source_dir / "eval_qa.json"
    if not chunks_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            "Expected HotPotQA source files at "
            f"{chunks_path} and {eval_path}. Run the existing dataset preparation first."
        )

    chunks = read_json(chunks_path)
    eval_qas = read_json(eval_path)
    seed = int(config["seed"])
    target_qa_count = config.get("target_qa_count")
    target_passage_count = config.get("target_passage_count")

    sample_qas = _stratified_sample(eval_qas, target_qa_count, seed)
    rng = random.Random(seed)

    passage_by_old_id = {}
    for i, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        old_id = metadata.get("passage_id", i)
        passage_by_old_id[int(old_id)] = chunk

    supporting_ids = set()
    for eq in sample_qas:
        supporting_ids.update(int(pid) for pid in eq.get("supporting_passage_ids", []))

    missing_support = sorted(pid for pid in supporting_ids if pid not in passage_by_old_id)
    if missing_support:
        raise ValueError(f"Sample references {len(missing_support)} missing supporting passages.")

    corpus_old_ids = sorted(supporting_ids)
    if target_passage_count is None:
        target_passage_count = len(chunks)

    remaining_ids = sorted(set(passage_by_old_id) - set(corpus_old_ids))
    distractor_count = max(0, min(int(target_passage_count) - len(corpus_old_ids), len(remaining_ids)))
    corpus_old_ids.extend(rng.sample(remaining_ids, distractor_count))
    corpus_old_ids = sorted(corpus_old_ids)

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(corpus_old_ids)}
    new_chunks = []
    for new_id, old_id in enumerate(corpus_old_ids):
        chunk = passage_by_old_id[old_id]
        metadata = dict(chunk.get("metadata", {}))
        metadata["original_passage_id"] = old_id
        metadata["passage_id"] = new_id
        new_chunks.append({"content": chunk["content"], "metadata": metadata})

    new_evals = []
    for eq in sample_qas:
        new_supporting = [
            old_to_new[int(pid)]
            for pid in eq.get("supporting_passage_ids", [])
            if int(pid) in old_to_new
        ]
        new_evals.append(
            {
                "id": eq["id"],
                "question": eq["question"],
                "answer": eq.get("answer"),
                "type": eq.get("type"),
                "level": eq.get("level"),
                "supporting_passage_ids": new_supporting,
                "supporting_passage_titles": eq.get("supporting_passage_titles", []),
            }
        )

    zero_support = [eq["id"] for eq in new_evals if not eq["supporting_passage_ids"]]
    if zero_support:
        raise ValueError(f"{len(zero_support)} sampled QA rows have no supporting passages.")

    manifest = {
        "dataset": config["dataset"],
        "split": config["split"],
        "sample_name": config["sample_name"],
        "seed": seed,
        "strategy": "stratified_by_type_level_keep_gold_supports_add_random_distractors",
        "target_qa_count": config.get("target_qa_count"),
        "target_passage_count": config.get("target_passage_count"),
        "source": {
            "chunks_path": repo_relative(chunks_path),
            "eval_path": repo_relative(eval_path),
            "chunks_sha256": sha256_file(chunks_path),
            "eval_sha256": sha256_file(eval_path),
            "full_passage_count": len(chunks),
            "full_qa_count": len(eval_qas),
        },
        "actual": {
            "passage_count": len(new_chunks),
            "qa_count": len(new_evals),
            "supporting_passage_count": len(supporting_ids),
            "distractor_count": distractor_count,
        },
        "qa_ids": [eq["id"] for eq in new_evals],
        "old_passage_ids": corpus_old_ids,
    }
    manifest["manifest_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})

    out_dir = sample_dir(config)
    write_json(out_dir / "chunks.json", new_chunks)
    write_json(out_dir / "eval_qa.json", new_evals)
    write_json(out_dir / "manifest.json", manifest)
    manifest_copy_path = config.get("manifest_copy_path")
    if manifest_copy_path:
        write_json(resolve_repo_path(manifest_copy_path), manifest)

    return {
        "out_dir": str(out_dir),
        "chunks": len(new_chunks),
        "eval_qas": len(new_evals),
        "supporting_passages": len(supporting_ids),
        "distractors": distractor_count,
        "manifest_sha256": manifest["manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to benchmark config JSON.")
    args = parser.parse_args()

    summary = build_sample(load_config(Path(args.config)))
    print("HotPotQA sample ready")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
