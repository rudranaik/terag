"""Compare a HotPotQA benchmark run against a saved baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from .common import load_config, read_json, resolve_repo_path


def _metric(results: Dict[str, Any], path: str) -> float:
    node: Any = results
    for part in path.split("."):
        node = node[part]
    return float(node)


def compare(config: Dict[str, Any], current_path: Optional[str], baseline_path: Optional[str]) -> int:
    current = read_json(resolve_repo_path(current_path or f"{config['results_dir']}/latest_{config['sample_name']}.json"))
    baseline = read_json(resolve_repo_path(baseline_path or config["baseline_path"]))
    thresholds = config.get("regression_thresholds", {})

    checks = []
    hit_drop = thresholds.get("hit_rate_at_10_abs_drop")
    if hit_drop is not None:
        checks.append(("Hit@10", "metrics.hit_rate.10", "min_delta", -float(hit_drop)))

    mrr_drop = thresholds.get("mrr_abs_drop")
    if mrr_drop is not None:
        checks.append(("MRR", "metrics.mrr", "min_delta", -float(mrr_drop)))

    avg_query_time_multiplier = thresholds.get("avg_query_time_multiplier")
    if avg_query_time_multiplier is not None:
        checks.append(
            (
                "Avg query time",
                "runtime.avg_query_time_s",
                "max_multiplier",
                float(avg_query_time_multiplier),
            )
        )

    build_time_multiplier = thresholds.get("build_time_multiplier")
    if build_time_multiplier is not None:
        checks.append(("Build time", "runtime.build_time_s", "max_multiplier", float(build_time_multiplier)))

    failures = []
    print("HotPotQA benchmark comparison")
    for label, path, mode, threshold in checks:
        current_value = _metric(current, path)
        baseline_value = _metric(baseline, path)
        if mode == "min_delta":
            delta = current_value - baseline_value
            ok = delta >= threshold
            print(f"  {label}: current={current_value:.6f} baseline={baseline_value:.6f} delta={delta:.6f}")
        else:
            limit = baseline_value * threshold
            ok = current_value <= limit
            print(f"  {label}: current={current_value:.6f} baseline={baseline_value:.6f} limit={limit:.6f}")
        if not ok:
            failures.append(label)

    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1

    print("PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to benchmark config JSON.")
    parser.add_argument("--current", help="Current results JSON. Defaults to latest result for config sample.")
    parser.add_argument("--baseline", help="Baseline JSON. Defaults to config baseline_path.")
    args = parser.parse_args()
    raise SystemExit(compare(load_config(Path(args.config)), args.current, args.baseline))


if __name__ == "__main__":
    main()
