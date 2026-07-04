"""Shared helpers for HotPotQA benchmark scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def resolve_repo_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    path = resolve_repo_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(path)
    return config


def sample_dir(config: Dict[str, Any]) -> Path:
    return resolve_repo_path(config["output_root"]) / config["sample_name"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
