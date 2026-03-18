"""Common utility helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries with override precedence."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_extends(path: Path) -> dict[str, Any]:
    """Load YAML config, resolving a single-chain `extends` directive."""

    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / parent).resolve()
    base = load_yaml_with_extends(parent_path)
    return deep_merge(base, raw)


def stable_hash(value: str) -> str:
    """Return a deterministic SHA256 hex digest."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash_bytes(value: bytes) -> str:
    """Return a deterministic SHA256 hex digest for raw bytes."""

    return hashlib.sha256(value).hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def load_json(path: Path) -> Any:
    """Load JSON from disk."""

    return json.loads(path.read_text())


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for supported libraries."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
