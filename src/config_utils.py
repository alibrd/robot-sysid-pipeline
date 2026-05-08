"""Shared configuration utilities used by config_loader, pybullet_validation, and runner."""
from copy import deepcopy
from pathlib import Path


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two JSON-like dictionaries (deep-copy safe)."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_path_value(value, base_dir: Path):
    """Resolve a path string against *base_dir* while leaving null/empty untouched."""
    if value in (None, ""):
        return value

    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())
