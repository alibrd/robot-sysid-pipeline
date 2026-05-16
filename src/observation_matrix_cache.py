"""Observation-matrix cache artifact helpers.

The cache stores the expensive Stage 8/9 products only: the stacked
observation matrix, the base observation matrix, and the regrouping metadata.
It intentionally does not store identified parameters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_FILENAME = "observation_matrix_cache.npz"


def resolve_cache_load_path(load_from: str | Path,
                            filename: str = DEFAULT_CACHE_FILENAME) -> Path:
    """Resolve a cache reference to an existing .npz file.

    ``load_from`` may be a direct .npz path, a pipeline output directory, or a
    unified output root containing a ``pipeline/`` subdirectory.
    """
    base = Path(load_from)
    if base.suffix.lower() == ".npz":
        candidates = [base]
    else:
        candidates = [base / filename, base / "pipeline" / filename]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Observation matrix cache not found. Checked: {checked}"
    )


def metadata_json(metadata: dict[str, Any]) -> str:
    """Serialize metadata deterministically."""
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def array_fingerprint(array: np.ndarray) -> str:
    """Return a stable SHA-256 fingerprint for an array's shape, dtype, and data."""
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def file_fingerprint(path: str | Path) -> str | None:
    """Return the SHA-256 fingerprint for a file, or None when it is absent."""
    if path in (None, ""):
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cache_metadata(*,
                         cfg: dict,
                         n_dof: int,
                         pi_full: np.ndarray,
                         el_kept_cols,
                         samples: dict,
                         W: np.ndarray | None = None,
                         W_base: np.ndarray | None = None,
                         P_mat: np.ndarray | None = None,
                         kept_cols=None,
                         rank: int | None = None,
                         source_cache_path: str | None = None,
                         load_status: str | None = None,
                         load_mismatches: list[str] | None = None) -> dict:
    """Build metadata used to validate and describe a cache artifact."""
    q_used = np.asarray(samples["q"])
    dq_used = np.asarray(samples["dq"])
    ddq_used = np.asarray(samples["ddq"])
    tau_used = np.asarray(samples["tau"])
    tau_vec = np.asarray(samples["tau_vec"])
    sample_indices = np.asarray(samples["sample_indices"], dtype=np.int64)

    full_parameter_count = int(len(pi_full))
    equation_count = int(tau_vec.size)
    matrix_shapes = {
        "W": (
            list(W.shape) if W is not None
            else [equation_count, full_parameter_count]
        ),
        "W_base": list(W_base.shape) if W_base is not None else None,
        "P_matrix": list(P_mat.shape) if P_mat is not None else None,
    }

    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": cfg["method"],
        "friction_model": cfg["friction"]["model"],
        "nDoF": int(n_dof),
        "full_parameter_count": full_parameter_count,
        "rank": None if rank is None else int(rank),
        "sample_count": int(sample_indices.size),
        "equation_count": equation_count,
        "sample_step": int(samples["step"]),
        "filtering": cfg.get("filtering", {}),
        "downsampling": cfg.get("downsampling", {}),
        "urdf_path": str(Path(cfg["urdf_path"]).resolve()),
        "urdf_sha256": file_fingerprint(cfg["urdf_path"]),
        "data_file": cfg.get("identification", {}).get("data_file"),
        "nominal_parameter_fingerprint": array_fingerprint(np.asarray(pi_full)),
        "el_kept_cols": (
            None if el_kept_cols is None
            else np.asarray(el_kept_cols, dtype=np.int64).tolist()
        ),
        "kept_cols": None if kept_cols is None else [int(c) for c in kept_cols],
        "matrix_shapes": matrix_shapes,
        "fingerprints": {
            "q": array_fingerprint(q_used),
            "dq": array_fingerprint(dq_used),
            "ddq": array_fingerprint(ddq_used),
            "tau": array_fingerprint(tau_used),
            "tau_vec": array_fingerprint(tau_vec),
            "sample_indices": array_fingerprint(sample_indices),
        },
    }
    if source_cache_path is not None:
        metadata["source_cache_path"] = str(source_cache_path)
    if load_status is not None:
        metadata["load_status"] = load_status
    if load_mismatches:
        metadata["load_mismatches"] = list(load_mismatches)
    return metadata


def save_observation_matrix_cache(path: str | Path,
                                  *,
                                  W: np.ndarray,
                                  W_base: np.ndarray,
                                  P_mat: np.ndarray,
                                  kept_cols,
                                  rank: int,
                                  tau_vec: np.ndarray,
                                  samples: dict,
                                  metadata: dict) -> Path:
    """Write an observation-matrix cache artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        W=np.asarray(W, dtype=float),
        W_base=np.asarray(W_base, dtype=float),
        P_matrix=np.asarray(P_mat, dtype=float),
        kept_cols=np.asarray(kept_cols, dtype=np.int64),
        rank=np.int64(rank),
        tau_vec=np.asarray(tau_vec, dtype=float),
        sample_indices=np.asarray(samples["sample_indices"], dtype=np.int64),
        q_used=np.asarray(samples["q"], dtype=float),
        dq_used=np.asarray(samples["dq"], dtype=float),
        ddq_used=np.asarray(samples["ddq"], dtype=float),
        tau_used=np.asarray(samples["tau"], dtype=float),
        metadata_json=np.asarray(metadata_json(metadata)),
    )
    return out_path


def load_observation_matrix_cache(path: str | Path) -> dict:
    """Load a cache artifact and validate its internal array contract."""
    cache_path = Path(path)
    with np.load(str(cache_path), allow_pickle=True) as data:
        required = {
            "W", "W_base", "P_matrix", "kept_cols", "rank", "metadata_json"
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Observation matrix cache is missing keys: {sorted(missing)}"
            )
        raw_meta = data["metadata_json"]
        metadata = json.loads(str(raw_meta.item() if raw_meta.ndim == 0 else raw_meta))
        cache = {
            "path": str(cache_path),
            "W": np.asarray(data["W"], dtype=float),
            "W_base": np.asarray(data["W_base"], dtype=float),
            "P_mat": np.asarray(data["P_matrix"], dtype=float),
            "kept_cols": np.asarray(data["kept_cols"], dtype=np.int64),
            "rank": int(data["rank"]),
            "metadata": metadata,
        }
        if {
            "tau_vec", "sample_indices", "q_used", "dq_used", "ddq_used", "tau_used"
        }.issubset(set(data.files)):
            cache["tau_vec"] = np.asarray(data["tau_vec"], dtype=float)
            cache["samples"] = {
                "q": np.asarray(data["q_used"], dtype=float),
                "dq": np.asarray(data["dq_used"], dtype=float),
                "ddq": np.asarray(data["ddq_used"], dtype=float),
                "tau": np.asarray(data["tau_used"], dtype=float),
                "tau_vec": np.asarray(data["tau_vec"], dtype=float),
                "sample_indices": np.asarray(data["sample_indices"], dtype=np.int64),
                "step": int(metadata.get("sample_step", 1)),
                "nDoF": int(metadata.get("nDoF", 0)),
            }
    _validate_internal_shapes(cache)
    return cache


def validate_cache_for_run(cache: dict,
                           expected_metadata: dict,
                           *,
                           force_load: bool = False) -> list[str]:
    """Validate cache compatibility for the current run.

    Returns a list of metadata mismatches. Shape mismatches always raise because
    they make the loaded matrices unusable even in forced mode.
    """
    W = cache["W"]
    W_base = cache["W_base"]
    P_mat = cache["P_mat"]
    rank = int(cache["rank"])
    expected_rows = int(expected_metadata["equation_count"])
    expected_cols = int(expected_metadata["full_parameter_count"])

    shape_errors = []
    if W.shape != (expected_rows, expected_cols):
        shape_errors.append(
            f"W shape {W.shape} != expected {(expected_rows, expected_cols)}"
        )
    if W_base.shape[0] != expected_rows:
        shape_errors.append(
            f"W_base row count {W_base.shape[0]} != expected {expected_rows}"
        )
    if W_base.shape[1] != rank:
        shape_errors.append(
            f"W_base column count {W_base.shape[1]} != rank {rank}"
        )
    if P_mat.shape != (rank, expected_cols):
        shape_errors.append(
            f"P_matrix shape {P_mat.shape} != expected {(rank, expected_cols)}"
        )
    if shape_errors:
        raise ValueError(
            "Observation matrix cache has incompatible dimensions: "
            + "; ".join(shape_errors)
        )

    metadata = cache["metadata"]
    mismatches = _metadata_mismatches(metadata, expected_metadata)
    if mismatches and not force_load:
        raise ValueError(
            "Observation matrix cache metadata does not match this run: "
            + "; ".join(mismatches)
        )
    return mismatches


def _validate_internal_shapes(cache: dict) -> None:
    W = cache["W"]
    W_base = cache["W_base"]
    P_mat = cache["P_mat"]
    kept_cols = cache["kept_cols"]
    rank = int(cache["rank"])

    if W.ndim != 2 or W_base.ndim != 2 or P_mat.ndim != 2:
        raise ValueError("Cached W, W_base, and P_matrix must be 2-D arrays.")
    if rank <= 0:
        raise ValueError("Cached rank must be positive.")
    if len(kept_cols) != rank:
        raise ValueError("Cached kept_cols length must equal rank.")
    if W_base.shape != (W.shape[0], rank):
        raise ValueError("Cached W_base shape is inconsistent with W/rank.")
    if P_mat.shape != (rank, W.shape[1]):
        raise ValueError("Cached P_matrix shape is inconsistent with W/rank.")
    if np.any(kept_cols < 0) or np.any(kept_cols >= W.shape[1]):
        raise ValueError("Cached kept_cols contains out-of-range columns.")
    if not np.allclose(W[:, kept_cols], W_base, atol=1e-10, rtol=1e-10):
        raise ValueError("Cached W_base does not match W[:, kept_cols].")


def _metadata_mismatches(metadata: dict, expected: dict) -> list[str]:
    mismatches = []
    keys = [
        "schema_version",
        "method",
        "friction_model",
        "nDoF",
        "full_parameter_count",
        "sample_count",
        "equation_count",
        "sample_step",
        "filtering",
        "downsampling",
        "urdf_sha256",
        "nominal_parameter_fingerprint",
        "el_kept_cols",
    ]
    for key in keys:
        if metadata.get(key) != expected.get(key):
            mismatches.append(f"{key}: cache={metadata.get(key)!r}, current={expected.get(key)!r}")

    meta_fp = metadata.get("fingerprints", {}) or {}
    expected_fp = expected.get("fingerprints", {}) or {}
    for key in ("q", "dq", "ddq", "tau", "tau_vec", "sample_indices"):
        if meta_fp.get(key) != expected_fp.get(key):
            mismatches.append(f"fingerprints.{key}")
    return mismatches
