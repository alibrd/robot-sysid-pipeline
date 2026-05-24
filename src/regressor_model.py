"""First-class regressor model used internally by the pipeline.

The public contract is always the full rigid-body inertial parameter order
per joint/link:

    [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

Friction parameters, when configured, are appended after all rigid columns.

External-use callable + parameter export is produced by
:mod:`src.regressor_export`; this module only retains the in-memory model
that pipeline stages 4-11 evaluate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics_euler_lagrange import euler_lagrange_regressor_builder
from .dynamics_newton_euler import newton_euler_regressor
from .friction import (
    build_friction_regressor,
    friction_param_count,
    friction_param_names,
)
from .kinematics import RobotKinematics
from .math_utils import GRAVITY
from .urdf_parser import RobotDescription, parse_urdf


RIGID_PARAMETER_BASIS = [
    "m",
    "mx",
    "my",
    "mz",
    "Ixx",
    "Ixy",
    "Ixz",
    "Iyy",
    "Iyz",
    "Izz",
]
VALID_BACKENDS = {"newton_euler", "euler_lagrange"}
REGRESSOR_SCHEMA_VERSION = 1


@dataclass
class RegressorModel:
    """Robot regressor model with rigid and friction-augmented variants."""

    robot: RobotDescription
    kin: RobotKinematics
    backend: str = "newton_euler"
    friction_model: str = "none"
    urdf_path: str | None = None
    cache_dir: str | None = None
    el_kept_cols: list[int] | None = None
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    _el_reduced_fn: Any = field(default=None, repr=False)

    @classmethod
    def from_urdf(cls,
                  urdf_path: str | Path,
                  friction_model: str = "none",
                  backend: str = "newton_euler",
                  cache_dir: str | Path | None = None) -> "RegressorModel":
        """Build a regressor model from a URDF or XACRO path."""
        robot = parse_urdf(str(urdf_path))
        kin = RobotKinematics(robot)
        return cls.from_robot(
            robot,
            kin,
            urdf_path=urdf_path,
            friction_model=friction_model,
            backend=backend,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_robot(cls,
                   robot: RobotDescription,
                   kin: RobotKinematics,
                   *,
                   urdf_path: str | Path | None = None,
                   friction_model: str = "none",
                   backend: str = "newton_euler",
                   cache_dir: str | Path | None = None) -> "RegressorModel":
        """Build a regressor model from already-parsed robot/kinematics data."""
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"Unknown regressor backend {backend!r}; expected one of "
                f"{sorted(VALID_BACKENDS)}."
            )

        cache_dir_str = None if cache_dir is None else str(cache_dir)
        el_kept_cols = None
        el_reduced_fn = None
        if backend == "euler_lagrange":
            if cache_dir is None:
                base = Path(urdf_path).resolve().parent if urdf_path else Path.cwd()
                cache_dir = base / ".regressor_el_cache"
            el_reduced_fn, el_kept_cols = euler_lagrange_regressor_builder(
                kin, str(cache_dir)
            )
            cache_dir_str = str(cache_dir)

        return cls(
            robot=robot,
            kin=kin,
            backend=backend,
            friction_model=friction_model,
            urdf_path=None if urdf_path is None else str(urdf_path),
            cache_dir=cache_dir_str,
            el_kept_cols=None if el_kept_cols is None else [int(c) for c in el_kept_cols],
            _el_reduced_fn=el_reduced_fn,
        )

    @property
    def nDoF(self) -> int:
        return int(self.kin.nDoF)

    @property
    def n_rigid_params(self) -> int:
        return 10 * self.nDoF

    @property
    def n_friction_params(self) -> int:
        return friction_param_count(self.nDoF, self.friction_model)

    @property
    def n_augmented_params(self) -> int:
        return self.n_rigid_params + self.n_friction_params

    def rigid(self, q, dq, ddq) -> np.ndarray:
        """Return the rigid-body regressor Y_rigid(q, dq, ddq)."""
        q_v, dq_v, ddq_v = self._single_state(q, dq, ddq)
        if self.backend == "newton_euler":
            return newton_euler_regressor(self.kin, q_v, dq_v, ddq_v)

        if self._el_reduced_fn is None or self.el_kept_cols is None:
            self._el_reduced_fn, kept_cols = euler_lagrange_regressor_builder(
                self.kin, self._el_cache_dir()
            )
            self.el_kept_cols = [int(c) for c in kept_cols]
        reduced_fn = self._el_reduced_fn
        kept_cols = self.el_kept_cols
        Y_reduced = np.asarray(reduced_fn(q_v, dq_v, ddq_v), dtype=float)
        Y_full = np.zeros((self.nDoF, self.n_rigid_params), dtype=float)
        Y_full[:, np.asarray(kept_cols, dtype=int)] = Y_reduced
        return Y_full

    def augmented(self, q, dq, ddq) -> np.ndarray:
        """Return [Y_rigid | Y_friction] for the configured friction model."""
        Y = self.rigid(q, dq, ddq)
        if self.friction_model == "none":
            return Y
        _, dq_v, _ = self._single_state(q, dq, ddq)
        Yf = build_friction_regressor(dq_v, self.friction_model)
        return np.hstack((Y, Yf))

    def stack(self, q, dq, ddq, include_friction: bool = False) -> np.ndarray:
        """Stack per-sample regressors vertically into an observation matrix."""
        q_s = self._state_samples(q, "q")
        dq_s = self._state_samples(dq, "dq")
        ddq_s = self._state_samples(ddq, "ddq")
        if q_s.shape != dq_s.shape or q_s.shape != ddq_s.shape:
            raise ValueError(
                "q, dq, and ddq must describe the same number of samples "
                f"and joints; got {q_s.shape}, {dq_s.shape}, {ddq_s.shape}."
            )
        fn = self.augmented if include_friction else self.rigid
        return np.vstack([
            fn(q_s[k], dq_s[k], ddq_s[k])
            for k in range(q_s.shape[0])
        ])

    def joint_names(self) -> list[str]:
        return list(self.robot.revolute_joint_names)

    def link_names(self) -> list[str]:
        return list(self.kin.link_names)

    def rigid_parameter_names(self) -> list[str]:
        names = []
        for link_name in self.link_names():
            names.extend(
                f"{link_name}.{basis}" for basis in RIGID_PARAMETER_BASIS
            )
        return names

    def friction_parameter_names(self) -> list[str]:
        return friction_param_names(self.nDoF, self.friction_model)

    def augmented_parameter_names(self) -> list[str]:
        return self.rigid_parameter_names() + self.friction_parameter_names()

    def metadata_dict(self,
                      *,
                      urdf_path: str | Path | None = None,
                      artifact_dir: str | Path | None = None) -> dict[str, Any]:
        """Return JSON-serializable regressor metadata."""
        urdf_ref = urdf_path if urdf_path is not None else self.urdf_path
        if urdf_ref is not None and artifact_dir is not None:
            try:
                urdf_ref = Path(urdf_ref).resolve().relative_to(
                    Path(artifact_dir).resolve()
                )
            except ValueError:
                pass

        return {
            "schema_version": REGRESSOR_SCHEMA_VERSION,
            "backend": self.backend,
            "method": self.backend,
            "friction_model": self.friction_model,
            "urdf_path": None if urdf_ref is None else str(urdf_ref),
            "gravity": np.asarray(GRAVITY, dtype=float).tolist(),
            "nDoF": self.nDoF,
            "joint_names": self.joint_names(),
            "link_names": self.link_names(),
            "rigid_parameter_order": list(RIGID_PARAMETER_BASIS),
            "rigid_parameter_names": self.rigid_parameter_names(),
            "friction_parameter_names": self.friction_parameter_names(),
            "augmented_parameter_names": self.augmented_parameter_names(),
            "n_rigid_params": self.n_rigid_params,
            "n_friction_params": self.n_friction_params,
            "n_augmented_params": self.n_augmented_params,
            "el_kept_cols": self.el_kept_cols,
        }

    def _el_cache_dir(self) -> str:
        if self.cache_dir is not None:
            return self.cache_dir
        if self.urdf_path is not None:
            return str(Path(self.urdf_path).resolve().parent / ".regressor_el_cache")
        return str(Path.cwd() / ".regressor_el_cache")

    def _single_state(self, q, dq, ddq) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_v = np.asarray(q, dtype=float).reshape(-1)
        dq_v = np.asarray(dq, dtype=float).reshape(-1)
        ddq_v = np.asarray(ddq, dtype=float).reshape(-1)
        for name, value in (("q", q_v), ("dq", dq_v), ("ddq", ddq_v)):
            if value.size != self.nDoF:
                raise ValueError(
                    f"{name} must have {self.nDoF} entries in pipeline joint "
                    f"order, got {value.size}."
                )
        return q_v, dq_v, ddq_v

    def _state_samples(self, value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            if arr.size != self.nDoF:
                raise ValueError(
                    f"{name} must have {self.nDoF} entries, got {arr.size}."
                )
            return arr.reshape(1, self.nDoF)
        if arr.ndim == 2:
            if arr.shape[1] == self.nDoF:
                return arr
            if arr.shape[0] == self.nDoF:
                return arr.T
        raise ValueError(
            f"{name} must be shape ({self.nDoF},), (N, {self.nDoF}), "
            f"or ({self.nDoF}, N); got {arr.shape}."
        )


