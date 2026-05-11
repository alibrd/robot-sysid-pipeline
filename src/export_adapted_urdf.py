"""Write the identified parameters back into an adapted URDF + friction sidecar.

This is a thin CLI wrapper around :func:`src.urdf_exporter.export_adapted_urdf`,
kept for users who want to re-export the adapted URDF from a saved
``identification_results.npz`` without re-running the pipeline.

The pipeline itself can emit the adapted URDF directly as Stage 12 by setting

.. code-block:: json

    "export": {
        "enabled": true,
        "urdf_filename": "adapted_robot.urdf",
        "friction_sidecar": true,
        "friction_sidecar_filename": "adapted_friction.json"
    }

in the unified config.  This script remains as the offline alternative:

Usage
-----
    python src/export_adapted_urdf.py \\
        --in-urdf  tests/assets/FrankaPanda_7DoF_with_hand.urdf \\
        --in-npz   tmp_output/.../identification_results.npz \\
        --out-urdf tmp_output/.../adapted_panda.urdf \\
        --out-friction-json tmp_output/.../adapted_friction.json

Notes
-----
* Reads ``pi_corrected`` from the npz when present, falling back to
  ``pi_identified``.  Both are length ``10*nDoF + n_friction``, where
  ``n_friction`` is 0, n, 2*n, or 3*n for the
  ``{none, viscous, coulomb, viscous_coulomb}`` models respectively.
* All adaptation logic lives in :mod:`src.urdf_exporter`; see that module
  for the full 10-vector → URDF inertial conversion and the friction
  sidecar JSON contract.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable so `from src.urdf_exporter import ...` works
# when this file is invoked directly via `python src/export_adapted_urdf.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.urdf_exporter import export_adapted_urdf  # noqa: E402


def export_adapted(
    in_urdf: Path,
    in_npz: Path,
    out_urdf: Path,
    out_friction_json: Path | None,
):
    """Load the npz and forward to :func:`src.urdf_exporter.export_adapted_urdf`."""
    with np.load(in_npz, allow_pickle=True) as data:
        if "pi_corrected" in data.files:
            pi_full = np.asarray(data["pi_corrected"], float).reshape(-1)
            param_source = "pi_corrected"
        elif "pi_identified" in data.files:
            pi_full = np.asarray(data["pi_identified"], float).reshape(-1)
            param_source = "pi_identified"
        else:
            raise ValueError(
                f"{in_npz} does not contain pi_corrected or pi_identified"
            )
        n_dof = int(np.asarray(data["nDoF"]).item())
        friction_model = (
            str(np.asarray(data["friction_model"]).item())
            if "friction_model" in data.files
            else "none"
        )

    meta = export_adapted_urdf(
        input_urdf_path=in_urdf,
        pi_full=pi_full,
        n_dof=n_dof,
        friction_model=friction_model,
        output_urdf_path=out_urdf,
        friction_sidecar_path=out_friction_json,
        parameter_source=param_source,
    )
    print(f"wrote adapted URDF: {meta['adapted_urdf_path']}")
    if meta["friction_sidecar_path"] is not None:
        print(f"wrote friction sidecar: {meta['friction_sidecar_path']}")
    return meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-urdf", required=True, type=Path)
    p.add_argument("--in-npz", required=True, type=Path)
    p.add_argument("--out-urdf", required=True, type=Path)
    p.add_argument("--out-friction-json", type=Path, default=None)
    args = p.parse_args()
    export_adapted(args.in_urdf, args.in_npz, args.out_urdf, args.out_friction_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
