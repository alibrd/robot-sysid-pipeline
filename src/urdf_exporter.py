"""Adapted-URDF export — Stage 12 of Mode 2.

Writes the identified parameter vector back into a *simulation-ready*
URDF: per-link ``<inertial>`` blocks carry the identified mass, centre
of mass, and inertia tensor about the COM, while ``<dynamics>`` tags on
each revolute joint receive the viscous coefficient (``damping``) and
the symmetric average of the asymmetric Coulomb pair (``friction``).

Because the URDF schema cannot represent direction-dependent Coulomb
friction, an optional sidecar JSON is emitted alongside the URDF
containing the full identified ``Fv``/``Fcp``/``Fcn`` triple per joint.

The function is consumed both by:

* the unified pipeline as the optional sub-step of Stage 12
  (``SystemIdentificationPipeline._run_stages_7_to_11`` with
  ``export.enabled = true``), and
* the standalone ``src/export_adapted_urdf.py`` CLI, which loads
  ``identification_results.npz`` and forwards to
  :func:`export_adapted_urdf` so the two entry points share one
  implementation.

The 10-vector convention (Atkeson, An, Hollerbach 1986) is

    pi_i = [ m_i,
             m_i*c_x, m_i*c_y, m_i*c_z,
             I_xx_at_origin, I_xy_at_origin, I_xz_at_origin,
             I_yy_at_origin, I_yz_at_origin, I_zz_at_origin ]

i.e. inertia is given **at the link-frame origin**.  The URDF schema
expects inertia about the COM placed via ``<origin xyz=COM>``; the
parallel-axis theorem ``I_COM = I_origin - m*(||c||^2 I - c c^T)``
recovers the COM-frame tensor.
"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .friction import friction_param_count
from .urdf_parser import parse_urdf, resolve_xacro_to_urdf_xml


PathLike = Union[str, Path]


def export_adapted_urdf(
    *,
    input_urdf_path: PathLike,
    pi_full: np.ndarray,
    n_dof: int,
    friction_model: str,
    output_urdf_path: PathLike,
    friction_sidecar_path: Optional[PathLike] = None,
    logger: Optional[logging.Logger] = None,
    parameter_source: str = "pi_corrected",
) -> dict:
    """Write an adapted URDF whose dynamic parameters match ``pi_full``.

    Parameters
    ----------
    input_urdf_path :
        Path to the nominal URDF (or .xacro). Topology, visuals, and
        collisions are preserved; only ``<inertial>`` blocks on revolute
        child links and ``<dynamics>`` tags on revolute joints are
        rewritten.
    pi_full :
        Length ``10*n_dof + n_friction`` parameter vector from the
        identification stage (typically ``identification['pi_corrected']``).
    n_dof :
        Number of revolute degrees of freedom.
    friction_model :
        One of ``"none"``, ``"viscous"``, ``"coulomb"``, ``"viscous_coulomb"``;
        controls how ``pi_full[10*n_dof:]`` is interpreted.
    output_urdf_path :
        Destination for the adapted URDF.
    friction_sidecar_path :
        Optional destination for a JSON sidecar carrying the *full*
        asymmetric friction model.  Skipped when ``None`` or when
        ``friction_model == "none"``.
    logger :
        Optional logger.  Falls back to ``logging.getLogger(__name__)``.
    parameter_source :
        Tag written into the sidecar metadata; defaults to ``"pi_corrected"``.

    Returns
    -------
    dict
        Metadata describing what was written; suitable for splicing into
        ``results_summary.json``.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    input_urdf_path = Path(input_urdf_path)
    output_urdf_path = Path(output_urdf_path)
    sidecar_path = Path(friction_sidecar_path) if friction_sidecar_path is not None else None

    pi_full = np.asarray(pi_full, dtype=float).reshape(-1)

    n_fric = friction_param_count(n_dof, friction_model)
    expected = 10 * n_dof + n_fric
    if pi_full.size != expected:
        raise ValueError(
            f"parameter length mismatch: got {pi_full.size}, expected {expected} "
            f"(10*{n_dof} rigid + {n_fric} {friction_model} friction)"
        )
    pi_rigid = pi_full[: 10 * n_dof]
    theta_f = pi_full[10 * n_dof :]

    # Reuse the official parser to determine chain order.
    robot = parse_urdf(str(input_urdf_path))
    if robot.nDoF != n_dof:
        raise ValueError(
            f"URDF nDoF ({robot.nDoF}) does not match identified nDoF ({n_dof})."
        )

    revolute_joint_names = list(robot.revolute_joint_names)
    revolute_child_links = [
        j.child for j in robot.chain_joints
        if j.joint_type in ("revolute", "continuous")
    ]
    if len(revolute_child_links) != n_dof:
        raise ValueError(
            f"Parsed revolute child link count ({len(revolute_child_links)}) "
            f"does not match identified nDoF ({n_dof})."
        )

    # Load an editable XML tree (handles xacro by resolving to URDF text first).
    tree = _load_editable_tree(input_urdf_path)
    root = tree.getroot()

    # 1. Overwrite per-link inertials.
    inertials = _unpack_link_inertials(pi_rigid, n_dof)
    name_to_link = {l.get("name"): l for l in root.findall("link")}
    for link_name, blk in zip(revolute_child_links, inertials):
        if link_name not in name_to_link:
            raise ValueError(
                f"Link '{link_name}' returned by parse_urdf is missing from "
                f"the raw XML root '{input_urdf_path}'. Cannot edit inertials."
            )
        link_el = name_to_link[link_name]
        inertial_el = _set_or_create(link_el, "inertial")
        origin_el = _set_or_create(inertial_el, "origin")
        origin_el.set("xyz", _format_xyz(blk["com"]))
        origin_el.set("rpy", "0 0 0")
        mass_el = _set_or_create(inertial_el, "mass")
        mass_el.set("value", f"{blk['mass']:.9g}")
        I = blk["inertia_at_com"]
        inertia_el = _set_or_create(inertial_el, "inertia")
        inertia_el.set("ixx", f"{I[0, 0]:.9g}")
        inertia_el.set("ixy", f"{I[0, 1]:.9g}")
        inertia_el.set("ixz", f"{I[0, 2]:.9g}")
        inertia_el.set("iyy", f"{I[1, 1]:.9g}")
        inertia_el.set("iyz", f"{I[1, 2]:.9g}")
        inertia_el.set("izz", f"{I[2, 2]:.9g}")

    # 2. Overwrite per-joint <dynamics> tags from the friction parameters.
    friction_per_joint = _split_friction(theta_f, n_dof, friction_model)
    name_to_joint = {j.get("name"): j for j in root.findall("joint")}
    if friction_model != "none":
        for joint_name, fric in zip(revolute_joint_names, friction_per_joint):
            if joint_name not in name_to_joint:
                raise ValueError(
                    f"Joint '{joint_name}' is missing from the URDF XML."
                )
            j_el = name_to_joint[joint_name]
            dyn_el = _set_or_create(j_el, "dynamics")
            if fric["Fv"] is not None:
                dyn_el.set("damping", f"{fric['Fv']:.9g}")
            if fric["Fcp"] is not None or fric["Fcn"] is not None:
                mag = 0.5 * (abs(fric["Fcp"] or 0.0) + abs(fric["Fcn"] or 0.0))
                dyn_el.set("friction", f"{mag:.9g}")

    # 3. Write the URDF.
    output_urdf_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(str(output_urdf_path), encoding="utf-8", xml_declaration=True)
    log.info("Wrote adapted URDF: %s", output_urdf_path)

    # 4. Friction sidecar JSON (full asymmetric model, kept identical to the
    #    contract used by src/export_adapted_urdf.py so external consumers
    #    keep working regardless of the entry point).
    sidecar_written: Optional[str] = None
    if sidecar_path is not None and friction_model != "none":
        sidecar = {
            "source": {
                "input_urdf": str(input_urdf_path),
                "parameter_source": parameter_source,
            },
            "friction_model": friction_model,
            "joints": [
                {
                    "name": jn,
                    "Fv_viscous": fr["Fv"],
                    "Fcp_coulomb_positive": fr["Fcp"],
                    "Fcn_coulomb_negative": fr["Fcn"],
                }
                for jn, fr in zip(revolute_joint_names, friction_per_joint)
            ],
            "notes": [
                "URDF <dynamics damping> carries Fv (viscous).",
                "Fv may have been clamped to 0 by the pipeline if the "
                "unconstrained solver returned a negative value; raw values "
                "remain in identification_results.npz['pi_identified'].",
                "URDF <dynamics friction> carries 0.5*(|Fcp|+|Fcn|) since the "
                "schema cannot represent direction-dependent Coulomb friction.",
                "Use this sidecar if your simulator supports asymmetric "
                "Coulomb friction.",
            ],
        }
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        log.info("Wrote friction sidecar: %s", sidecar_path)
        sidecar_written = str(sidecar_path)

    return {
        "adapted_urdf_path": str(output_urdf_path),
        "friction_sidecar_path": sidecar_written,
        "n_dof": n_dof,
        "friction_model": friction_model,
        "n_friction_params": n_fric,
        "revolute_joint_names": revolute_joint_names,
        "revolute_child_links": revolute_child_links,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_editable_tree(input_urdf_path: Path) -> ET.ElementTree:
    """Return an editable ElementTree from a .urdf or .xacro source."""
    if input_urdf_path.suffix.lower() != ".xacro":
        return ET.parse(str(input_urdf_path))
    xml_text = resolve_xacro_to_urdf_xml(input_urdf_path)
    return ET.ElementTree(ET.fromstring(xml_text))


def _split_friction(theta_f: np.ndarray, n_dof: int, model: str):
    """Return per-joint dict of {Fv, Fcp, Fcn} (each may be None)."""
    out = [dict(Fv=None, Fcp=None, Fcn=None) for _ in range(n_dof)]
    if model == "none" or theta_f.size == 0:
        return out
    if model == "viscous":
        for j in range(n_dof):
            out[j]["Fv"] = float(theta_f[j])
    elif model == "coulomb":
        for j in range(n_dof):
            out[j]["Fcp"] = float(theta_f[j])
            out[j]["Fcn"] = float(theta_f[n_dof + j])
    elif model == "viscous_coulomb":
        for j in range(n_dof):
            out[j]["Fv"] = float(theta_f[j])
            out[j]["Fcp"] = float(theta_f[n_dof + j])
            out[j]["Fcn"] = float(theta_f[2 * n_dof + j])
    else:
        raise ValueError(f"unknown friction model: {model}")
    return out


def _unpack_link_inertials(pi_rigid: np.ndarray, n_dof: int):
    """Map the (10*n,) rigid-body parameter vector back to per-link
    (mass, COM in link frame, inertia about COM in link frame)."""
    out = []
    for j in range(n_dof):
        b = pi_rigid[10 * j : 10 * (j + 1)]
        m = float(b[0])
        if m <= 0.0:
            raise ValueError(
                f"link index {j}: identified mass {m} is non-positive; "
                f"refusing to export. Re-run identification with "
                f"identification.parameter_bounds=true (or feasibility "
                f"constraints enabled) to obtain a physically consistent "
                f"adapted URDF."
            )
        h = np.array([b[1], b[2], b[3]], dtype=float)            # m * c
        c = h / m                                                 # COM in link frame
        I_origin = np.array([
            [b[4], b[5], b[6]],
            [b[5], b[7], b[8]],
            [b[6], b[8], b[9]],
        ], dtype=float)
        # Parallel-axis: I_origin = I_COM + m*(||c||^2 I - c c^T)
        # so          I_COM    = I_origin - m*(||c||^2 I - c c^T)
        I_com = I_origin - m * (float(c.dot(c)) * np.eye(3) - np.outer(c, c))
        out.append({"mass": m, "com": c, "inertia_at_com": I_com})
    return out


def _set_or_create(parent: ET.Element, tag: str) -> ET.Element:
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    return el


def _format_xyz(v: np.ndarray) -> str:
    return " ".join(f"{x: .9g}" for x in v)
