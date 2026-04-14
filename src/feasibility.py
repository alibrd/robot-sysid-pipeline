"""Physical feasibility checks for identified inertial parameters.

The definitive criterion for physical consistency of rigid-body parameters
is that the **4×4 pseudo-inertia matrix** J is positive semidefinite:

    J = [[σI₃ − I,  h],
         [hᵀ,       m]]

where σ = ½ tr(I), h = [mx, my, mz]ᵀ.

J ≽ 0 implies (and is stronger than) positive mass, inertia PSD, and
triangle inequalities combined.  See Sousa & Cortesão (2014) and
Wensing, Kim & Slotine (2018).

Feasibility enforcement methods:
  - "lmi" : eigenvalue-clipping projection of each link's pseudo-inertia
            J onto the PSD cone, combined with SLSQP-constrained optimisation
            in the solver.
  - "cholesky" : Cholesky-factored reparameterisation.  Each link's
            pseudo-inertia is written as J = L Lᵀ with L lower-triangular,
            guaranteeing J ≽ 0 by construction during optimisation.  Post-hoc
            projection (if needed) uses eigenvalue-clipping.
"""
import logging
import numpy as np

logger = logging.getLogger("sysid_pipeline")


# ── Public helpers ─────────────────────────────────────────────────────

def pi_from_pseudo_inertia_matrix(J: np.ndarray) -> np.ndarray:
    """Extract a 10-element parameter block from a 4×4 pseudo-inertia matrix.

    Inverse of :func:`pseudo_inertia_matrix`.

    Returns
    -------
    pi_link : [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    """
    m   = J[3, 3]
    hx  = J[0, 3]
    hy  = J[1, 3]
    hz  = J[2, 3]
    Ixx = J[1, 1] + J[2, 2]
    Iyy = J[0, 0] + J[2, 2]
    Izz = J[0, 0] + J[1, 1]
    Ixy = -J[0, 1]
    Ixz = -J[0, 2]
    Iyz = -J[1, 2]
    return np.array([m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz])


def pseudo_inertia_matrix(pi_link: np.ndarray) -> np.ndarray:
    """Build the 4×4 pseudo-inertia matrix from a 10-element parameter block.

    Parameters
    ----------
    pi_link : [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

    Returns
    -------
    J : (4, 4) symmetric matrix.  J ≽ 0 ⟺ physically consistent.
    """
    m   = pi_link[0]
    hx  = pi_link[1]
    hy  = pi_link[2]
    hz  = pi_link[3]
    Ixx = pi_link[4]
    Ixy = pi_link[5]
    Ixz = pi_link[6]
    Iyy = pi_link[7]
    Iyz = pi_link[8]
    Izz = pi_link[9]
    sigma = 0.5 * (Ixx + Iyy + Izz)
    return np.array([
        [sigma - Ixx, -Ixy,        -Ixz,        hx],
        [-Ixy,        sigma - Iyy, -Iyz,        hy],
        [-Ixz,        -Iyz,        sigma - Izz, hz],
        [hx,          hy,          hz,           m ],
    ])


def is_pseudo_inertia_psd(pi_link: np.ndarray, tol: float = -1e-10) -> bool:
    """Return True if the pseudo-inertia matrix of *pi_link* is PSD."""
    J = pseudo_inertia_matrix(pi_link)
    return bool(np.min(np.linalg.eigvalsh(J)) >= tol)


def check_feasibility(pi: np.ndarray, nDoF: int, method: str = "none"):
    """Check physical feasibility of the identified parameter vector.

    The primary criterion is pseudo-inertia PSD (J ≽ 0) per link.
    Legacy checks (positive mass, inertia PSD, triangle inequality) are
    reported for diagnostics but the feasibility verdict is driven by J.

    Parameters
    ----------
    pi : (10*nDoF,) or (p,) parameter vector.  First 10*nDoF entries
         are [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] per link.
    nDoF : number of links/joints
    method : "none" | "lmi" | "cholesky"

    Returns
    -------
    report : list of dicts with per-link feasibility information
    feasible : bool — True only if every link passes pseudo-inertia PSD
    pi_out : corrected parameter vector (only differs from *pi* when
             *method* is "lmi" or "cholesky" and violations are found)
    """
    report = []
    all_feasible = True

    for i in range(nDoF):
        base = 10 * i
        if base + 10 > len(pi):
            break  # parameters may be reduced
        pi_link = pi[base:base + 10]
        m_i = pi_link[0]
        Ixx, Ixy, Ixz = pi_link[4], pi_link[5], pi_link[6]
        Iyy, Iyz, Izz = pi_link[7], pi_link[8], pi_link[9]

        I_mat = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])
        inertia_eigs = np.linalg.eigvalsh(I_mat)

        link_report = {"link": i + 1, "mass": m_i, "feasible": True, "issues": []}
        link_report["inertia_eigenvalues"] = inertia_eigs.tolist()

        # ── Diagnostic checks (informational) ────────────────────────
        if m_i <= 0:
            link_report["issues"].append(f"Non-positive mass: {m_i:.6e}")
        if np.any(inertia_eigs < -1e-12):
            link_report["issues"].append(
                f"Inertia not PSD: eigenvalues={inertia_eigs}")
        tri_checks = [
            (Ixx + Iyy - Izz, "Ixx + Iyy >= Izz"),
            (Ixx + Izz - Iyy, "Ixx + Izz >= Iyy"),
            (Iyy + Izz - Ixx, "Iyy + Izz >= Ixx"),
        ]
        for val, desc in tri_checks:
            if val < -1e-12:
                link_report["issues"].append(
                    f"Triangle ineq. violated: {desc} (deficit={val:.6e})")

        # ── Definitive criterion: pseudo-inertia PSD ─────────────────
        J = pseudo_inertia_matrix(pi_link)
        J_eigs = np.linalg.eigvalsh(J)
        link_report["pseudo_inertia_eigenvalues"] = J_eigs.tolist()

        if np.min(J_eigs) < -1e-10:
            link_report["issues"].append(
                f"Pseudo-inertia NOT PSD: min eigenvalue={np.min(J_eigs):.6e}")
            link_report["feasible"] = False

        if not link_report["feasible"]:
            all_feasible = False

        report.append(link_report)

    # ── Method-specific post-hoc projection ────────────────────────────
    if method in ("lmi", "cholesky") and not all_feasible:
        logger.warning("Pseudo-inertia PSD violations detected. "
                       "Projecting onto feasible set (method=%s).", method)
        pi_corrected = _project_pseudo_inertia(pi, nDoF, report)
        return report, all_feasible, pi_corrected

    return report, all_feasible, pi


def _project_pseudo_inertia(pi, nDoF, report):
    """Project each link's pseudo-inertia matrix onto the PSD cone.

    Eigenvalue-clipping of J followed by parameter extraction.
    Used by both the "lmi" and "cholesky" paths.
    """
    pi_out = pi.copy()
    eps_min = 1e-8

    for r in report:
        if r["feasible"]:
            continue
        i = r["link"] - 1
        base = 10 * i
        pi_link = pi_out[base:base + 10].copy()

        J = pseudo_inertia_matrix(pi_link)
        eigvals, eigvecs = np.linalg.eigh(J)
        eigvals_clipped = np.maximum(eigvals, eps_min)
        J_fixed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

        pi_out[base:base + 10] = pi_from_pseudo_inertia_matrix(J_fixed)
        logger.debug("Link %d: pseudo-inertia projected. New J eigenvalues: %s",
                     i + 1, eigvals_clipped)

    return pi_out
