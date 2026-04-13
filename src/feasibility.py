"""Physical feasibility checks for identified inertial parameters.

The definitive criterion for physical consistency of rigid-body parameters
is that the **4×4 pseudo-inertia matrix** J is positive semidefinite:

    J = [[σI₃ − I,  h],
         [hᵀ,       m]]

where σ = ½ tr(I), h = [mx, my, mz]ᵀ.

J ≽ 0 implies (and is stronger than) positive mass, inertia PSD, and
triangle inequalities combined.  See Sousa & Cortesão (2014) and
Wensing, Kim & Slotine (2018).

Post-hoc projection method:
  - "lmi" : eigenvalue-clipping of each link's pseudo-inertia J onto
            the PSD cone.  ("cholesky" is a deprecated alias for the
            same projection — no true Cholesky reparameterisation exists.)
"""
import logging
import numpy as np

logger = logging.getLogger("sysid_pipeline")


# ── Public helpers ─────────────────────────────────────────────────────

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

        # Extract corrected parameters from J_fixed
        # sigma = J[0,0] + J[1,1] + J[2,2]  (since sum of diag of upper-left 3x3)
        sigma = J_fixed[0, 0] + J_fixed[1, 1] + J_fixed[2, 2]
        Ixx_new = sigma - J_fixed[0, 0]
        Iyy_new = sigma - J_fixed[1, 1]
        Izz_new = sigma - J_fixed[2, 2]
        Ixy_new = -J_fixed[0, 1]
        Ixz_new = -J_fixed[0, 2]
        Iyz_new = -J_fixed[1, 2]
        hx_new  = J_fixed[0, 3]
        hy_new  = J_fixed[1, 3]
        hz_new  = J_fixed[2, 3]
        m_new   = J_fixed[3, 3]

        pi_out[base + 0] = m_new
        pi_out[base + 1] = hx_new
        pi_out[base + 2] = hy_new
        pi_out[base + 3] = hz_new
        pi_out[base + 4] = Ixx_new
        pi_out[base + 5] = Ixy_new
        pi_out[base + 6] = Ixz_new
        pi_out[base + 7] = Iyy_new
        pi_out[base + 8] = Iyz_new
        pi_out[base + 9] = Izz_new
        logger.debug("Link %d: pseudo-inertia projected. New J eigenvalues: %s",
                     i + 1, eigvals_clipped)

    return pi_out
