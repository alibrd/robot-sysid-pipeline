"""Fourier-based parameterised trajectory generation.

Supports three basis-function modes (cosine, sine, both) with optional phase
optimisation and the λ-correction terms from the reference documents.
"""
import numpy as np


def fourier_trajectory(params: np.ndarray,
                       freqs: np.ndarray,
                       t: np.ndarray,
                       q0: np.ndarray,
                       basis: str = "both",
                       optimize_phase: bool = False):
    """Evaluate parameterised Fourier trajectory and its derivatives.

    Parameters
    ----------
    params : optimisation variables (shape depends on *basis* and *optimize_phase*)
        - "cosine" : (nDoF * m,) amplitudes a
        - "sine"   : (nDoF * m,) amplitudes b
        - "both" + optimize_phase=False : (nDoF * 2 * m,) [a | b]
        - "both" + optimize_phase=True  : (nDoF * 2 * m,) [a | phi]
    freqs : (m,) harmonic frequencies [Hz]
    t : (N,) or scalar time values
    q0 : (nDoF,) initial joint configuration
    basis : "cosine" | "sine" | "both"
    optimize_phase : only used when basis="both"

    Returns
    -------
    q, dq, ddq : each (nDoF, N) arrays
    """
    t = np.atleast_1d(t)
    N = t.size
    m = freqs.size
    nDoF = q0.size
    twopi = 2.0 * np.pi

    # Harmonic matrices (m × N)
    phase = twopi * np.outer(freqs, t)  # (m, N)
    cos_h = np.cos(phase)
    sin_h = np.sin(phase)
    coeff = twopi * freqs  # (m,)

    q = np.zeros((nDoF, N))
    dq = np.zeros((nDoF, N))
    ddq = np.zeros((nDoF, N))

    if basis == "cosine":
        a = params.reshape(nDoF, m)
        for i in range(nDoF):
            # q_i = Σ a_ij [cos(2πf_j t) - 1]  + q0_i
            q[i] = np.sum(a[i, :, None] * (cos_h - 1.0), axis=0) + q0[i]
            dq[i] = np.sum(-a[i, :, None] * coeff[:, None] * sin_h, axis=0)
            ddq[i] = np.sum(-a[i, :, None] * (coeff**2)[:, None] * cos_h, axis=0)

    elif basis == "sine":
        b = params.reshape(nDoF, m)
        for i in range(nDoF):
            # Correction: ensure q(0) = q0 and dq(0) = 0
            # sin(0) = 0, so q(0) = λ_{i,0} + q0 → λ_{i,0} = 0
            # dq(0) = Σ b_ij 2πf_j cos(0) + λ_{i,1} = 0  →
            #   λ_{i,1} = - 2π Σ b_ij f_j
            lam1 = -twopi * np.sum(b[i] * freqs)
            q[i] = (np.sum(b[i, :, None] * sin_h, axis=0)
                     + lam1 * t + q0[i])
            dq[i] = (np.sum(b[i, :, None] * coeff[:, None] * cos_h, axis=0)
                      + lam1)
            ddq[i] = np.sum(-b[i, :, None] * (coeff**2)[:, None] * sin_h, axis=0)

            # Ensure dq at end is zero: add cubic correction if needed
            # For now rely on optimizer constraints

    elif basis == "both" and not optimize_phase:
        # Ref-1 §4.1.3: a and b amplitudes
        ab = params.reshape(nDoF, 2 * m)
        a = ab[:, :m]
        b = ab[:, m:]
        for i in range(nDoF):
            lam1 = -twopi * np.sum(b[i] * freqs)
            lam0 = -np.sum(a[i])
            q[i] = (np.sum(a[i, :, None] * cos_h + b[i, :, None] * sin_h, axis=0)
                     + lam1 * t + lam0 + q0[i])
            dq[i] = (np.sum(-a[i, :, None] * coeff[:, None] * sin_h
                             + b[i, :, None] * coeff[:, None] * cos_h, axis=0)
                      + lam1)
            ddq[i] = np.sum(
                -a[i, :, None] * (coeff**2)[:, None] * cos_h
                + b[i, :, None] * (coeff**2)[:, None] * (-sin_h), axis=0
            )

    elif basis == "both" and optimize_phase:
        # Ref-2 §4.1: a and phi
        ap = params.reshape(nDoF, 2 * m)
        a = ap[:, :m]
        phi = ap[:, m:]
        for i in range(nDoF):
            lam1 = twopi * np.sum(a[i] * freqs * np.sin(phi[i]))
            lam0 = -np.sum(a[i] * np.cos(phi[i]))
            q[i] = (np.sum(a[i, :, None] * np.cos(phase + phi[i, :, None]), axis=0)
                     + lam1 * t + lam0 + q0[i])
            dq[i] = (np.sum(-a[i, :, None] * coeff[:, None]
                             * np.sin(phase + phi[i, :, None]), axis=0)
                      + lam1)
            ddq[i] = np.sum(
                -a[i, :, None] * (coeff**2)[:, None]
                * np.cos(phase + phi[i, :, None]), axis=0
            )
    else:
        raise ValueError(f"Invalid basis/phase combination: {basis}, {optimize_phase}")

    return q, dq, ddq


def build_frequencies(base_freq: float, num_harmonics: int) -> np.ndarray:
    """Return (m,) array of harmonic frequencies."""
    return np.array([(i + 1) * base_freq for i in range(num_harmonics)])


def param_count(nDoF: int, num_harmonics: int, basis: str, optimize_phase: bool) -> int:
    """Number of optimisation variables for the trajectory."""
    m = num_harmonics
    if basis in ("cosine", "sine"):
        return nDoF * m
    return nDoF * 2 * m  # both (a+b or a+phi)


def param_bounds(nDoF: int, num_harmonics: int, basis: str, optimize_phase: bool,
                 q_lim: np.ndarray, freqs: np.ndarray = None,
                 dq_lim: np.ndarray = None, ddq_lim: np.ndarray = None):
    """Bounds for trajectory optimisation variables.

    When ``freqs``, ``dq_lim``, and ``ddq_lim`` are provided the per-harmonic
    amplitude bound is tightened so that a single harmonic at maximum amplitude
    cannot alone exceed the velocity or acceleration limits.  This shrinks the
    search space to a region where feasibility is achievable, which is critical
    for gradient-based optimisers such as SLSQP.
    """
    m = num_harmonics
    bounds = []

    def _amp_bound(i, j):
        amp = (q_lim[i, 1] - q_lim[i, 0]) / 2.0
        if freqs is not None:
            omega = 2.0 * np.pi * freqs[j]
            if dq_lim is not None:
                dq_max = min(abs(dq_lim[i, 0]), abs(dq_lim[i, 1]))
                amp = min(amp, dq_max / omega)
            if ddq_lim is not None:
                ddq_max = min(abs(ddq_lim[i, 0]), abs(ddq_lim[i, 1]))
                amp = min(amp, ddq_max / omega ** 2)
        return amp

    if basis == "cosine" or basis == "sine":
        for i in range(nDoF):
            for j in range(m):
                a = _amp_bound(i, j)
                bounds.append((-a, a))
    elif basis == "both" and not optimize_phase:
        for i in range(nDoF):
            for j in range(m):          # cosine amplitudes
                a = _amp_bound(i, j)
                bounds.append((-a, a))
            for j in range(m):          # sine amplitudes
                a = _amp_bound(i, j)
                bounds.append((-a, a))
    elif basis == "both" and optimize_phase:
        for i in range(nDoF):
            for j in range(m):          # amplitudes
                a = _amp_bound(i, j)
                bounds.append((-a, a))
            for _ in range(m):          # phases
                bounds.append((-np.pi, np.pi))
    return bounds
