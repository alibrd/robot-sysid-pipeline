"""Base (identifiable) parameter reduction via QR decomposition.

Implements two-step reduction:
  1. Remove zero columns from the observation matrix.
  2. QR decomposition with column pivoting to find linearly independent columns.

The regrouping matrix ``P`` satisfies ``W @ pi_full = W_base @ (P @ pi_full)``
so the base observation equation is ``W_base @ pi_base = tau`` where
``pi_base = P @ pi_full``.

Works for both Newton-Euler and Euler-Lagrange regressor matrices.
"""
import logging
import numpy as np
from scipy.linalg import qr as scipy_qr

logger = logging.getLogger("sysid_pipeline")


def compute_base_parameters(W: np.ndarray,
                            pi_full: np.ndarray,
                            tol: float = 1e-10):
    """Reduce observation matrix W (N*n x p) to base parameter form.

    Parameters
    ----------
    W : (m, p) stacked observation matrix (regressor over all time steps)
    pi_full : (p,) full parameter vector
    tol : tolerance for rank determination

    Returns
    -------
    W_base : (m, r) reduced observation matrix preserving W @ pi = W_base @ pi_base
    P : (r, p) regrouping matrix such that pi_base = P @ pi_full
    kept_cols : list of independent column indices (in full-parameter indexing)
    rank : numerical rank of W
    pi_base : (r,) base parameter vector = P @ pi_full
    """
    m, p = W.shape

    # Step 1: remove zero columns ------------------------------------------------
    col_norms = np.linalg.norm(W, axis=0)
    nonzero_mask = col_norms > tol
    nonzero_idx = np.where(nonzero_mask)[0]
    W_nz = W[:, nonzero_idx]

    if W_nz.shape[1] == 0:
        raise ValueError("All columns of the observation matrix are zero.")

    logger.debug("Step 1: removed %d zero columns (%d -> %d).",
                 p - len(nonzero_idx), p, len(nonzero_idx))

    # Step 2: QR with column pivoting (scipy) ------------------------------------
    Q, R, piv = scipy_qr(W_nz, pivoting=True)

    # Determine numerical rank from diagonal of R
    diag_R = np.abs(np.diag(R))
    rank = int(np.sum(diag_R > tol * diag_R[0]))

    if rank == 0:
        raise ValueError("Observation matrix has numerical rank 0.")

    logger.debug("Step 2: QR rank = %d (of %d non-zero columns).", rank, len(nonzero_idx))

    # Independent / dependent split in the pivoted order
    base_local_idx = piv[:rank]      # indices into W_nz
    dep_local_idx = piv[rank:]       # indices into W_nz
    kept_cols = nonzero_idx[base_local_idx].tolist()   # full-parameter indices

    # Build regrouping matrix P (r x p)
    # Identity part: pi_base[i] absorbs pi_full[kept_cols[i]]
    P = np.zeros((rank, p))
    for i, col in enumerate(kept_cols):
        P[i, col] = 1.0

    # Merge dependent columns into base parameters
    R11 = R[:rank, :rank]
    if len(dep_local_idx) > 0:
        R12 = R[:rank, rank:]
        dep_global = nonzero_idx[dep_local_idx]  # full-parameter indices

        # beta (r x n_dep): W_dep ~= W_base @ beta  =>  R11 @ beta = R12
        beta = np.linalg.solve(R11, R12)

        for j in range(beta.shape[1]):
            dep_col = dep_global[j]
            for i in range(rank):
                P[i, dep_col] = beta[i, j]

    # Build reduced observation matrix: W_base = W[:, kept_cols] + contributions
    # The key identity is W ~= W_base @ P  (in rank sense), hence
    # W_base @ pi_base = W_base @ P @ pi_full ~= W @ pi_full
    # W_base here is the independent columns.
    W_base = W[:, kept_cols]

    pi_base = P @ pi_full

    # Verify the observation equation is preserved
    err = np.linalg.norm(W @ pi_full - W_base @ pi_base)
    if err > 1e-6 * np.linalg.norm(W @ pi_full):
        logger.warning("Base-parameter verification: ||W*pi - W_b*pi_b|| = %.3e "
                       "(relative %.3e)", err, err / (np.linalg.norm(W @ pi_full) + 1e-30))

    logger.info("Base parameter reduction: %d -> %d identifiable parameters.", p, rank)
    return W_base, P, kept_cols, rank, pi_base
