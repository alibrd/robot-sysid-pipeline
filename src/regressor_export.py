"""Standalone external-use regressor + parameter export.

Emits two artifacts that a downstream user can copy into another Python
project (NumPy-only runtime, no `src.*` imports):

  * ``regressor.py`` - self-contained callable regressor with
    ``Y_rigid``, ``Y_augmented``, ``Y``, ``tau``, ``Y_stack``,
    ``tau_traj``, and a ``META`` dict.
  * ``parameters.pkl`` - pickled dict carrying ``pi_rigid``,
    ``pi_augmented``, ``pi_friction`` (and an alias ``pi``) plus enough
    metadata to interpret the vector without this repository.

Two backends are supported, matching the pipeline's ``method`` config:

  * ``newton_euler`` - emits per-joint kinematic primitives via SymPy
    printing plus a transcribed Newton-Euler recursive routine.
  * ``euler_lagrange`` - reloads the cached symbolic ``Y_sym`` matrix
    and prints it entry-by-entry as NumPy source.

The emitted file inlines a ``_skew`` helper and a friction block, so the
runtime artifact has no ``src.math_utils`` or ``src.friction`` imports.
"""
from __future__ import annotations

import json
import pickle
import textwrap
from pathlib import Path

import numpy as np
import sympy
from sympy.printing.numpy import NumPyPrinter

from .dynamics_euler_lagrange import euler_lagrange_regressor_builder
from .regressor_model import RegressorModel


_PRINTER = NumPyPrinter()


def export_standalone(model: RegressorModel, output_dir: str | Path) -> Path:
    """Write ``regressor.py`` to *output_dir* and return its path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if model.backend == "newton_euler":
        source = _emit_ne_source(model)
    elif model.backend == "euler_lagrange":
        source = _emit_el_source(model)
    else:
        raise ValueError(
            f"Unsupported regressor backend {model.backend!r} for export; "
            "expected 'newton_euler' or 'euler_lagrange'."
        )

    path = out_dir / "regressor.py"
    path.write_text(source, encoding="utf-8")
    return path


def export_dynamics_model_closed_form(
        model: RegressorModel,
        pi_aug: np.ndarray,
        output_dir: str | Path,
        *,
        simplify: str = "trigsimp",
        include_coriolis_matrix: bool = False) -> Path:
    """Emit standalone closed-form ``dynamics_model.py`` to *output_dir*.

    The emitted module has

        tau(q, dq, ddq) = M(q) @ ddq + c(q, dq) + g(q) + tau_f(dq)

    where each of ``M``, ``c``, ``g``, ``tau_f`` is a pure NumPy closed-form
    function of its inputs with the active parameter vector baked into the
    printed source. No ``parameters.pkl``, no inter-function calls.

    Parameters
    ----------
    model : RegressorModel
        Provides nDoF, friction model, and access to the cached symbolic EL
        regressor. The EL symbolic build is forced/loaded regardless of the
        runtime backend.
    pi_aug : ndarray
        Augmented parameter vector of length ``10*nDoF + n_friction``.
    output_dir : str | Path
        Destination directory; the file is written as
        ``<output_dir>/dynamics_model.py``.
    simplify : {"none", "trigsimp", "full"}
        SymPy simplification pass applied to each printed expression.
    include_coriolis_matrix : bool
        Also emit a closed-form Christoffel ``C(q, dq)`` such that
        ``c = C @ dq``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = model.nDoF
    n_rigid = 10 * n
    n_fric = model.n_friction_params

    pi_aug = np.asarray(pi_aug, dtype=float).reshape(-1)
    expected = n_rigid + n_fric
    if pi_aug.size != expected:
        raise ValueError(
            f"pi_aug has {pi_aug.size} entries, expected {expected} "
            f"({n_rigid} rigid + {n_fric} friction)."
        )
    pi_rigid_num = pi_aug[:n_rigid]
    pi_friction_num = pi_aug[n_rigid:]

    Y_sym, kept_cols = _load_symbolic_rigid_regressor(model)
    q_syms = [sympy.Symbol(f"q{i + 1}") for i in range(n)]
    dq_syms = [sympy.Symbol(f"dq{i + 1}") for i in range(n)]
    ddq_syms = [sympy.Symbol(f"ddq{i + 1}") for i in range(n)]

    # tau_rigid[i] = sum_k pi_rigid[k] * Y_sym_full[i, k], skipping zero-coef.
    tau_rigid = []
    for i in range(n):
        expr = sympy.S.Zero
        for c_idx, k in enumerate(kept_cols):
            coef = float(pi_rigid_num[k])
            if coef == 0.0:
                continue
            expr = expr + sympy.Float(coef) * Y_sym[i, c_idx]
        tau_rigid.append(expr)

    zero_dq = {dq_syms[i]: sympy.S.Zero for i in range(n)}
    zero_ddq = {ddq_syms[i]: sympy.S.Zero for i in range(n)}

    g_sym = [sympy.expand(tau_rigid[i].subs(zero_dq).subs(zero_ddq))
             for i in range(n)]
    M_sym = [
        [sympy.diff(tau_rigid[i], ddq_syms[j]) for j in range(n)]
        for i in range(n)
    ]
    c_sym = [
        sympy.expand(tau_rigid[i].subs(zero_ddq) - g_sym[i])
        for i in range(n)
    ]

    if include_coriolis_matrix:
        C_sym = [[sympy.S.Zero for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                expr = sympy.S.Zero
                for k in range(n):
                    G_ijk = sympy.Rational(1, 2) * (
                        sympy.diff(M_sym[i][j], q_syms[k])
                        + sympy.diff(M_sym[i][k], q_syms[j])
                        - sympy.diff(M_sym[j][k], q_syms[i])
                    )
                    expr = expr + G_ijk * dq_syms[k]
                C_sym[i][j] = expr
    else:
        C_sym = None

    tau_f_sym = _sympy_friction_block(
        dq_syms, model.friction_model, pi_friction_num, n
    )

    simplifier = _resolve_simplifier(simplify)
    if simplifier is not None:
        for i in range(n):
            g_sym[i] = simplifier(g_sym[i])
            c_sym[i] = simplifier(c_sym[i])
            tau_f_sym[i] = simplifier(tau_f_sym[i])
            for j in range(n):
                M_sym[i][j] = simplifier(M_sym[i][j])
            if C_sym is not None:
                for j in range(n):
                    C_sym[i][j] = simplifier(C_sym[i][j])

    aliases = {}
    for i in range(n):
        aliases[q_syms[i]] = sympy.Symbol(f"q_{i}")
        aliases[dq_syms[i]] = sympy.Symbol(f"dq_{i}")

    def _alias(e):
        return e.xreplace(aliases) if hasattr(e, "xreplace") else e

    g_sym = [_alias(e) for e in g_sym]
    c_sym = [_alias(e) for e in c_sym]
    M_sym = [[_alias(M_sym[i][j]) for j in range(n)] for i in range(n)]
    tau_f_sym = [_alias(e) for e in tau_f_sym]
    if C_sym is not None:
        C_sym = [[_alias(C_sym[i][j]) for j in range(n)] for i in range(n)]

    source = _emit_dynamics_module_source(
        model=model,
        g_sym=g_sym,
        M_sym=M_sym,
        c_sym=c_sym,
        tau_f_sym=tau_f_sym,
        C_sym=C_sym,
    )
    path = out_dir / "dynamics_model.py"
    path.write_text(source, encoding="utf-8")
    return path


def _load_symbolic_rigid_regressor(model: RegressorModel):
    """Return ``(Y_sym, kept_cols)`` for the EL symbolic rigid regressor."""
    kin = model.kin
    cache_dir = model._el_cache_dir()
    cache_path = Path(cache_dir) / "el_regressor_cache.pkl"
    if not cache_path.exists():
        euler_lagrange_regressor_builder(kin, cache_dir)
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    Y_sym = data["Y_sym"]
    kept_cols = [int(c) for c in data["kept_cols"]]
    return Y_sym, kept_cols


def _sympy_friction_block(dq_syms, friction_model, pi_friction_num, n):
    """Return list of length-n symbolic friction torques with baked params."""
    expr_list = [sympy.S.Zero for _ in range(n)]
    if friction_model == "none" or pi_friction_num.size == 0:
        return expr_list
    a = sympy.Float(10.0)
    b = sympy.Float(1000.0)
    offset = 0
    if friction_model in ("viscous", "viscous_coulomb"):
        for i in range(n):
            coef = float(pi_friction_num[offset + i])
            if coef != 0.0:
                expr_list[i] = expr_list[i] + sympy.Float(coef) * dq_syms[i]
        offset += n
    if friction_model in ("coulomb", "viscous_coulomb"):
        for i in range(n):
            coef = float(pi_friction_num[offset + i])
            if coef != 0.0:
                expr_list[i] = expr_list[i] + sympy.Float(coef) / (
                    1 + sympy.exp(a - b * dq_syms[i])
                )
        offset += n
        for i in range(n):
            coef = float(pi_friction_num[offset + i])
            if coef != 0.0:
                expr_list[i] = expr_list[i] - sympy.Float(coef) / (
                    1 + sympy.exp(a + b * dq_syms[i])
                )
        offset += n
    return expr_list


def _resolve_simplifier(name):
    if name in (None, "none", ""):
        return None
    if name == "trigsimp":
        return sympy.trigsimp
    if name == "full":
        return sympy.simplify
    raise ValueError(
        f"Unknown simplify mode {name!r}; expected 'none', 'trigsimp', or 'full'."
    )


def _emit_dynamics_module_source(*, model, g_sym, M_sym, c_sym,
                                 tau_f_sym, C_sym):
    n = model.nDoF
    friction_model = model.friction_model

    header = (
        '"""Standalone closed-form dynamics model.\n\n'
        "Generated by the sysid pipeline. Self-contained: pure NumPy, no\n"
        "sibling-file dependencies, no inter-function calls at runtime.\n\n"
        "    tau(q, dq, ddq) = M(q) @ ddq + c(q, dq) + g(q) + tau_f(dq)\n\n"
        "Each of M(q), g(q), c(q, dq), tau_f(dq) is a closed-form NumPy\n"
        "expression in its inputs with the active parameter vector baked\n"
        "in at export time. The only inter-function calls happen inside\n"
        "the convenience wrapper ``tau``; individual terms are pure.\n\n"
        f"nDoF: {n}\n"
        f"friction_model: {friction_model}\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "import numpy\n\n\n"
        f"_NDOF = {n}\n"
        f"_FRICTION_MODEL = {friction_model!r}\n\n\n"
        "def _coerce(v, n=_NDOF):\n"
        "    a = numpy.asarray(v, dtype=float).reshape(-1)\n"
        "    if a.size != n:\n"
        '        raise ValueError(f"expected {n} entries, got {a.size}.")\n'
        "    return a\n\n\n"
    )

    blocks = [header]
    blocks.append(_emit_g_function(g_sym, n))
    blocks.append(_emit_M_function(M_sym, n))
    blocks.append(_emit_c_function(c_sym, n))
    blocks.append(_emit_tau_f_function(tau_f_sym, n))
    if C_sym is not None:
        blocks.append(_emit_C_function(C_sym, n))
    blocks.append(_emit_tau_function())
    return "".join(blocks)


def _vec_body(exprs, n, indent="        "):
    cells = [_PRINTER.doprint(exprs[i]) for i in range(n)]
    return (",\n" + indent).join(cells)


def _mat_body(rows, n, indent="        "):
    row_strs = []
    for i in range(n):
        cells = [_PRINTER.doprint(rows[i][j]) for j in range(n)]
        row_strs.append("[" + ", ".join(cells) + "]")
    return (",\n" + indent).join(row_strs)


def _emit_g_function(g_sym, n):
    locals_q = "; ".join(f"q_{i} = q[{i}]" for i in range(n))
    body = _vec_body(g_sym, n)
    return (
        "def g(q):\n"
        "    q = _coerce(q)\n"
        f"    {locals_q}\n"
        "    return numpy.array([\n"
        f"        {body},\n"
        "    ], dtype=float)\n\n\n"
    )


def _emit_M_function(M_sym, n):
    locals_q = "; ".join(f"q_{i} = q[{i}]" for i in range(n))
    body = _mat_body(M_sym, n)
    return (
        "def M(q):\n"
        "    q = _coerce(q)\n"
        f"    {locals_q}\n"
        "    return numpy.array([\n"
        f"        {body},\n"
        "    ], dtype=float)\n\n\n"
    )


def _emit_c_function(c_sym, n):
    locals_q = "; ".join(f"q_{i} = q[{i}]" for i in range(n))
    locals_dq = "; ".join(f"dq_{i} = dq[{i}]" for i in range(n))
    body = _vec_body(c_sym, n)
    return (
        "def c(q, dq):\n"
        "    q = _coerce(q); dq = _coerce(dq)\n"
        f"    {locals_q}\n"
        f"    {locals_dq}\n"
        "    return numpy.array([\n"
        f"        {body},\n"
        "    ], dtype=float)\n\n\n"
    )


def _emit_tau_f_function(tau_f_sym, n):
    locals_dq = "; ".join(f"dq_{i} = dq[{i}]" for i in range(n))
    body = _vec_body(tau_f_sym, n)
    return (
        "def tau_f(dq):\n"
        "    dq = _coerce(dq)\n"
        f"    {locals_dq}\n"
        "    return numpy.array([\n"
        f"        {body},\n"
        "    ], dtype=float)\n\n\n"
    )


def _emit_C_function(C_sym, n):
    locals_q = "; ".join(f"q_{i} = q[{i}]" for i in range(n))
    locals_dq = "; ".join(f"dq_{i} = dq[{i}]" for i in range(n))
    body = _mat_body(C_sym, n)
    return (
        "def C(q, dq):\n"
        "    q = _coerce(q); dq = _coerce(dq)\n"
        f"    {locals_q}\n"
        f"    {locals_dq}\n"
        "    return numpy.array([\n"
        f"        {body},\n"
        "    ], dtype=float)\n\n\n"
    )


def _emit_tau_function():
    return (
        "def tau(q, dq, ddq):\n"
        '    """Total inverse-dynamics torque: M(q)@ddq + c(q,dq) + g(q) + tau_f(dq)."""\n'
        "    ddq = _coerce(ddq)\n"
        "    return M(q) @ ddq + c(q, dq) + g(q) + tau_f(dq)\n"
    )


def export_parameter_pickle(model: RegressorModel,
                            output_dir: str | Path,
                            pi: np.ndarray,
                            kind: str,
                            *,
                            residual: float | None = None,
                            feasibility_method: str | None = None) -> Path:
    """Write ``parameters.pkl`` carrying *pi* plus metadata.

    Parameters
    ----------
    model : RegressorModel
        Provides joint/link names, parameter ordering, and friction model.
    output_dir : str | Path
        Destination directory; the file is written as
        ``<output_dir>/parameters.pkl``.
    pi : ndarray
        Augmented parameter vector of length ``10*nDoF + n_friction``.
    kind : {"nominal", "identified"}
        Indicates whether the vector is URDF-extracted (Mode 1) or the
        identification-corrected estimate (Mode 2).
    residual : float | None
        Identification residual (Mode 2 only).
    feasibility_method : str | None
        Solver feasibility tag (Mode 2 only).
    """
    if kind not in {"nominal", "identified"}:
        raise ValueError(
            f"kind must be 'nominal' or 'identified', got {kind!r}."
        )

    pi_aug = np.asarray(pi, dtype=float).reshape(-1)
    n_rigid = model.n_rigid_params
    n_fric = model.n_friction_params
    expected = n_rigid + n_fric
    if pi_aug.size != expected:
        raise ValueError(
            f"pi has length {pi_aug.size}, expected {expected} "
            f"({n_rigid} rigid + {n_fric} friction)."
        )

    payload = {
        "pi": pi_aug.copy(),
        "pi_augmented": pi_aug.copy(),
        "pi_rigid": pi_aug[:n_rigid].copy(),
        "pi_friction": pi_aug[n_rigid:].copy(),
        "kind": kind,
        "nDoF": model.nDoF,
        "joint_names": model.joint_names(),
        "link_names": model.link_names(),
        "friction_model": model.friction_model,
        "backend": model.backend,
        "rigid_parameter_names": model.rigid_parameter_names(),
        "friction_parameter_names": model.friction_parameter_names(),
        "augmented_parameter_names": model.augmented_parameter_names(),
        "n_rigid_params": n_rigid,
        "n_friction_params": n_fric,
        "n_augmented_params": expected,
        "gravity": np.asarray(_GRAVITY_FROM_MODEL(), dtype=float).tolist(),
        "residual": None if residual is None else float(residual),
        "feasibility_method": (
            None if feasibility_method is None else str(feasibility_method)
        ),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "parameters.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.DEFAULT_PROTOCOL)
    return path


# ---------------------------------------------------------------------------
# Source emission
# ---------------------------------------------------------------------------

def _emit_ne_source(model: RegressorModel) -> str:
    kin = model.kin
    n = model.nDoF
    common_pi_var = sympy.Symbol("q")  # single-scalar arg in per-joint funcs

    per_joint_blocks: list[str] = []
    for i in range(n):
        lk = kin.link_kin[i]
        T_sub = lk.T_sym.subs(lk.q_sym, common_pi_var)
        aJ_sub = lk.aJ_sym.subs(lk.q_sym, common_pi_var)
        pR_sub = lk.pR_sym.subs(lk.q_sym, common_pi_var)
        paJ_sub = lk.paJ_sym.subs(lk.q_sym, common_pi_var)
        per_joint_blocks.append(_emit_matrix_fn(f"_T_{i}", "q", T_sub, 4, 4))
        per_joint_blocks.append(_emit_matrix_fn(f"_aJ_{i}", "q", aJ_sub, 3, 1))
        per_joint_blocks.append(_emit_matrix_fn(f"_pR_{i}", "q", pR_sub, 3, 3))
        per_joint_blocks.append(_emit_matrix_fn(f"_paJ_{i}", "q", paJ_sub, 3, 1))

    func_lists = (
        "_T_FUNCS = [" + ", ".join(f"_T_{i}" for i in range(n)) + "]\n"
        "_AJ_FUNCS = [" + ", ".join(f"_aJ_{i}" for i in range(n)) + "]\n"
        "_PR_FUNCS = [" + ", ".join(f"_pR_{i}" for i in range(n)) + "]\n"
        "_PAJ_FUNCS = [" + ", ".join(f"_paJ_{i}" for i in range(n)) + "]\n"
    )

    return "".join([
        _emit_header(model),
        _emit_ne_constants(model),
        _emit_skew_helper(),
        "\n".join(per_joint_blocks),
        "\n\n",
        func_lists,
        "\n",
        _NE_CORE_SOURCE,
        "\n",
        _emit_friction_block_source(),
        "\n",
        _emit_public_api(),
    ])


def _emit_el_source(model: RegressorModel) -> str:
    kin = model.kin
    n = model.nDoF

    cache_dir = model._el_cache_dir()
    cache_path = Path(cache_dir) / "el_regressor_cache.pkl"
    if not cache_path.exists():
        # Force the EL symbolic build to populate the cache.
        euler_lagrange_regressor_builder(kin, cache_dir)
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    Y_sym = data["Y_sym"]
    kept_cols = list(data["kept_cols"])

    q_alias = {sympy.Symbol(f"q{i + 1}"): sympy.Symbol(f"q_{i}") for i in range(n)}
    dq_alias = {sympy.Symbol(f"dq{i + 1}"): sympy.Symbol(f"dq_{i}") for i in range(n)}
    ddq_alias = {sympy.Symbol(f"ddq{i + 1}"): sympy.Symbol(f"ddq_{i}") for i in range(n)}
    aliases: dict[sympy.Symbol, sympy.Symbol] = {}
    aliases.update(q_alias)
    aliases.update(dq_alias)
    aliases.update(ddq_alias)

    n_red = Y_sym.shape[1]
    body_lines: list[str] = []
    body_lines.append("    Y = numpy.zeros((_NDOF, _N_RED_COLS))")
    for r in range(n):
        for c in range(n_red):
            entry = Y_sym[r, c]
            if entry == 0 or entry is sympy.S.Zero:
                continue
            substituted = entry.xreplace(aliases)
            expr = _PRINTER.doprint(substituted)
            body_lines.append(f"    Y[{r}, {c}] = {expr}")
    body_lines.append("    return Y")
    reduced_body = "\n".join(body_lines)

    locals_block = []
    for i in range(n):
        locals_block.append(f"    q_{i} = q[{i}]")
    for i in range(n):
        locals_block.append(f"    dq_{i} = dq[{i}]")
    for i in range(n):
        locals_block.append(f"    ddq_{i} = ddq[{i}]")
    locals_text = "\n".join(locals_block)

    el_section = textwrap.dedent(
        """
        _KEPT_COLS = {kept_cols!r}
        _N_RED_COLS = {n_red}


        def _Y_reduced(q, dq, ddq):
        {locals_text}
        {reduced_body}


        def _Y_rigid_impl(q, dq, ddq):
            q = numpy.asarray(q, dtype=float).reshape(-1)
            dq = numpy.asarray(dq, dtype=float).reshape(-1)
            ddq = numpy.asarray(ddq, dtype=float).reshape(-1)
            Y_red = _Y_reduced(q, dq, ddq)
            Y_full = numpy.zeros((_NDOF, _N_RIGID_PARAMS))
            Y_full[:, _KEPT_COLS] = Y_red
            return Y_full
        """
    ).format(
        kept_cols=kept_cols,
        n_red=n_red,
        locals_text=locals_text,
        reduced_body=reduced_body,
    )

    return "".join([
        _emit_header(model),
        _emit_el_constants(model),
        _emit_skew_helper(),
        el_section,
        "\n",
        _emit_friction_block_source(),
        "\n",
        _emit_public_api(el_backend=True),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _GRAVITY_FROM_MODEL() -> np.ndarray:
    """Return the GRAVITY constant (kept local to avoid src.* leakage)."""
    from .math_utils import GRAVITY
    return np.asarray(GRAVITY, dtype=float)


def _emit_header(model: RegressorModel) -> str:
    meta_dict = {
        "backend": model.backend,
        "friction_model": model.friction_model,
        "nDoF": model.nDoF,
        "joint_names": model.joint_names(),
        "link_names": model.link_names(),
        "rigid_parameter_names": model.rigid_parameter_names(),
        "friction_parameter_names": model.friction_parameter_names(),
        "augmented_parameter_names": model.augmented_parameter_names(),
        "n_rigid_params": model.n_rigid_params,
        "n_friction_params": model.n_friction_params,
        "n_augmented_params": model.n_augmented_params,
        "gravity": _GRAVITY_FROM_MODEL().tolist(),
    }
    meta_literal = json.dumps(meta_dict, indent=4)

    header = textwrap.dedent(
        '''\
        """Standalone callable regressor for inverse dynamics.

        Public API
        ----------
        Y_rigid(q, dq, ddq)
            Rigid-body regressor, shape (nDoF, 10*nDoF).
        Y_augmented(q, dq, ddq)
            Rigid + friction regressor, shape (nDoF, 10*nDoF + n_friction).
        Y(q, dq, ddq)
            Alias for Y_augmented.
        tau(q, dq, ddq, pi=None)
            Inverse dynamics torques. ``pi`` may be either the rigid
            (length 10*nDoF) or the augmented vector (length
            10*nDoF + n_friction); the regressor is auto-selected from
            ``pi.size``. When ``pi`` is None the bundled
            ``parameters.pkl`` next to this file is loaded automatically.
        Y_stack(q_traj, dq_traj, ddq_traj)
            Vertically stacked observation matrix for a (N, nDoF) batch,
            shape (N*nDoF, 10*nDoF + n_friction).
        tau_traj(q_traj, dq_traj, ddq_traj, pi=None)
            Per-sample torques for a (N, nDoF) batch, shape (N, nDoF).
        META : dict
            Backend, friction model, joint/link names, parameter names,
            and gravity vector.

        Shape conventions
        -----------------
        Single state:  q, dq, ddq are (nDoF,).
        Batched:       q, dq, ddq are (N, nDoF) or (nDoF, N).

        Runtime dependencies: numpy only.
        """
        from __future__ import annotations

        import pickle
        from pathlib import Path

        import numpy

        '''
    )

    return header + f"META = {meta_literal}\n\n"


def _emit_ne_constants(model: RegressorModel) -> str:
    kin = model.kin
    n = model.nDoF
    gravity = _GRAVITY_FROM_MODEL().tolist()
    tw_0 = np.asarray(kin.Tw_0, dtype=float).tolist()
    pi_next = [np.asarray(kin.pi_i_raw[i], dtype=float).flatten().tolist()
               for i in range(n)]
    trs = [list(map(int, kin.torque_row_sign[i])) for i in range(n)]
    return textwrap.dedent(
        f"""
        _NDOF = {n}
        _N_RIGID_PARAMS = {10 * n}
        _N_FRICTION_PARAMS = {model.n_friction_params}
        _FRICTION_MODEL = {model.friction_model!r}
        _GRAVITY = numpy.array({gravity!r}, dtype=float)
        _TW_0 = numpy.array({tw_0!r}, dtype=float)
        _PI_NEXT = [numpy.array(v, dtype=float) for v in {pi_next!r}]
        _TORQUE_ROW_SIGN = {trs!r}

        """
    )


def _emit_el_constants(model: RegressorModel) -> str:
    n = model.nDoF
    gravity = _GRAVITY_FROM_MODEL().tolist()
    return textwrap.dedent(
        f"""
        _NDOF = {n}
        _N_RIGID_PARAMS = {10 * n}
        _N_FRICTION_PARAMS = {model.n_friction_params}
        _FRICTION_MODEL = {model.friction_model!r}
        _GRAVITY = numpy.array({gravity!r}, dtype=float)

        """
    )


def _emit_skew_helper() -> str:
    return textwrap.dedent(
        """
        def _skew(v):
            v = numpy.asarray(v, dtype=float).flatten()
            return numpy.array([
                [0.0,  -v[2],  v[1]],
                [v[2],  0.0,  -v[0]],
                [-v[1], v[0],  0.0],
            ])

        """
    )


def _emit_matrix_fn(name: str, arg: str, expr: sympy.Matrix,
                    rows: int, cols: int) -> str:
    """Emit ``def name(arg): return numpy.array([[...], ...])``."""
    lines = []
    for r in range(rows):
        row_cells = [_PRINTER.doprint(expr[r, c]) for c in range(cols)]
        lines.append("[" + ", ".join(row_cells) + "]")
    body = ",\n        ".join(lines)
    return textwrap.dedent(
        f"""
        def {name}({arg}):
            return numpy.array([
                {body},
            ], dtype=float)
        """
    )


def _emit_friction_block_source() -> str:
    return textwrap.dedent(
        """
        def _friction_block(dq, model):
            dq = numpy.asarray(dq, dtype=float).reshape(-1)
            n = dq.size
            if model == "none":
                return numpy.zeros((n, 0))
            blocks = []
            if model in ("viscous", "viscous_coulomb"):
                blocks.append(numpy.diag(dq))
            if model in ("coulomb", "viscous_coulomb"):
                a, b = 10.0, 1000.0
                blocks.append(numpy.diag(1.0 / (1.0 + numpy.exp(a - b * dq))))
                blocks.append(numpy.diag(-1.0 / (1.0 + numpy.exp(a + b * dq))))
            return numpy.hstack(blocks)

        """
    )


def _emit_public_api(el_backend: bool = False) -> str:
    if el_backend:
        y_rigid_def = textwrap.dedent(
            """
            def Y_rigid(q, dq, ddq):
                q, dq, ddq = _coerce_single(q, dq, ddq)
                return _Y_rigid_impl(q, dq, ddq)
            """
        )
    else:
        y_rigid_def = textwrap.dedent(
            """
            def Y_rigid(q, dq, ddq):
                q, dq, ddq = _coerce_single(q, dq, ddq)
                return _newton_euler_regressor(q, dq, ddq)
            """
        )

    common = textwrap.dedent(
        '''
        def _coerce_single(q, dq, ddq):
            q = numpy.asarray(q, dtype=float).reshape(-1)
            dq = numpy.asarray(dq, dtype=float).reshape(-1)
            ddq = numpy.asarray(ddq, dtype=float).reshape(-1)
            for name, value in (("q", q), ("dq", dq), ("ddq", ddq)):
                if value.size != _NDOF:
                    raise ValueError(
                        f"{name} must have {_NDOF} entries, got {value.size}."
                    )
            return q, dq, ddq


        def _coerce_batch(q, dq, ddq):
            def _to_2d(name, value):
                arr = numpy.asarray(value, dtype=float)
                if arr.ndim == 1:
                    if arr.size != _NDOF:
                        raise ValueError(
                            f"{name} must have {_NDOF} entries, got {arr.size}."
                        )
                    return arr.reshape(1, _NDOF)
                if arr.ndim == 2:
                    if arr.shape[1] == _NDOF:
                        return arr
                    if arr.shape[0] == _NDOF:
                        return arr.T
                raise ValueError(
                    f"{name} must be shape ({_NDOF},), (N, {_NDOF}), "
                    f"or ({_NDOF}, N); got {arr.shape}."
                )
            return _to_2d("q", q), _to_2d("dq", dq), _to_2d("ddq", ddq)


        def Y_augmented(q, dq, ddq):
            Yr = Y_rigid(q, dq, ddq)
            if _FRICTION_MODEL == "none":
                return Yr
            _, dq_v, _ = _coerce_single(q, dq, ddq)
            return numpy.hstack((Yr, _friction_block(dq_v, _FRICTION_MODEL)))


        def Y(q, dq, ddq):
            return Y_augmented(q, dq, ddq)


        _DEFAULT_PI_CACHE = {"value": None}


        def _load_default_pi():
            if _DEFAULT_PI_CACHE["value"] is not None:
                return _DEFAULT_PI_CACHE["value"]
            pkl_path = Path(__file__).with_name("parameters.pkl")
            if not pkl_path.exists():
                raise FileNotFoundError(
                    f"No pi provided and parameters.pkl not found next to "
                    f"{Path(__file__).name} at {pkl_path}."
                )
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            pi = numpy.asarray(data["pi_augmented"], dtype=float).reshape(-1)
            _DEFAULT_PI_CACHE["value"] = pi
            return pi


        def tau(q, dq, ddq, pi=None):
            if pi is None:
                pi = _load_default_pi()
            pi = numpy.asarray(pi, dtype=float).reshape(-1)
            if pi.size == _N_RIGID_PARAMS:
                return Y_rigid(q, dq, ddq) @ pi
            if pi.size == _N_RIGID_PARAMS + _N_FRICTION_PARAMS:
                return Y_augmented(q, dq, ddq) @ pi
            raise ValueError(
                f"pi has length {pi.size}; expected {_N_RIGID_PARAMS} "
                f"(rigid) or {_N_RIGID_PARAMS + _N_FRICTION_PARAMS} (augmented)."
            )


        def Y_stack(q_traj, dq_traj, ddq_traj):
            q_b, dq_b, ddq_b = _coerce_batch(q_traj, dq_traj, ddq_traj)
            N = q_b.shape[0]
            return numpy.vstack([
                Y_augmented(q_b[k], dq_b[k], ddq_b[k]) for k in range(N)
            ])


        def tau_traj(q_traj, dq_traj, ddq_traj, pi=None):
            if pi is None:
                pi = _load_default_pi()
            pi = numpy.asarray(pi, dtype=float).reshape(-1)
            q_b, dq_b, ddq_b = _coerce_batch(q_traj, dq_traj, ddq_traj)
            N = q_b.shape[0]
            out = numpy.zeros((N, _NDOF))
            for k in range(N):
                out[k, :] = tau(q_b[k], dq_b[k], ddq_b[k], pi=pi)
            return out
        '''
    )

    return y_rigid_def + common


# ---------------------------------------------------------------------------
# Transcribed Newton-Euler core (string, embedded verbatim into the export)
# ---------------------------------------------------------------------------

_NE_CORE_SOURCE = textwrap.dedent(
    """
    def _adjoint(T):
        R = T[:3, :3]
        d = T[:3, 3]
        S = _skew(d)
        top = numpy.hstack((R, numpy.zeros((3, 3))))
        bot = numpy.hstack((S @ R, R))
        return numpy.vstack((top, bot))


    def _link_regressor(p, dv, dw, w):
        p = numpy.asarray(p, dtype=float).flatten()
        dv = numpy.asarray(dv, dtype=float).reshape(3, 1)
        dw = numpy.asarray(dw, dtype=float).flatten()
        w = numpy.asarray(w, dtype=float).flatten()

        Y_m = numpy.vstack((dv, _skew(p) @ dv))

        Sdw = _skew(dw)
        Sw = _skew(w)
        Y_mp_t = Sdw + Sw @ Sw
        Y_mp_r = (_skew(p) @ Sdw
                  + _skew(p) @ Sw @ Sw
                  - _skew(dv.flatten()))
        Y_mp = numpy.vstack((Y_mp_t, Y_mp_r))

        dwx, dwy, dwz = dw
        wx, wy, wz = w
        Y_I = numpy.zeros((6, 6))
        Y_I[3:, :] = numpy.array([
            [dwx,    dwy - wx*wz,  dwz + wx*wy, -wy*wz,       wy**2 - wz**2,  wy*wz],
            [wx*wz,  dwx + wy*wz, -wx**2 + wz**2, dwy,         dwz - wx*wy,  -wx*wz],
            [-wx*wy, wx**2 - wy**2, dwx - wy*wz,  wx*wy,       dwy + wx*wz,   dwz],
        ])

        return numpy.hstack((Y_m, Y_mp, Y_I))


    def _newton_euler_regressor(q, dq, ddq):
        n = _NDOF
        g = _GRAVITY.reshape(3, 1)

        w = numpy.zeros((3, n))
        dw = numpy.zeros((3, n))
        dv_origin = numpy.zeros((3, n))
        dv_transport = numpy.zeros((3, n))
        A = numpy.zeros((6, 6, max(n - 1, 1)))
        reg = numpy.zeros((6, 10, n))

        p_zero = numpy.zeros((3, 1))

        for i in range(n):
            Ti = numpy.asarray(_T_FUNCS[i](q[i]), dtype=float).reshape(4, 4)
            Ri = Ti[:3, :3]
            aJ = numpy.asarray(_AJ_FUNCS[i](q[i]), dtype=float).reshape(3, 1)
            pR = numpy.asarray(_PR_FUNCS[i](q[i]), dtype=float).reshape(3, 3)
            paJ = numpy.asarray(_PAJ_FUNCS[i](q[i]), dtype=float).reshape(3, 1)
            p = numpy.asarray(_PI_NEXT[i], dtype=float).reshape(3, 1)

            if i == 0:
                w[:, 0] = (Ri.T @ (dq[0] * aJ)).flatten()
                dw[:, 0] = (
                    dq[i]**2 * (pR.T @ aJ + Ri.T @ paJ)
                    + ddq[i] * Ri.T @ aJ
                ).flatten()
                dv_origin[:, 0] = (
                    Ri.T @ _TW_0[:3, :3].T @ (-g)
                ).flatten()
                dv_transport[:, 0] = (
                    dv_origin[:, 0].reshape(3, 1)
                    + _skew(dw[:, 0]) @ p
                    + _skew(w[:, 0]) @ (_skew(w[:, 0]) @ p)
                ).flatten()
            else:
                w[:, i] = (
                    Ri.T @ (w[:, i-1].reshape(3, 1) + dq[i] * aJ)
                ).flatten()
                dw[:, i] = (
                    Ri.T @ dw[:, i-1].reshape(3, 1)
                    + dq[i] * pR.T @ w[:, i-1].reshape(3, 1)
                    + dq[i]**2 * (pR.T @ aJ + Ri.T @ paJ)
                    + ddq[i] * Ri.T @ aJ
                ).flatten()
                dv_origin[:, i] = (
                    Ri.T @ dv_transport[:, i-1].reshape(3, 1)
                ).flatten()
                dv_transport[:, i] = (
                    dv_origin[:, i].reshape(3, 1)
                    + _skew(dw[:, i]) @ p
                    + _skew(w[:, i]) @ (_skew(w[:, i]) @ p)
                ).flatten()
                A[:, :, i-1] = _adjoint(Ti)

            reg[:, :, i] = _link_regressor(
                p_zero, dv_origin[:, i], dw[:, i], w[:, i]
            )

        Y = numpy.zeros((n, 10 * n))
        for rw in range(n):
            row_idx, sgn = _TORQUE_ROW_SIGN[rw]
            row_block = numpy.zeros((6, 0))
            for cl in range(n):
                if cl < rw:
                    row_block = numpy.hstack((row_block, numpy.zeros((6, 10))))
                elif cl == rw:
                    row_block = numpy.hstack((row_block, reg[:, :, cl]))
                else:
                    tmp = numpy.eye(6)
                    for j in range(rw, cl):
                        tmp = tmp @ A[:, :, j]
                    row_block = numpy.hstack((row_block, tmp @ reg[:, :, cl]))
            Y[rw, :] = sgn * row_block[row_idx, :]

        return Y
    """
)
