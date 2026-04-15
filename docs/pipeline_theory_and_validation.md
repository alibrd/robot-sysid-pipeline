# `sysid_pipeline`: Theory, Mathematics, and Verification

This document explains the standalone `sysid_pipeline` package from Stage 0 to the final outputs. The goal is to make the pipeline understandable, auditable, and scientifically grounded without relying on any code or history outside [`sysid_pipeline`](..).

The central modeling assumption is the standard inverse-dynamics identification model

$$
\tau(q,\dot q,\ddot q) = Y(q,\dot q,\ddot q)\,\pi,
$$

where $\tau$ is the joint-torque vector, $Y$ is the regressor, and $\pi$ is the inertial-parameter vector. This linear-in-parameters formulation is the foundation of classical robot identification, including Atkeson, An, and Hollerbach (1986), Gautier (1991), Gautier and Khalil (1992), Swevers et al. (1997), and Khalil and Dombre (2002, Ch. 13).

## Stage 0. Scope, Notation, and Supported Features

### Notation
- $n$: number of joints / identified links.
- $q,\dot q,\ddot q \in \mathbb{R}^n$: joint position, velocity, acceleration.
- $Y_k \in \mathbb{R}^{n \times p}$: single-sample regressor at sample $k$.
- $W = \begin{bmatrix} Y_1 \\ \cdots \\ Y_N \end{bmatrix}$: stacked observation matrix.
- $\pi_i = [m_i,\ h_{x,i},\ h_{y,i},\ h_{z,i},\ I_{xx,i},\ I_{xy,i},\ I_{xz,i},\ I_{yy,i},\ I_{yz,i},\ I_{zz,i}]^\top$: 10-parameter rigid-body block for link $i$.
- $\pi = [\pi_1^\top,\ldots,\pi_n^\top]^\top$: full rigid-body parameter vector.
- $h_i = m_i c_i$: first moment of mass, with center of mass $c_i$.

### What this standalone package supports

| Capability | Current status |
|---|---|
| URDF / XACRO parsing | Supported |
| Newton-Euler regressor | Supported |
| Euler-Lagrange regressor | Supported |
| Harmonic excitation with `cosine`, `sine`, `both` | Supported |
| `legacy_excTrajGen`, `urdf_reference`, `literature_standard` excitation styles | Supported |
| Friction augmentation (`none`, `viscous`, `coulomb`, `viscous_coulomb`) | Supported |
| Zero-phase Butterworth filtering | Supported |
| Downsampling after filtering | Supported |
| QR-based base-parameter reduction | Supported |
| OLS / WLS / bounded least squares | Supported |
| Pseudo-inertia feasibility check | Supported |
| Constrained identification with pseudo-inertia PSD (LMI) | Supported for `newton_euler` only |
| Cholesky-factored feasibility reparameterization | Supported for `newton_euler` only |
| Cartesian / workspace / torque excitation constraints | **Not implemented** |
| Automatic differentiation from raw $q$ to $\dot q,\ddot q$ | **Not implemented**; external data must provide `dq` and `ddq` |

### Standalone evidence
- Internal URDF fixtures live in [`tests/assets`](../tests/assets), so the verification suite no longer depends on sibling directories.
- The default verification suite is in [`tests/test_pipeline_theory.py`](../tests/test_pipeline_theory.py).
- The optional deeper verification suite is in [`tests/test_pipeline_theory_slow.py`](../tests/test_pipeline_theory_slow.py) and is enabled with `--run-slow`.

## Stage 1. Parse the URDF/XACRO and Extract a Serial Chain

### Governing equations
Each URDF joint contributes a homogeneous transform

$$
{}^{i-1}T_i =
\begin{bmatrix}
{}^{i-1}R_i & {}^{i-1}p_i \\
0 & 1
\end{bmatrix},
$$

with rotation and translation coming from the joint origin and axis. Serial-chain kinematics are then built by composition,

$$
{}^wT_i = {}^wT_0 \, {}^0T_1 \cdots {}^{i-1}T_i.
$$

This is standard rigid-body kinematics, as used throughout robot modeling texts such as Khalil and Dombre (2002).

### Code path
- URDF / XACRO parsing: [`src/urdf_parser.py`](../src/urdf_parser.py)
- Serial-chain extraction and `Tw_0` construction: `parse_urdf()`

The parser:
- resolves `.xacro` files through the `xacro` CLI,
- reads links, joints, inertial blocks, and limits,
- extracts one serial chain by topology walk,
- accumulates fixed transforms before the first revolute joint into `Tw_0`.

### Verification evidence
- Standalone-chain extraction and parameter-vector setup: `test_stage_1_and_3_parser_and_kinematics_build_standalone_model`
- Existing chain-order and `Tw_0` checks: [`tests/test_pipeline.py`](../tests/test_pipeline.py)

## Stage 2. Resolve Joint Limits and Fail Early if They Are Missing

### Governing rule
The excitation stage requires bounds on $q$, $\dot q$, and $\ddot q$. If those limits are not present in the URDF/XACRO, the pipeline requires them from JSON and raises an error otherwise.

This is an implementation policy rather than a literature equation, but it is the correct contract for a safe excitation optimizer.

### Code path
- Joint-limit extraction: `extract_joint_limits()` in [`src/urdf_parser.py`](../src/urdf_parser.py)

The function applies this precedence:
1. JSON override, if supplied.
2. URDF limit, if present.
3. Hard failure with a clear error message.

Acceleration limits are always expected from JSON for the supplied standalone fixtures because URDFs typically do not store them.

### Verification evidence
- Missing-limit rejection: `test_stage_2_joint_limit_extraction_rejects_missing_json_overrides`

## Stage 3. Build Kinematics and the Rigid-Body Parameter Vector

### Governing equations
For each link, the inertial parameters are stored as

$$
\pi_i =
\begin{bmatrix}
m_i &
m_i c_{x,i} &
m_i c_{y,i} &
m_i c_{z,i} &
I_{xx,i} &
I_{xy,i} &
I_{xz,i} &
I_{yy,i} &
I_{yz,i} &
I_{zz,i}
\end{bmatrix}^\top.
$$

If the URDF inertia is given in the link's inertial frame, the pipeline rotates it to the link frame and shifts it to the joint frame with the parallel-axis theorem:

$$
I_{O_i} = R_{i,c}\,I_{C_i}\,R_{i,c}^\top + m_i\,S(c_i)^\top S(c_i),
$$

where $S(c_i)$ is the skew-symmetric matrix of the center-of-mass offset. This is standard rigid-body mechanics; see Khalil and Dombre (2002) and classical rigid-body dynamics references.

### Code path
- Kinematics model: [`src/kinematics.py`](../src/kinematics.py)
- Constants and skew matrices: [`src/math_utils.py`](../src/math_utils.py)

`RobotKinematics` constructs:
- symbolic per-joint transforms and Jacobian-related terms,
- the per-link parameter blocks `PI_i`,
- the full stacked parameter vector `PI`.

The gravity constant is hardcoded as $g = 9.80665\ \text{m/s}^2$ in [`src/math_utils.py`](../src/math_utils.py), matching the standard SI gravity constant.

### Verification evidence
- Standalone kinematics and parameter-vector values: `test_stage_1_and_3_parser_and_kinematics_build_standalone_model`
- Existing gravity/kinematics coverage: [`tests/test_pipeline.py`](../tests/test_pipeline.py)

## Stage 4. Build the Newton-Euler Regressor

### Governing equations
The Newton-Euler branch uses recursive body-frame kinematics and builds a regressor that is linear in inertial parameters:

$$
\tau = Y_{NE}(q,\dot q,\ddot q)\,\pi.
$$

The body-frame recursion is of the standard form

$$
\omega_i = R_i^\top \big(\omega_{i-1} + \dot q_i a_i\big),
$$

$$
\dot\omega_i = R_i^\top \dot\omega_{i-1}
+ \dot q_i \,\partial_q R_i^\top \omega_{i-1}
+ \dot q_i^2 \big(\partial_q R_i^\top a_i + R_i^\top \partial_q a_i\big)
+ \ddot q_i R_i^\top a_i,
$$

followed by the usual acceleration transport and backward force propagation. This is the classical recursive inverse-dynamics structure; see Khalil and Dombre (2002), Ch. 9 and Ch. 13.

The link regressor is partitioned into mass, first-moment, and inertia contributions:

$$
Y_i = \begin{bmatrix} Y_m & Y_h & Y_I \end{bmatrix},
$$

which is exactly the linear rigid-body decomposition used in identification literature.

### Code path
- Newton-Euler regressor: [`src/dynamics_newton_euler.py`](../src/dynamics_newton_euler.py)

Key implementation points:
- gravity enters through `GRAVITY`,
- `_adjoint()` uses the child-origin translation embedded in the homogeneous transform,
- `_link_regressor()` builds the $6 \times 10$ per-link linear block,
- `newton_euler_regressor()` assembles the final joint-space regressor row by row.

### Verification evidence
- Shared-torque agreement with Euler-Lagrange on a standalone 1-DoF robot: `test_stage_4_and_5_newton_euler_and_euler_lagrange_match_shared_torques`
- Broader 3-DoF random-state agreement: `test_slow_ne_el_regressors_match_across_random_3dof_states`

## Stage 5. Build the Euler-Lagrange Regressor

### Governing equations
The Euler-Lagrange branch starts from the Lagrangian

$$
L(q,\dot q,\pi) = T(q,\dot q,\pi) - V(q,\pi),
$$

which is also linear in inertial parameters:

$$
L(q,\dot q,\pi) = Y_L(q,\dot q)\,\pi.
$$

The regressor rows are then obtained from the Euler-Lagrange operator applied componentwise:

$$
Y_{EL,j}(q,\dot q,\ddot q)
=
\frac{d}{dt}\left(\frac{\partial Y_L}{\partial \dot q_j}\right)
-
\frac{\partial Y_L}{\partial q_j}.
$$

The pipeline follows the body-frame formulation used in energy-based identification derivations, with

$$
v_i^b = R_{wi}^\top v_i^w,
\qquad
\omega_i^b = R_{wi}^\top \omega_i^w,
$$

which is the correct frame transform for link kinetic energy. See Khalil and Dombre (2002), Ch. 9 and Ch. 13.

### Code path
- Symbolic builder: [`src/dynamics_euler_lagrange.py`](../src/dynamics_euler_lagrange.py)

The implementation:
- rebuilds the symbolic transforms,
- forms $Y_L$ one link at a time,
- applies the Euler-Lagrange differentiation rule in `_differentiate_lagrangian()`,
- removes structurally zero columns (columns whose entries are all literal zero, without invoking symbolic reasoning),
- caches the symbolic regressor to disk and re-lambdifies it on load.

### Verification evidence
- Shared-torque agreement with Newton-Euler: `test_stage_4_and_5_newton_euler_and_euler_lagrange_match_shared_torques`
- Broader 3-DoF random-state agreement: `test_slow_ne_el_regressors_match_across_random_3dof_states`

### Current implementation note
The EL branch returns a reduced symbolic regressor after removing structurally zero columns (i.e. columns that are literal zero in their symbolic form, without invoking symbolic reasoning such as `equals` or `simplify`). Columns that are algebraically zero but not yet simplified are left in place and handled later by the numeric base-parameter reduction in Stage 9. This is mathematically fine for unconstrained identification, but it means the current EL path cannot support the per-link full-parameter feasibility constraints used by the pseudo-inertia method in Stage 11.

## Stage 6. Build the Excitation Trajectory and Choose an Optimization Style

### Governing equations: harmonic parameterization
The code supports three basis families. With $\omega_j = 2\pi f_j$:

**Cosine basis**

$$
q_i(t) = \sum_{j=1}^{m} a_{ij}\big(\cos(\omega_j t)-1\big) + q_{0,i},
$$

$$
\dot q_i(t) = \sum_{j=1}^{m} -a_{ij}\omega_j \sin(\omega_j t),
$$

$$
\ddot q_i(t) = \sum_{j=1}^{m} -a_{ij}\omega_j^2 \cos(\omega_j t).
$$

**Sine basis with endpoint correction**

$$
q_i(t) = \sum_{j=1}^{m} b_{ij}\sin(\omega_j t) + \lambda_{1,i} t + q_{0,i},
\qquad
\lambda_{1,i} = -\sum_{j=1}^{m} b_{ij}\omega_j,
$$

so that $q_i(0)=q_{0,i}$ and $\dot q_i(0)=0$. If the total duration is an integer number of base periods, then $\sin(\omega_j T)=0$ and $\cos(\omega_j T)=1$ for every harmonic, which yields $\dot q_i(T)=0$ and $\ddot q_i(T)=0$.

**Both basis functions**

Without phase optimization:

$$
q_i(t) =
\sum_{j=1}^{m} a_{ij}\cos(\omega_j t)
\sum_{j=1}^{m} b_{ij}\sin(\omega_j t)
\lambda_{1,i}t + \lambda_{0,i} + q_{0,i},
$$

with

$$
\lambda_{1,i} = -\sum_{j=1}^{m} b_{ij}\omega_j,
\qquad
\lambda_{0,i} = -\sum_{j=1}^{m} a_{ij}.
$$

With phase optimization:

$$
q_i(t) =
\sum_{j=1}^{m} a_{ij}\cos(\omega_j t + \phi_{ij})
 + \lambda_{1,i}t + \lambda_{0,i} + q_{0,i},
$$

with

$$
\lambda_{1,i} = \sum_{j=1}^{m} a_{ij}\omega_j \sin(\phi_{ij}),
\qquad
\lambda_{0,i} = -\sum_{j=1}^{m} a_{ij}\cos(\phi_{ij}).
$$

Harmonic excitation design is standard in robot identification; see Gautier and Khalil (1992) and Swevers et al. (1997).

### Governing equations: excitation objectives
The literature-standard conditioning objective is based on the observation matrix. Classical criteria include minimizing

$$
\kappa_2(W^\top W)
$$

or the equivalent singular-value condition metric on $W$ itself. Because

$$
\kappa_2(W^\top W) = \kappa_2(W)^2,
$$

minimizing $\kappa_2(W)$ is monotonic-equivalent to minimizing $\kappa_2(W^\top W)$ whenever $W$ has nonzero singular values. This is the basis of the code’s `_condition_cost()` and `_condition_cost_base()` implementations. See Gautier and Khalil (1992) and Swevers et al. (1997).

When condition-number optimization is disabled, the current code uses the fallback amplitude objective

$$
J_{\text{amp}} = -\left(\|\dot q\|_1 + \|\ddot q\|_1\right),
$$

which is an engineering heuristic for large excitation subject to limits, not a direct optimal-design criterion from the identification literature.

### Constraint styles currently implemented

1. **`legacy_excTrajGen`**
   - Compatibility mode for the old heuristic.
   - Uses
     $$
     J_{\text{legacy}} =
     \sum \frac{1}{\varepsilon + |q|}
     + 100 \sum \frac{1}{\varepsilon + |\dot q|},
     $$
     plus soft exponential penalties.
   - Uses differential evolution.

2. **`urdf_reference`**
   - Uses condition-number objective on the full stacked regressor when enabled.
   - Uses the implemented two-way sigmoid penalty
     $$
     c(x) = \frac{2}{1+e^{\alpha(hi-x)}} - \frac{2}{1+e^{-\alpha(lo-x)}}
     $$
     on sampled $q$, $\dot q$, $\ddot q$.
   - Uses differential evolution.

3. **`literature_standard`**
   - Uses condition number on the **base-parameter** observation matrix when enabled.
   - Uses sampled hard inequality constraints on $q$, $\dot q$, $\ddot q$ through SLSQP.
   - Uses the base-parameter observation matrix, which is aligned with the standard identification workflow in Gautier (1991) and Swevers et al. (1997).

### Code path
- Harmonic trajectory generation: [`src/trajectory.py`](../src/trajectory.py)
- Excitation optimization: [`src/excitation.py`](../src/excitation.py)

### Verification evidence
- Sine boundary-condition proof by direct evaluation: `test_stage_6_sine_basis_enforces_boundary_conditions_on_integer_periods`
- Config guard for noninteger sine periods: `test_stage_6_noninteger_sine_periods_are_rejected_by_config`
- Solver-path separation between the three styles: `test_stage_6_excitation_styles_dispatch_to_distinct_solver_paths`
- Slower literature-standard run with base-condition objective: `test_slow_literature_standard_excitation_returns_finite_conditioning_result`

### Current implementation note
- `legacy_excTrajGen` is intentionally kept as a compatibility mode and is **not** a literature-optimal excitation objective.
- `urdf_reference` and `literature_standard` only enforce joint-space constraints; optional Cartesian/workspace/torque constraints are not implemented.
- The code exposes `optimize_phase` only for `basis_functions="both"`.

## Stage 7. Generate Synthetic Data or Load External Data

### Governing equations
For synthetic identification data, the pipeline uses the same linear inverse-dynamics model:

$$
\tau_k = Y_k \pi,
$$

or, with friction augmentation,

$$
\tau_k = \begin{bmatrix} Y_k & Y_{f,k} \end{bmatrix}
\begin{bmatrix} \pi \\ \pi_f \end{bmatrix}.
$$

The friction blocks are

$$
Y_v = \operatorname{diag}(\dot q),
$$

$$
Y_{cp} = \operatorname{diag}\left(\frac{1}{1+e^{a-b\dot q_i}}\right),
\qquad
Y_{cn} = \operatorname{diag}\left(\frac{-1}{1+e^{a+b\dot q_i}}\right),
$$

matching the smooth Coulomb-friction augmentation commonly used in inverse-dynamics identification.

### Code path
- Main orchestration: [`src/pipeline.py`](../src/pipeline.py)
- Friction augmentation: [`src/friction.py`](../src/friction.py)

If `identification.data_file` is absent, the pipeline synthesizes `q`, `dq`, `ddq` from the optimized excitation trajectory and computes torques from the chosen regressor. If `data_file` is present, the pipeline expects `q`, `dq`, `ddq`, `tau`, and optionally `fs` to already exist in the `.npz`.

### Verification evidence
- Friction-regressor block checks: `test_stage_6_friction_regressor_augmentation_matches_supported_models`

### Current implementation note
The pipeline does **not** currently estimate $\dot q$ or $\ddot q$ from position measurements. If external data is used, those signals must already be available.

## Stage 8. Filter Signals, Downsample Them, and Build the Stacked Observation Matrix

### Governing equations
The identification equations are stacked over $N$ samples as

$$
W =
\begin{bmatrix}
Y(q_1,\dot q_1,\ddot q_1) \\
Y(q_2,\dot q_2,\ddot q_2) \\
\vdots \\
Y(q_N,\dot q_N,\ddot q_N)
\end{bmatrix},
\qquad
\tau_{\text{vec}} =
\begin{bmatrix}
\tau_1 \\
\tau_2 \\
\vdots \\
\tau_N
\end{bmatrix}.
$$

This is the standard observation-matrix construction used throughout robot identification (Atkeson, An, and Hollerbach, 1986; Swevers et al., 1997; Khalil and Dombre, 2002, Ch. 13).

Filtering is currently a zero-phase Butterworth low-pass filter applied independently to every channel:

$$
\tilde x = \operatorname{filtfilt}\big(\operatorname{Butterworth}(x)\big),
$$

and downsampling is then performed by sampled index selection after filtering.

### Code path
- Filtering: [`src/filtering.py`](../src/filtering.py)
- Observation-matrix assembly: [`src/observation_matrix.py`](../src/observation_matrix.py)

The implementation:
- filters `q`, `dq`, `ddq`, and `tau` with the same filter settings,
- downsamples after filtering,
- stacks the resulting regressors row-wise,
- raises an error if the number of equations is smaller than the number of unknowns,
- warns when overdetermination is low.

### Verification evidence
- Manual stack equivalence: `test_stage_8_observation_matrix_matches_manual_stacking_equation`
- Filtering-before-downsampling proof: `test_stage_8_filtering_happens_before_downsampling`
- Existing sample-sufficiency rejection: [`tests/test_pipeline.py`](../tests/test_pipeline.py)

## Stage 9. Reduce the Full Model to Base Parameters

### Governing equations
Robot inertial regressors are usually rank-deficient; only the base parameters are identifiable from data. Following Gautier (1991), the pipeline reduces the stacked matrix in two numerical steps:

1. Remove exact zero columns from $W$.
2. Apply column-pivoted QR to the remaining matrix:
   $$
   W_{nz}\Pi = QR.
   $$

If the QR rank is $r$, partition the upper-triangular factor as

$$
R =
\begin{bmatrix}
R_{11} & R_{12} \\
0 & 0
\end{bmatrix},
$$

with $R_{11} \in \mathbb{R}^{r \times r}$. Then

$$
\beta = R_{11}^{-1}R_{12}
$$

defines how the dependent columns are merged into the base coordinates. The code stores that regrouping in a matrix $P$ such that

$$
\pi_b = P\pi,
\qquad
W\pi = W_b\pi_b.
$$

This is precisely the base-parameter identification idea advocated by Gautier (1991) and Khalil and Dombre (2002, Ch. 13.4).

### Code path
- Base reduction: [`src/base_parameters.py`](../src/base_parameters.py)

The same reduction routine is used for Newton-Euler and Euler-Lagrange observation matrices.

### Verification evidence
- NE and EL observation-equation preservation: `test_stage_9_base_parameter_reduction_preserves_observation_equation_for_ne_and_el`
- Stronger random 3-DoF preservation: `test_slow_base_parameter_reduction_preserves_multiple_random_3dof_observation_matrices`

## Stage 10. Solve the Identification Problem

### Governing equations
The standard least-squares identification problem is

$$
\hat \pi = \arg\min_{\pi}\|W\pi - \tau_{\text{vec}}\|_2^2.
$$

If $W$ has full column rank, the ordinary least-squares solution is

$$
\hat \pi_{OLS} = (W^\top W)^{-1}W^\top \tau_{\text{vec}},
$$

or more generally the pseudoinverse solution. This is the classical identification estimator discussed by Atkeson, An, and Hollerbach (1986) and Khalil and Dombre (2002, Ch. 13).

Weighted least squares solves

$$
\hat \pi_{WLS} =
\arg\min_{\pi}
(W\pi-\tau)^\top \Omega (W\pi-\tau),
$$

with solution

$$
\hat \pi_{WLS} = (W^\top \Omega W)^{-1}W^\top \Omega \tau.
$$

Bounded least squares solves the same quadratic objective subject to lower and upper parameter bounds.

### Code path
- Solvers: [`src/solver.py`](../src/solver.py)
- Pipeline integration and automatic bounds selection: [`src/pipeline.py`](../src/pipeline.py)

Current solver behavior:
- `ols`: `np.linalg.lstsq`
- `wls`: one IRLS-style weighting step from OLS residuals
- `bounded_ls`: `scipy.optimize.lsq_linear`
- `parameter_bounds=true`: auto-generates $\pm 50\%$ style bounds around the current base-parameter magnitudes and switches `ols` to `bounded_ls`

### Verification evidence
- Automatic bounded-LS switching: `test_stage_10_parameter_bounds_enable_bounded_ls`

### Current implementation note
When `feasibility_method="lmi"` is requested, the pipeline does **not** stay in the reduced base-parameter space. It lifts the problem back to a full-space constrained least-squares problem to apply per-link pseudo-inertia constraints. That is a deliberate implementation choice in the current code.

## Stage 11. Check or Enforce Physical Feasibility

### Governing equations
For each link block $\pi_i$, define the pseudo-inertia matrix

$$
J_i =
\begin{bmatrix}
\Sigma_i & h_i \\
h_i^\top & m_i
\end{bmatrix},
\qquad
\Sigma_i = \frac{1}{2}\operatorname{tr}(I_i)I_3 - I_i.
$$

The physically consistent rigid-body criterion is

$$
J_i \succeq 0.
$$

This criterion implies positive mass, a positive-semidefinite inertia tensor, and the triangle inequalities on the principal moments. See Sousa and Cortesão (2014) and Wensing, Kim, and Slotine (2018).

The constrained identification problem used by the Newton-Euler branch has two solver paths.

**LMI path** (`feasibility_method="lmi"`): explicit eigenvalue inequality constraints solved via SLSQP:

$$
\min_{\pi_{full}} \frac{1}{2}\|W_b P \pi_{full} - \tau\|_2^2
\quad \text{subject to} \quad
\lambda_{\min}(J_i(\pi_{full})) \ge \varepsilon
\ \forall i.
$$

**Cholesky path** (`feasibility_method="cholesky"`): reparameterises each link's pseudo-inertia as $J_i = L_i L_i^\top$ with $L_i$ lower-triangular, which guarantees $J_i \succeq 0$ by construction. The optimisation is unconstrained in L-space (L-BFGS-B with box bounds $L_{ii} \ge \varepsilon$ for strict positive-definiteness):

$$
\min_{L_1,\ldots,L_n,\,\pi_{extra}} \frac{1}{2}\|W_b P \,\pi_{full}(L_1,\ldots,L_n,\pi_{extra}) - \tau\|_2^2,
$$

where $\pi_{full}$ is reconstructed from each link's $J_i = L_i L_i^\top$ via the standard pseudo-inertia extraction. The analytical gradient is computed through the chain $L_{\text{vec}} \to L \to J = LL^\top \to \pi_i$. This approach follows Traversaro, Brossette, Escande, and Nori (2016).

### Code path
- Feasibility checks, PSD projection, and parameter extraction: [`src/feasibility.py`](../src/feasibility.py)
- Constrained solver (LMI via SLSQP, Cholesky via L-BFGS-B): [`src/solver.py`](../src/solver.py)
- Config validation: [`src/config_loader.py`](../src/config_loader.py)

The code offers:
- post-hoc feasibility diagnostics for any parameter vector with complete 10-parameter link blocks,
- constrained full-space SLSQP with pseudo-inertia eigenvalue constraints (`"lmi"`) for `method="newton_euler"`,
- Cholesky-reparameterised unconstrained L-BFGS-B optimisation (`"cholesky"`) for `method="newton_euler"`,
- PSD projection by eigenvalue clipping when post-hoc correction is requested.

### Verification evidence
- Standard rigid-body failure diagnostics: `test_stage_11_pseudo_inertia_checks_report_standard_rigid_body_failures`
- EL rejection for constrained feasibility: `test_stage_11_euler_lagrange_rejects_constrained_feasibility_modes`
- Constrained NE run that returns a feasible pseudo-inertia model: `test_stage_12_constrained_lmi_returns_feasible_newton_euler_model`
- Pseudo-inertia roundtrip extraction: `test_pi_from_pseudo_inertia_roundtrip`
- Cholesky solver guarantees PSD output: `test_cholesky_solver_guarantees_psd`
- End-to-end Cholesky-constrained pipeline: `test_cholesky_constrained_produces_feasible_model`

### Current implementation note
- `feasibility_method="lmi"` uses SLSQP with explicit eigenvalue inequality constraints on $J_i$.
- `feasibility_method="cholesky"` reparameterises $J_i = L_i L_i^\top$ and uses unconstrained L-BFGS-B with analytical gradients, following Traversaro et al. (2016). This guarantees $J_i \succeq 0$ by construction throughout optimisation.
- The EL branch cannot use either constrained feasibility path because its reduced symbolic regressor cannot be mapped back to full per-link 10-parameter blocks in the current implementation.
- If a reduced parameter vector has fewer than $10n$ entries, `check_feasibility()` cannot perform complete per-link pseudo-inertia checks on missing blocks.

## Stage 12. Save Outputs, Write Logs, and Interpret Success Correctly

### Output artifacts
The pipeline writes:
- `pipeline.log`
- `excitation_trajectory.npz`
- `identification_results.npz`
- `results_summary.json`

These are orchestrated from [`src/pipeline.py`](../src/pipeline.py), with logging configured in [`src/pipeline_logger.py`](../src/pipeline_logger.py).

### Important semantic distinction
Pipeline completion and physical feasibility are **not** the same statement.

An unconstrained run can:
- stack the data correctly,
- identify parameters correctly in the least-squares sense,
- save all artifacts successfully,
- and still report `feasible = false` because the identified inertial parameters violate pseudo-inertia PSD.

This distinction is scientifically important. The log file tells you whether the pipeline executed; the feasibility report tells you whether the identified rigid-body parameters correspond to a physically valid mass distribution.

### Verification evidence
- Unconstrained successful run with infeasible result: `test_stage_12_pipeline_success_and_feasibility_are_distinct_for_unconstrained_run`
- Constrained LMI feasible run: `test_stage_12_constrained_lmi_returns_feasible_newton_euler_model`
- Constrained Cholesky feasible run: `test_cholesky_constrained_produces_feasible_model`

## Traceability Appendix

| Stage | Main claim | Code | Verification |
|---|---|---|---|
| 1 | Standalone URDF parsing and serial-chain extraction work | [`src/urdf_parser.py`](../src/urdf_parser.py) | `test_stage_1_and_3_parser_and_kinematics_build_standalone_model` |
| 2 | Missing limits are rejected clearly | [`src/urdf_parser.py`](../src/urdf_parser.py) | `test_stage_2_joint_limit_extraction_rejects_missing_json_overrides` |
| 3 | The inertial parameter vector matches the URDF data and parallel-axis shift | [`src/kinematics.py`](../src/kinematics.py) | `test_stage_1_and_3_parser_and_kinematics_build_standalone_model` |
| 4-5 | NE and EL produce the same torques on the same robot/state | [`src/dynamics_newton_euler.py`](../src/dynamics_newton_euler.py), [`src/dynamics_euler_lagrange.py`](../src/dynamics_euler_lagrange.py) | `test_stage_4_and_5_newton_euler_and_euler_lagrange_match_shared_torques`, `test_slow_ne_el_regressors_match_across_random_3dof_states` |
| 6 | Sine endpoint conditions hold on integer periods | [`src/trajectory.py`](../src/trajectory.py) | `test_stage_6_sine_basis_enforces_boundary_conditions_on_integer_periods` |
| 6 | Unsupported noninteger sine periods are rejected | [`src/config_loader.py`](../src/config_loader.py) | `test_stage_6_noninteger_sine_periods_are_rejected_by_config` |
| 6 | Friction augmentation matches the documented matrices | [`src/friction.py`](../src/friction.py) | `test_stage_6_friction_regressor_augmentation_matches_supported_models` |
| 6 | Excitation styles map to distinct solver paths | [`src/excitation.py`](../src/excitation.py) | `test_stage_6_excitation_styles_dispatch_to_distinct_solver_paths` |
| 8 | $W$ is stacked exactly as the theory states | [`src/observation_matrix.py`](../src/observation_matrix.py) | `test_stage_8_observation_matrix_matches_manual_stacking_equation` |
| 8 | Filtering happens before downsampling | [`src/filtering.py`](../src/filtering.py), [`src/observation_matrix.py`](../src/observation_matrix.py) | `test_stage_8_filtering_happens_before_downsampling` |
| 9 | Base reduction preserves the observation equation | [`src/base_parameters.py`](../src/base_parameters.py) | `test_stage_9_base_parameter_reduction_preserves_observation_equation_for_ne_and_el`, `test_slow_base_parameter_reduction_preserves_multiple_random_3dof_observation_matrices` |
| 10 | Parameter bounds switch the pipeline into bounded LS | [`src/pipeline.py`](../src/pipeline.py), [`src/solver.py`](../src/solver.py) | `test_stage_10_parameter_bounds_enable_bounded_ls` |
| 11 | Pseudo-inertia detects physically invalid bodies | [`src/feasibility.py`](../src/feasibility.py) | `test_stage_11_pseudo_inertia_checks_report_standard_rigid_body_failures` |
| 11 | EL constrained feasibility modes are rejected honestly | [`src/config_loader.py`](../src/config_loader.py) | `test_stage_11_euler_lagrange_rejects_constrained_feasibility_modes` |
| 12 | Successful execution is distinct from physical feasibility | [`src/pipeline.py`](../src/pipeline.py) | `test_stage_12_pipeline_success_and_feasibility_are_distinct_for_unconstrained_run` |
| 12 | Constrained NE identification can return a feasible model (LMI) | [`src/solver.py`](../src/solver.py), [`src/feasibility.py`](../src/feasibility.py) | `test_stage_12_constrained_lmi_returns_feasible_newton_euler_model` |
| 12 | Constrained NE identification can return a feasible model (Cholesky) | [`src/solver.py`](../src/solver.py), [`src/feasibility.py`](../src/feasibility.py) | `test_cholesky_constrained_produces_feasible_model` |

## References

1. Atkeson, C. G., An, C. H., & Hollerbach, J. M. (1986). *Estimation of inertial parameters of manipulator loads and links*. The International Journal of Robotics Research, 5(3), 101-119.
2. Gautier, M. (1991). *Numerical calculation of the base inertial parameters of robots*. Journal of Robotic Systems, 8(4), 485-506.
3. Gautier, M., & Khalil, W. (1992). *Exciting trajectories for the identification of base inertial parameters of robots*. The International Journal of Robotics Research, 11(4), 362-375.
4. Swevers, J., Ganseman, C., Tukel, D. B., de Schutter, J., & Van Brussel, H. (1997). *Optimal robot excitation and identification*. IEEE Transactions on Robotics and Automation, 13(5), 730-740.
5. Khalil, W., & Dombre, E. (2002). *Modeling, Identification and Control of Robots*. Hermes Penton Science.
6. Sousa, C. D., & Cortesao, R. (2014). *Physical feasibility of robot base inertial parameter identification: A linear matrix inequality approach*. The International Journal of Robotics Research, 33(6), 931-944.
7. Wensing, P. M., Kim, S., & Slotine, J.-J. E. (2018). *Linear matrix inequalities for physically consistent inertial parameter identification: A statistical perspective on the mass distribution*. IEEE Robotics and Automation Letters, 3(1), 60-67.
8. Traversaro, S., Brossette, S., Escande, A., & Nori, F. (2016). *Identification of fully physical consistent inertial parameters using optimization on manifolds*. IEEE/RSJ International Conference on Intelligent Robots and Systems.
