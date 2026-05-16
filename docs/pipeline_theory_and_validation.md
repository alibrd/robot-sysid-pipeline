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
| Literature-standard SLSQP excitation | Supported |
| Friction augmentation (`none`, `viscous`, `coulomb`, `viscous_coulomb`) | Supported |
| Zero-phase Butterworth filtering | Supported |
| Downsampling after filtering | Supported |
| QR-based base-parameter reduction | Supported |
| OLS / WLS / bounded least squares | Supported |
| Pseudo-inertia feasibility check | Supported |
| Constrained identification with pseudo-inertia PSD (LMI) | Supported |
| Cholesky-factored feasibility reparameterization | Supported |
| Torque-limited excitation (`nominal_hard`, `soft_penalty`, `robust_box`, `chance`, `actuator_envelope`, `sequential_redesign`) | Supported |
| Unified single-config runner for pipeline, validation, report, benchmark, and plot stages | Supported |
| Cartesian / workspace excitation constraints | **Not implemented** |
| Automatic differentiation from raw $q$ to $\dot q,\ddot q$ | **Not implemented**; external data must provide `dq` and `ddq` |

### Standalone evidence
- Internal URDF and xacro fixtures live in [`tests/assets`](../tests/assets), including RRBot, Drake pendulum, FingerEdu, and Elbow manipulator reference models.
- The default verification suite is in [`tests/test_pipeline_theory.py`](../tests/test_pipeline_theory.py).
- The optional deeper verification suite is in [`tests/test_pipeline_theory_slow.py`](../tests/test_pipeline_theory_slow.py) and is enabled with `--run-slow`.
- Excitation preflight regressions live in [`tests/test_excitation_x0.py`](../tests/test_excitation_x0.py).
- Torque-limited excitation regressions live in [`tests/test_torque_constraints.py`](../tests/test_torque_constraints.py) and [`tests/test_torque_constraints_slow.py`](../tests/test_torque_constraints_slow.py).
- Unified runner regressions live in [`tests/test_runner.py`](../tests/test_runner.py).

Run all theory verification tests:

```bash
# Fast suite
pytest tests/test_pipeline_theory.py -v -s

# Full suite including slow tests
pytest tests/test_pipeline_theory.py tests/test_pipeline_theory_slow.py --run-slow -v -s
```

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
- resolves `.xacro` files through the `xacro` CLI using the xacro file's parent directory as the working directory,
- reads links, joints, inertial blocks, and limits,
- extracts one serial chain by topology walk,
- accumulates fixed transforms before the first revolute joint into `Tw_0`.

### Verification evidence

**What is verified**: The URDF parser extracts the correct serial chain (number of joints, joint names) and the `RobotKinematics` constructor builds a parameter vector `PI` whose values match the URDF inertial data.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_1_and_3_parser_and_kinematics_build_standalone_model -v -s
```

**Expected output** (excerpt):
```
STAGE 1 & 3: URDF parsing and inertial-parameter vector construction
  Parsed nDoF          = 2
  Revolute joint names = ['single_rrbot_joint1', 'single_rrbot_joint2']
  PI link-1 expected   = [1.     0.     0.     0.45   1.2025 0.     0.     1.2025 0.     1.    ]
  max|PI_actual - PI_expected| = 0.00e+00
  VERIFIED: URDF parsed correctly, PI vector matches expected values (atol=1e-12)
```

**What is verified**: The Drake pendulum and FingerEdu reference fixtures parse correctly.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_1_reference_models_remain_supported -v -s
```

**Expected output** (excerpt):
```
STAGE 1: Additional URDF fixtures remain supported
  DrakePendulum_1DoF nDoF = 1
  FingerEdu_3DoF nDoF = 3
  VERIFIED: DrakePendulum_1DoF and FingerEdu_3DoF fixtures parse correctly
```

- Existing chain-order and `Tw_0` checks: [`tests/test_pipeline.py`](../tests/test_pipeline.py)
- Elbow manipulator parsing checks cover a spatial 3-DoF RRR fixture with
  expected joint names, axes, masses, identity `Tw_0`, Newton-Euler pipeline
  smoke coverage, and base-parameter count plausibility:
  [`tests/test_pipeline.py`](../tests/test_pipeline.py). A slow unified-runner
  check exercises the same fixture through pipeline and PyBullet validation:
  [`tests/test_runner.py`](../tests/test_runner.py).

## Stage 2. Resolve Joint Limits, Resolve Torque Limits When Needed, and Fail Early

### Governing rule
The excitation stage always requires bounds on $q$, $\dot q$, and $\ddot q$.
If torque-limited excitation is enabled, it also requires per-joint torque
limits $\tau_{\min}, \tau_{\max}$. Missing required limits are rejected before
optimization starts.

This is an implementation policy rather than a literature equation, but it is
the correct contract for a safe excitation optimizer.

### Code path
- Joint-limit extraction: `extract_joint_limits()` in [`src/urdf_parser.py`](../src/urdf_parser.py)
- Torque-limit extraction: `extract_torque_limits()` in [`src/urdf_parser.py`](../src/urdf_parser.py)

For position and velocity, the function applies this precedence:
1. JSON override, if supplied.
2. URDF limit, if present.
3. Hard failure with a clear error message.

Acceleration limits are always expected from JSON for the supplied standalone
fixtures because URDFs typically do not store them.

For torque limits, the current implementation applies this precedence when
torque-limited excitation is requested:
1. URDF/XACRO effort limit.
2. JSON fallback from `joint_limits.torque`.
3. Hard failure with a clear error message.

### Verification evidence

**What is verified**: The pipeline raises a `ValueError` when required joint
limits are missing from both the URDF and the JSON override.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_2_joint_limit_extraction_rejects_missing_json_overrides -v -s
```

**Expected output** (excerpt):
```
STAGE 2: Missing joint limits must be rejected early
  ValueError raised with 'Position limits missing'
  VERIFIED: Pipeline fails early when required joint limits are absent
```

**What is verified**: Torque-limit precedence is URDF effort first, JSON
fallback second, and hard failure third when torque-constrained excitation
requires limits.

**Run**:
```bash
pytest tests/test_torque_constraints.py::test_stage_2_torque_limit_precedence_urdf_then_json_then_error -v -s
```

**Expected output** (excerpt):
```
tests/test_torque_constraints.py::test_stage_2_torque_limit_precedence_urdf_then_json_then_error PASSED
```

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

**What is verified**: The per-link parameter vector `PI` matches the URDF inertial data (including parallel-axis shift) and the gravity constant is the standard SI value.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_1_and_3_parser_and_kinematics_build_standalone_model -v -s
```

**Expected output** (excerpt):
```
STAGE 1 & 3: URDF parsing and inertial-parameter vector construction
  PI link-1 actual     = [1.     0.     0.     0.45   1.2025 0.     0.     1.2025 0.     1.    ]
  max|PI_actual - PI_expected| = 0.00e+00
  VERIFIED: URDF parsed correctly, PI vector matches expected values (atol=1e-12)
```

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

**What is verified**: The Newton-Euler and Euler-Lagrange regressors produce identical torques $\tau = Y \pi$ on the default 2-DoF RRBot for random joint states.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_4_and_5_newton_euler_and_euler_lagrange_match_shared_torques -v -s
```

**Expected output** (excerpt):
```
STAGE 4 & 5: NE and EL regressors must produce identical torques (2-DoF RRBot)
  State 1: tau_ne=[...], tau_el=[...], max|diff|=X.XXe-XX
  State 2: ...
  State 3: ...
  VERIFIED: NE and EL torques agree across 3 random states (atol=1e-10)
```

**What is verified** (slow): Same agreement across 10 random states on the default 2-DoF RRBot.

**Run**:
```bash
pytest tests/test_pipeline_theory_slow.py::test_slow_ne_el_regressors_match_across_random_default_states --run-slow -v -s
```

**Expected output** (excerpt):
```
STAGE 4 & 5 (slow): NE/EL torque agreement across 10 random states (2-DoF RRBot)
  State  1: max|tau_ne - tau_el| = X.XXe-XX
  ...
  VERIFIED: NE and EL torques agree across 10 random states (atol=1e-10)
```

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

**What is verified**: The Euler-Lagrange and Newton-Euler regressors produce identical torques $\tau = Y \pi$ on the default 2-DoF RRBot for random joint states, confirming the two independently-derived regressors are equivalent.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_4_and_5_newton_euler_and_euler_lagrange_match_shared_torques -v -s
```

**Expected output** (excerpt):
```
STAGE 4 & 5: NE and EL regressors must produce identical torques (2-DoF RRBot)
  State 1: tau_ne=[...], tau_el=[...], max|diff|=X.XXe-XX
  State 2: ...
  State 3: ...
  VERIFIED: NE and EL torques agree across 3 random states (atol=1e-10)
```

**What is verified** (slow): Same agreement across 10 random states on the default 2-DoF RRBot.

**Run**:
```bash
pytest tests/test_pipeline_theory_slow.py::test_slow_ne_el_regressors_match_across_random_default_states --run-slow -v -s
```

**Expected output** (excerpt):
```
STAGE 4 & 5 (slow): NE/EL torque agreement across 10 random states (2-DoF RRBot)
  State  1: max|tau_ne - tau_el| = X.XXe-XX
  ...
  VERIFIED: NE and EL torques agree across 10 random states (atol=1e-10)
```

### Current implementation note
The EL symbolic builder returns a reduced symbolic regressor after removing structurally zero columns (i.e. columns that are literal zero in their symbolic form, without invoking symbolic reasoning such as `equals` or `simplify`). Columns that are algebraically zero but not yet simplified are left in place and handled later by the numeric base-parameter reduction in Stage 9. The public `RegressorModel` wrapper reinserts the pruned columns before downstream pipeline stages, preserving the full per-link 10-parameter contract used by Stage 11 feasibility constraints.

## Stage 6. Build the Excitation Trajectory and Choose an Optimization Style

### Governing equations: harmonic parameterization
The code supports three basis families. For harmonic frequencies
$f_j = j f_0$, define

$$
\omega_j = 2 \pi f_j.
$$

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

**Both basis functions, no phase optimization**

$$
q_i(t) =
\sum_{j=1}^{m} a_{ij}\cos(\omega_j t)
+ \sum_{j=1}^{m} b_{ij}\sin(\omega_j t)
+ \lambda_{1,i} t + \lambda_{0,i} + q_{0,i},
$$

with

$$
\lambda_{1,i} = -\sum_{j=1}^{m} b_{ij}\omega_j,
\qquad
\lambda_{0,i} = -\sum_{j=1}^{m} a_{ij}.
$$

The corresponding derivatives are

$$
\dot q_i(t) =
\sum_{j=1}^{m} -a_{ij}\omega_j \sin(\omega_j t)
+ \sum_{j=1}^{m} b_{ij}\omega_j \cos(\omega_j t)
+ \lambda_{1,i},
$$

$$
\ddot q_i(t) =
\sum_{j=1}^{m} -a_{ij}\omega_j^2 \cos(\omega_j t)
- \sum_{j=1}^{m} b_{ij}\omega_j^2 \sin(\omega_j t).
$$

**Both basis functions with phase optimization**

$$
q_i(t) =
\sum_{j=1}^{m} a_{ij}\cos(\omega_j t + \phi_{ij})
+ \lambda_{1,i} t + \lambda_{0,i} + q_{0,i},
$$

with

$$
\lambda_{1,i} = \sum_{j=1}^{m} a_{ij}\omega_j \sin(\phi_{ij}),
\qquad
\lambda_{0,i} = -\sum_{j=1}^{m} a_{ij}\cos(\phi_{ij}).
$$

This gives

$$
\dot q_i(t) =
\sum_{j=1}^{m} -a_{ij}\omega_j \sin(\omega_j t + \phi_{ij}) + \lambda_{1,i},
$$

$$
\ddot q_i(t) =
\sum_{j=1}^{m} -a_{ij}\omega_j^2 \cos(\omega_j t + \phi_{ij}).
$$

Harmonic excitation design is standard in robot identification; see Gautier and Khalil (1992) and Swevers et al. (1997).

### Governing equations: tightened search bounds and long-horizon sine feasibility
The current implementation does not bound each harmonic amplitude only by the
position range. For harmonic $j$ on joint $i$, the bound used by
`param_bounds()` is

$$
\bar a_{ij} =
\min \left(
\frac{q_{i,\max} - q_{i,\min}}{2},
\frac{\dot q_{i,\max}^{\mathrm{sym}}}{\omega_j},
\frac{\ddot q_{i,\max}^{\mathrm{sym}}}{\omega_j^2}
\right),
$$

where $\dot q_{i,\max}^{\mathrm{sym}}$ and
$\ddot q_{i,\max}^{\mathrm{sym}}$ are the smaller-magnitude sides of the
implemented symmetric velocity and acceleration limits. The same per-harmonic
cap is used for cosine amplitudes and phase-optimized amplitudes.

For `basis="both"` with `optimize_phase=false`, the sine amplitudes use a
tighter bound that also accounts for the secular drift term
$\lambda_{1,i} t = -2\pi \sum_j b_{ij} f_j \, t$. Because the position
contribution of sine harmonic $j$ is $b_{ij}[\sin(\omega_j t) - \omega_j t]$,
the non-periodic drift grows linearly with time.  At $t = t_f$ the drift
magnitude is $\omega_j t_f |b_{ij}|$, which must not exceed
$(q_{i,\max} - q_{i,\min})/2$ for the position constraint to remain feasible.
The per-harmonic sine amplitude bound is therefore

$$
\bar b_{ij} =
\min\!\left(
\bar a_{ij},\;
\frac{q_{i,\max} - q_{i,\min}}{2\,\omega_j\,t_f}
\right),
$$

where $\bar a_{ij}$ is the standard velocity/acceleration-tightened bound
defined above. This is a conservative per-harmonic search-box restriction: the
actual drift is coupled through $\lambda_1 = -\sum_j \omega_j b_j$, so multiple
coefficients can combine into aggregate drift that exceeds any single
per-coefficient bound. The linear trajectory constraints (which check the
total position at every grid point) remain the authoritative feasibility
guarantee; the tightened bounds serve as a preconditioning step that shrinks the
search box toward the feasible region, improving SLSQP convergence. For short
trajectories the drift term is non-binding, but for long horizons (large $t_f$)
it can dominate and shrink the feasible sine amplitudes significantly.

For `basis="sine"`, the drift correction term $\lambda_{1,i}$ also creates a
duration-dependent amplitude cap. The code enforces

$$
|\lambda_{1,i}| \le
\min \left(
\dot q_{i,\max}^{\mathrm{sym}},
\frac{q_{i,\max} - q_{i,\min}}{2 t_f}
\right),
$$

with

$$
\lambda_{1,i} = -\sum_{j=1}^{m} b_{ij}\omega_j.
$$

As $t_f$ grows, the admissible drift shrinks like
$O((q_{i,\max} - q_{i,\min}) / t_f)$, so the feasible sine amplitudes collapse
toward near-zero motion. The preflight logic computes a usable-fraction metric

$$
\eta_i =
\frac{\lambda_{1,i}^{\max}}
{\sum_{j=1}^{m} \omega_j \bar b_{ij}},
\qquad
\eta = \min_i \eta_i,
$$

and rejects sine-only trajectories when the feasible amplitude fraction falls
below the implemented threshold of $1\%$. For long trajectories, the supported
alternative is `basis_functions="both"` with `optimize_phase=false`.

### Governing equations: excitation objectives
The literature-standard conditioning objective is based on the observation matrix. Classical criteria include minimizing

$$
\kappa_2(W^\top W)
$$

or the equivalent singular-value condition metric on $W$ itself. Because

$$
\kappa_2(W^\top W) = \kappa_2(W)^2,
$$

minimizing $\kappa_2(W)$ is monotonic-equivalent to minimizing $\kappa_2(W^\top W)$ whenever $W$ has nonzero singular values. This is the basis of the code's `_condition_cost_base()` implementation. See Gautier and Khalil (1992) and Swevers et al. (1997).

The optimizer minimizes $\log_{10}(\kappa_2)$ rather than the raw condition
number. This is a monotone transformation that preserves the minimizer for the
pure conditioning objective but compresses the gradient scale. The $\log_{10}$ transformation brings the
cost and constraint gradients into comparable magnitudes, which is critical for
reliable SLSQP convergence.

When `torque_constraint_method="soft_penalty"` is active, the total objective
becomes $\log_{10}(\kappa_2) + w \cdot P_{\text{soft}}$, a scalarized
multi-objective that trades conditioning against torque-limit compliance. The
$\log_{10}$ transformation changes the relative weighting between the
conditioning and penalty terms compared to what a raw-$\kappa_2$ formulation
would produce. Users who need precise control over the conditioning-vs-penalty
tradeoff should adjust `soft_penalty_weight` accordingly.

### Governing equations: initial guess construction

The literature-standard initial guess for `basis="both"` with
`optimize_phase=false` sets each amplitude to a fraction of its upper bound.
The cosine amplitudes alternate in sign across harmonics:

$$
a_{ij}^{(0)} = (-1)^{j}\, s \, \bar a_{ij} \, (1 + \epsilon_{ij}),
$$

where $s = 0.5/m$ is the scale factor, $\bar a_{ij}$ is the per-harmonic upper
bound, and $\epsilon_{ij} \sim \mathcal{N}(0, 0.05^2)$ is a small random
perturbation to break symmetry. The same alternating-sign pattern and scale
factor are used for the sine amplitudes $b_{ij}^{(0)}$.

The sign alternation is a heuristic that reduces $\lambda_0$ bias. The cosine
basis includes the correction $\lambda_{0,i} = -\sum_j a_{ij}$. If all cosine
amplitudes share the same sign, $\lambda_0$ shifts the trajectory baseline far
from $q_0$, producing a one-sided initial guess that wastes half the available
joint range. Alternating signs make the $\lambda_0$ contributions partially
cancel, reducing but not eliminating the offset (since the first harmonic still
dominates in amplitude). This gives the optimizer a better starting point but
does not guarantee a symmetric trajectory — the final symmetry depends on the
optimized amplitudes, not the initial guess.

When condition-number optimization is disabled, the current code uses the fallback amplitude objective

$$
J_{\text{amp}} = -\left(\|\dot q\|_1 + \|\ddot q\|_1\right),
$$

which is an engineering heuristic for large excitation subject to limits, not a direct optimal-design criterion from the identification literature.

### Excitation optimization currently implemented

- Uses condition number on the **base-parameter** observation matrix when enabled.
- Uses SLSQP.
- Uses sampled hard inequality constraints on $q$, $\dot q$, $\ddot q$.
- Uses the base-parameter observation matrix, which is aligned with the standard identification workflow in Gautier (1991) and Swevers et al. (1997).

### Torque-limited excitation extensions
The torque-limited modes are implemented on top of the literature-standard SLSQP formulation. Let

$$
\tau_k^{\mathrm{nom}} = Y_k^{\mathrm{aug}} \pi^{\mathrm{nom}},
$$

where $Y_k^{\mathrm{aug}}$ is the rigid-body regressor optionally augmented with
friction columns and $\pi^{\mathrm{nom}}$ is the nominal parameter vector used
for trajectory design.

**`nominal_hard`**

The design-time torque is the nominal torque itself:

$$
\tau_{k,\mathrm{design}}^{\mathrm{lo}} =
\tau_{k,\mathrm{design}}^{\mathrm{hi}} =
\tau_k^{\mathrm{nom}}.
$$

SLSQP then enforces hard inequalities against the design limits.

**`soft_penalty`**

This mode does not add hard torque constraints. Instead, it adds a smooth
violation penalty to the objective:

$$
J = J_{\mathrm{base}} +
w \sum_k \left(
\operatorname{softplus}\left(
\frac{\tau_k^{\mathrm{nom}} - \tau_{k,\max}}{s}
\right)^2 s^2
+
\operatorname{softplus}\left(
\frac{\tau_{k,\min} - \tau_k^{\mathrm{nom}}}{s}
\right)^2 s^2
\right),
$$

where $w$ is `soft_penalty_weight` and $s$ is `soft_penalty_smoothing`.

**`robust_box`**

This mode models bounded parameter uncertainty with radius

$$
\delta = \max \big( \rho |\pi^{\mathrm{nom}}|, \delta_{\min} \big),
$$

where $\rho$ is `relative_uncertainty` and $\delta_{\min}$ is
`absolute_uncertainty_floor`. The induced worst-case torque interval is

$$
\tau_{k,\mathrm{design}}^{\mathrm{lo}} =
\tau_k^{\mathrm{nom}} - |Y_k^{\mathrm{aug}}| \delta,
\qquad
\tau_{k,\mathrm{design}}^{\mathrm{hi}} =
\tau_k^{\mathrm{nom}} + |Y_k^{\mathrm{aug}}| \delta.
$$

**`chance`**

This mode models Gaussian parameter uncertainty with standard deviation

$$
\sigma = \max \big( \rho_\sigma |\pi^{\mathrm{nom}}|, \sigma_{\min} \big),
$$

and confidence quantile

$$
z_\alpha = \Phi^{-1}(\alpha),
$$

where $\alpha$ is `chance_confidence`. In the current implementation,
`chance_confidence` is the one-sided Gaussian quantile level used to build the
symmetric interval $\tau_k^{\mathrm{nom}} \pm z_\alpha \sigma_\tau$; it is not
interpreted as a direct two-sided central-coverage percentage. The design
interval becomes

$$
\tau_{k,\mathrm{design}}^{\mathrm{lo}} =
\tau_k^{\mathrm{nom}} -
z_\alpha \sqrt{ (Y_k^{\mathrm{aug}})^2 \sigma^2 },
$$

$$
\tau_{k,\mathrm{design}}^{\mathrm{hi}} =
\tau_k^{\mathrm{nom}} +
z_\alpha \sqrt{ (Y_k^{\mathrm{aug}})^2 \sigma^2 }.
$$

**`actuator_envelope`**

This mode keeps the nominal design torque but makes the admissible torque limits
state-dependent. For `envelope_type="constant"`, the admissible limits remain
equal to the fixed actuator limits. For the implemented
`envelope_type="speed_linear"` envelope,

$$
\gamma_i(\dot q_i) =
\operatorname{clip}
\left(
1 - s_i \frac{|\dot q_i|}{v_i^{\mathrm{ref}}},
\gamma_i^{\min},
\gamma_i^{\max}
\right),
$$

and the effective lower/upper torque limits are scaled by $\gamma_i(\dot q_i)$.
The optional RMS constraint is

$$
\sqrt{\frac{1}{N} \sum_{k=1}^{N} \tau_{i,k}^2}
\le
r_i \max \left( |\tau_{i,\min}|, |\tau_{i,\max}| \right),
$$

where $r_i$ is `rms_limit_ratio`.

**`sequential_redesign`**

This mode is an outer-loop design strategy rather than a single optimizer
constraint. At iteration $\ell$, the pipeline:

1. runs trajectory optimization using `nominal_hard` with the current nominal
   parameter vector $\pi^{(\ell)}$,
2. identifies parameters from the resulting synthetic data,
3. replaces the nominal model with the corrected identified model
   $\pi^{(\ell+1)}$,
4. stops when the relative model change falls below `convergence_tol` or
   `max_iterations` is reached.

This path is intentionally restricted to `method="newton_euler"` and synthetic
data runs in the current implementation.

### Optimization and replay contract
The literature-standard optimizer path has several implementation-specific contracts
that matter for interpreting the theory:

- For `basis="sine"` and `basis="both"` with `optimize_phase=false`, the
  trajectory is linear in the optimization variables, so the pipeline builds
  explicit `LinearConstraint` objects for $q$, $\dot q$, and $\ddot q$ on an
  oversampled time grid.
- The oversampled grid uses the same dense replay spacing as the
  post-optimization torque validation:
  $$
  \Delta t_{\mathrm{dense}} =
  \frac{1}{2 f_{\max} \, \text{oversample}},
  $$
  where `oversample = torque_validation_oversample_factor`.
- For bases that are nonlinear in the optimization variables (`cosine`, or
  `both` with `optimize_phase=true`), the pipeline falls back to nonlinear
  max/min inequality constraints.
- Hard torque methods (`nominal_hard`, `robust_box`, `chance`,
  `actuator_envelope`) tighten the design-time limits by the guard-band factor
  `optimization_guard_band` (default `0.02`), so the optimizer solves against
  $\tau^{\mathrm{inner}} = (1-g)\tau^{\mathrm{limit}}$ before the dense replay
  checks the unshrunk limits.
- After optimization, the code replays the trajectory on the dense grid. If a
  hard torque method still violates its dense replay limits, the optimizer raises
  a `ValueError` when `strict_validation=true` and otherwise logs a warning.
- `soft_penalty` and `sequential_redesign` also emit dense replay summaries and
  torque-validation artifacts, but they do not add their own distinct hard
  torque inequalities to the optimizer. Their replay reporting is computed
  against the nominal torque limits.
- The pipeline also writes a dense replay artifact and summary for the nominal,
  identified, and corrected models so that design-time and post-identification
  torque compliance can be compared directly.

### Code path
- Harmonic trajectory generation: [`src/trajectory.py`](../src/trajectory.py)
- Excitation optimization: [`src/excitation.py`](../src/excitation.py)
- Torque design and replay helpers: [`src/torque_constraints.py`](../src/torque_constraints.py)

### Verification evidence

**What is verified**: The sine basis satisfies $q(0) = q_0$, $\dot q(0) = 0$, $\dot q(T) = 0$, and $\ddot q(T) = 0$ when the trajectory duration is an integer number of base periods.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_6_sine_basis_enforces_boundary_conditions_on_integer_periods -v -s
```

**Expected output** (excerpt):
```
STAGE 6: Sine basis boundary conditions on integer periods
  q(0)    = [ 0.1 -0.2],   expected q0 = [ 0.1 -0.2]
  dq(T)   = [...], expected = 0
  |dq(T)|    = X.XXe-XX
  |ddq(T)|   = X.XXe-XX
  VERIFIED: q(0)=q0, dq(0)=0, dq(T)=0, ddq(T)=0 on integer periods
```

**What is verified**: Config validation rejects `sine` basis with non-integer `trajectory_duration_periods`.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_6_noninteger_sine_periods_are_rejected_by_config -v -s
```

**Expected output** (excerpt):
```
STAGE 6: Noninteger sine periods must be rejected by config validation
  ValueError raised matching 'integer'
  VERIFIED: Config rejects sine basis with trajectory_duration_periods=1.5
```

**What is verified**: Long-horizon `sine` excitation is rejected during pipeline
preflight before the optimizer starts.

**Run**:
```bash
pytest tests/test_excitation_x0.py -v -k preflight
```

**Expected output** (excerpt):
```
 tests/test_excitation_x0.py::test_preflight_rejects_sine_with_optimize_phase_on_long_horizon PASSED
```

**What is verified**: The literature-standard initial guess remains safely
inside the high-harmonic acceleration limits for the current 20-harmonic
supported setup.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_6_initial_guess_keeps_high_harmonic_ddq_margin -v -s
```

**Expected output** (excerpt):
```
tests/test_pipeline_theory.py::test_stage_6_initial_guess_keeps_high_harmonic_ddq_margin PASSED
```

**What is verified**: The literature-standard initialization stays inside the intended high-harmonic acceleration limits for the supported 20-harmonic setup.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_6_excitation_uses_single_slsqp_path -v -s
```

**Expected output** (excerpt):
```
STAGE 6: Excitation uses the single literature-standard SLSQP path
  scipy.minimize calls = 1 (expected 1)
  VERIFIED: Excitation dispatches to the single SLSQP solver path
```

**What is verified** (slow): The `_condition_cost_base()` function agrees with a manual SVD-based condition-number calculation on the base-parameter observation matrix.

**Run**:
```bash
pytest tests/test_pipeline_theory_slow.py::test_slow_condition_cost_matches_manual_base_matrix --run-slow -v -s
```

**Expected output** (excerpt):
```
STAGE 6 (slow): _condition_cost_base() must agree with manual SVD condition number
  _condition_cost_base() = ...
  manual SVD ratio       = ...
  |difference|           = X.XXe-XX
  VERIFIED: Condition-cost function matches manual SVD computation (rtol=1e-10)
```

**What is verified**: Torque-limited excitation accepts the documented config
fields, preserves torque-limit precedence, and the per-method design formulas
match their analytical definitions.

**Run**:
```bash
pytest tests/test_torque_constraints.py -v
```

**Expected output** (excerpt):
```
tests/test_torque_constraints.py::test_stage_0_config_accepts_torque_fields_and_rejects_invalid_combinations PASSED
tests/test_torque_constraints.py::test_stage_2_torque_limit_precedence_urdf_then_json_then_error PASSED
tests/test_torque_constraints.py::test_stage_6_torque_design_dispatch_produces_expected_method_specific_fields PASSED
tests/test_torque_constraints.py::test_torque_pipeline_end_to_end_nominal_hard PASSED
...
```

**What is verified** (slow): The chance-constraint margin matches Monte Carlo
behavior, the six torque modes emit comparable summary metrics, and oversampled
dense replay catches violations hidden between sparse samples.

**Run**:
```bash
pytest tests/test_torque_constraints_slow.py --run-slow -v -s
```

**Expected output** (excerpt):
```
tests/test_torque_constraints_slow.py::test_slow_chance_monte_carlo_empirically_respects_reported_confidence PASSED
tests/test_torque_constraints_slow.py::test_slow_all_six_methods_emit_comparable_summary_metrics PASSED
tests/test_torque_constraints_slow.py::test_slow_oversampled_replay_detects_hidden_between_sample_violations PASSED
```

### Current implementation note
- The literature-standard SLSQP excitation formulation supports torque-limited excitation.
- `sequential_redesign` is an outer-loop redesign policy, not a single convex or smooth constrained problem.
- Cartesian/workspace excitation constraints are still not implemented.
- The code exposes `optimize_phase` only for `basis_functions="both"`.
- `config/rrbot_single_2min_20harm_pipeline.json` is the maintained 2-minute
  `basis_functions="both"` / `optimize_phase=false` reference setup used by the
  excitation preflight regression tests.

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

### Verification evidence

**What is verified**: The friction regressor blocks for all four models (`none`, `viscous`, `coulomb`, `viscous_coulomb`) match their analytical definitions.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_6_friction_regressor_augmentation_matches_supported_models -v -s
```

**Expected output** (excerpt):
```
STAGE 7 (friction): Friction regressor blocks match analytical definitions
  'none'    : shape=(3, 0)
  'viscous' : shape=(3, 3), max|err|=0.00e+00
  'coulomb' : shape=(3, 6), max|err|=0.00e+00
  'viscous_coulomb': shape=(3, 9), max|err|=0.00e+00
  VERIFIED: All friction models match analytical definitions (atol=1e-12)
```

**What is verified**: The pipeline's synthetic-data path is exercised end-to-end: the test runs the full pipeline, loads the saved excitation trajectory, independently reconstructs $q, \dot q, \ddot q$ and $\tau = Y \pi$, and verifies (1) $\tau_{\text{vec}} = W \pi$ on the reconstructed data and (2) the pipeline's identified base parameters match the true projection $P \pi$.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_7_synthetic_tau_equals_regressor_times_pi -v -s
```

**Expected output** (excerpt):
```
STAGE 7: Synthetic data must satisfy tau_k = Y_k @ pi
  nDoF = 2, n_samples = ..., PI length = 20
  max|tau_vec - W @ pi|           = 0.00e+00
  max|pi_base_pipeline - pi_true| = X.XXe-XX
  VERIFIED: Pipeline synthetic-data path produces tau = Y @ pi and recovers true base params
```

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

**What is verified**: The stacked observation matrix $W$ built by `build_observation_matrix()` equals the manual `np.vstack` of per-sample regressors, and $\tau_{\text{vec}}$ matches the flattened input.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_8_observation_matrix_matches_manual_stacking_equation -v -s
```

**Expected output** (excerpt):
```
STAGE 8: Observation matrix W must equal manual row-by-row stacking
  W shape       = (24, 20)
  max|W - W_manual|       = 0.00e+00
  max|tau_vec - tau_flat|  = 0.00e+00
  VERIFIED: W matches manual stacking and tau_vec matches flat tau (atol=1e-12)
```

**What is verified**: Filtering is applied before downsampling (not the reverse), so that high-frequency content is removed before decimation.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_8_filtering_happens_before_downsampling -v -s
```

**Expected output** (excerpt):
```
STAGE 8: Filtering must happen before downsampling
  max|W - filtered_then_ds| = 0.00e+00 (should be ~0)
  max|W - raw_ds|           = X.XXe-01 (should be >> 0)
  VERIFIED: Pipeline filters before downsampling (not the reverse)
```

- Existing sample-sufficiency rejection: [`tests/test_pipeline.py`](../tests/test_pipeline.py)

### Current implementation note
An optional observation-matrix cache can store the expensive Stage 8/9 products: `W`, `W_base`, `P`, rank, kept columns, filtered/downsampled samples, and compatibility metadata. The cache intentionally excludes identified parameters. Strict loading validates method, friction model, URDF fingerprint, nominal-parameter fingerprint, filter/downsampling settings, sample fingerprints, and matrix shapes; forced loading records mismatches in the run summary.

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

**What is verified**: The base-parameter reduction preserves the observation equation $W \pi = W_b \pi_b$ for both Newton-Euler and Euler-Lagrange observation matrices on the default 2-DoF RRBot.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_9_base_parameter_reduction_preserves_observation_equation_for_ne_and_el -v -s
```

**Expected output** (excerpt):
```
STAGE 9: Base-parameter reduction preserves observation equation (newton_euler)
  Full W shape  = (16, 20) (20 params)
  Base W shape  = (16, 6) (rank=6)
  max|W*pi - W_b*pi_b| = X.XXe-XX
  VERIFIED: W*pi == W_b*pi_b for newton_euler (atol=1e-10, rank=6)
```

**What is verified** (slow): Same preservation across 3 random 60-sample observation matrices on the default 2-DoF RRBot.

**Run**:
```bash
pytest tests/test_pipeline_theory_slow.py::test_slow_base_parameter_reduction_preserves_multiple_random_default_observation_matrices --run-slow -v -s
```

**Expected output** (excerpt):
```
STAGE 9 (slow): Base-parameter reduction across 3 random observation matrices (2-DoF RRBot)
  Trial 1: W (120, 20) -> W_b (120, 6), rank=6, max|W*pi - W_b*pi_b|=X.XXe-XX
  Trial 2: ...
  Trial 3: ...
  VERIFIED: Observation equation preserved across 3 random matrices (atol=1e-10)
```

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

**What is verified**: When `parameter_bounds=true`, the pipeline automatically switches from OLS to `bounded_ls`.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_10_parameter_bounds_enable_bounded_ls -v -s
```

**Expected output** (excerpt):
```
STAGE 10: parameter_bounds=true must switch solver to bounded_ls
  Solver in results_summary.json = 'bounded_ls'
  VERIFIED: Pipeline auto-switched from OLS to bounded_ls when parameter_bounds=true
```

**What is verified**: OLS recovers exact base parameters from noiseless synthetic data when $W$ has full column rank.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_10_ols_recovers_exact_base_parameters_from_noiseless_data -v -s
```

**Expected output** (excerpt):
```
STAGE 10: OLS must recover exact base parameters from noiseless data
  W_base shape    = (50, 7)
  max|pi_hat - pi_true| = X.XXe-15
  VERIFIED: OLS recovers exact base parameters (atol=1e-10)
```

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

The constrained identification problem operates on the full per-link 10-parameter rigid-body blocks exposed by the pipeline regressor model and has two solver paths.

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
- Full-column regressor contract: [`src/regressor_model.py`](../src/regressor_model.py)
- Config validation: [`src/config_loader.py`](../src/config_loader.py)

The code offers:
- post-hoc feasibility diagnostics for any parameter vector with complete 10-parameter link blocks,
- constrained full-space SLSQP with pseudo-inertia eigenvalue constraints (`"lmi"`),
- Cholesky-reparameterised unconstrained L-BFGS-B optimisation (`"cholesky"`),
- PSD projection by eigenvalue clipping when post-hoc correction is requested.

### Verification evidence

**What is verified**: The post-hoc feasibility check detects all standard rigid-body failure conditions (negative mass, non-PSD inertia, triangle inequality violation, non-PSD pseudo-inertia).

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_11_pseudo_inertia_checks_report_standard_rigid_body_failures -v -s
```

**Expected output** (excerpt):
```
STAGE 11: Pseudo-inertia checks must detect physically invalid bodies
  pi_bad      = [-1.  0.  0.  0. -0.1 0.  0.  0.1 0.  0.1]
  feasible    = False
  Issues: Non-positive mass | Inertia not PSD | Triangle ineq. | Pseudo-inertia NOT PSD
  VERIFIED: All standard rigid-body failure conditions detected
```

**What is verified**: The Euler-Lagrange backend is wrapped back to the full 10-column-per-link public parameter contract, so constrained feasibility modes are accepted at config load.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_11_euler_lagrange_accepts_constrained_feasibility_modes -v -s
```

**Expected output** (excerpt):
```
STAGE 11: EL method keeps the full 10-column contract
  VERIFIED: euler_lagrange + feasibility_method='lmi' is accepted at config load
```

**What is verified**: The pseudo-inertia roundtrip $\pi \to J \to \pi$ recovers the original parameter vector.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_11_pseudo_inertia_roundtrip -v -s
```

**Expected output** (excerpt):
```
STAGE 11: Pseudo-inertia roundtrip pi -> J -> pi
  max|pi_back - pi| = X.XXe-17
  VERIFIED: pi -> J -> pi roundtrip recovers original parameters (atol=1e-14)
```

**What is verified**: The Cholesky solver path produces parameters with $J \succeq 0$ by construction.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_11_cholesky_solver_guarantees_psd -v -s
```

**Expected output** (excerpt):
```
STAGE 11: Cholesky solver must produce pseudo-inertia PSD output
  Pseudo-inertia eigenvalues = [...]
  is_pseudo_inertia_psd      = True
  VERIFIED: Cholesky solver guarantees J >= 0 by construction
```

**What is verified**: LMI-constrained NE identification returns a feasible pseudo-inertia model.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_12_constrained_lmi_returns_feasible_newton_euler_model -v -s
```

**Expected output** (excerpt):
```
STAGE 12: LMI-constrained NE identification must return a feasible model
  feasible            = True
  solved_in_full_space= True
  Link 1 pseudo-inertia eigenvalues = [...]
  Link 2 pseudo-inertia eigenvalues = [...]
  VERIFIED: LMI-constrained model is feasible with J_i >= 0 for all links
```

### Current implementation note
- `feasibility_method="lmi"` uses SLSQP with explicit eigenvalue inequality constraints on $J_i$.
- `feasibility_method="cholesky"` reparameterises $J_i = L_i L_i^\top$ and uses unconstrained L-BFGS-B with analytical gradients, following Traversaro et al. (2016). This guarantees $J_i \succeq 0$ by construction throughout optimisation.
- The EL symbolic builder still prunes structural zero columns internally for efficiency, but the public `RegressorModel` wrapper reinserts those columns before the matrix reaches pipeline stages.
- All pipeline backends now expose full per-link 10-parameter blocks to `check_feasibility()`.

## Stage 12. Save Outputs, Write Logs, Optionally Export an Adapted URDF, and Interpret Success Correctly

### Output artifacts
The pipeline writes:
- `pipeline.log`
- `regressor_model.json`
- `regressor_model.urdf`
- `regressor_function.py`
- `observation_matrix_cache.npz` when observation-matrix cache saving is enabled
- `excitation_trajectory.npz`
- `torque_limit_validation.npz` when torque-constrained replay is produced
- `identification_results.npz`
- `results_summary.json`
- `<export.urdf_filename>` (default `adapted_robot.urdf`) when `export.enabled=true`
- `<export.friction_sidecar_filename>` (default `adapted_friction.json`) when `export.enabled=true`, `friction.model != "none"`, and `export.friction_sidecar=true`

These are orchestrated from [`src/pipeline.py`](../src/pipeline.py), with logging configured in [`src/pipeline_logger.py`](../src/pipeline_logger.py).

The unified entry point [`sysid.py`](../sysid.py) uses
[`src/runner.py`](../src/runner.py) to translate one JSON config into the
pipeline, PyBullet validation, report, benchmark, and plot stages. This runner
sets the artifact layout; it does not change the inverse-dynamics model,
excitation objective, observation matrix, solver equations, or feasibility
criteria. For a unified `output_dir`, pipeline artifacts live under
`<output_dir>/pipeline`, PyBullet validation runs live under
`<output_dir>/validation`, benchmark artifacts live under
`<output_dir>/validation`, and excitation plots live under
`<output_dir>/plots`.

The excitation artifact stores the Fourier replay contract (`params`, `freqs`,
`q0`, `basis`, `optimize_phase`, `cost`) plus sampled time-series arrays
(`t`, `q`, `dq`, `ddq`) and joint-limit arrays (`q_lim`, `dq_lim`, `ddq_lim`).
The sampled joint arrays are stored with joints as rows, matching the plotting
and inspection utilities.

The torque-validation artifact stores the dense replay grid, actual torque
limits, replayed nominal/identified/corrected torques, normalized torque ratios,
torque-limit provenance, and sequential redesign history when that mode is used.
For hard torque methods it also stores design margins, and for the chance method
it stores the applied Gaussian quantile.

The human-readable summary reports the torque method, torque-limit
source, nominal/identified/corrected pass flags, worst normalized torque ratio,
worst joint, worst replay time, and any sequential redesign history. The binary
identification artifact stores the same method metadata alongside the identified
parameter arrays.

The standalone PyBullet validation path in
[`src/pybullet_validation.py`](../src/pybullet_validation.py) accepts URDF and
xacro robot paths. Xacro inputs are resolved through the `xacro` CLI into a
temporary loadable URDF, revolute and continuous joints receive missing limit
attributes before loading, and known vendored FingerEdu
`package://robot_properties_fingers/meshes` URIs are rewritten to local mesh
asset paths when the assets are present.

### Adapted-URDF export (opt-in Stage 12 sub-step)

When `export.enabled=true`, Stage 12 additionally writes a simulation-ready
URDF whose dynamic parameters reproduce the identified vector. The adaptation
is a deterministic re-emission of the input URDF:

- **Topology, visuals, collisions, and `<mesh>` / `package://` references
  are preserved verbatim.** Only `<inertial>` blocks of revolute child links
  and `<dynamics>` tags of revolute joints are rewritten. Xacro sources are
  resolved to URDF text via `urdf_parser.resolve_xacro_to_urdf_xml` before
  editing, reusing the same code path as the PyBullet validator.
- **Per-link `<inertial>` block.** The Atkeson 10-vector
  $\boldsymbol{\pi}_i = [m_i,\,m_i c_x,\,m_i c_y,\,m_i c_z,\,
  I^O_{xx}, I^O_{xy}, I^O_{xz}, I^O_{yy}, I^O_{yz}, I^O_{zz}]$ stores
  inertia at the *link-frame origin*. The URDF schema instead expects mass
  at the COM placed via `<origin xyz=COM>` plus a *COM-frame* inertia
  tensor. The exporter therefore recovers
  $\mathbf{c}_i = (m_i \mathbf{c}_i)/m_i$ and applies the parallel-axis
  inverse
  $$
  I^{COM}_i \;=\; I^O_i \;-\; m_i\big(\|\mathbf{c}_i\|^2\,\mathbf{I}_3
  \;-\;\mathbf{c}_i \mathbf{c}_i^\top\big)
  $$
  before writing the six unique components into `<inertia ixx ... izz>`.
  The construction is mathematically lossless: re-parsing the adapted URDF
  through `parse_urdf` + `RobotKinematics` recovers the original
  $\boldsymbol{\pi}_i$ exactly (cited in the verification subsection
  below).
- **Per-joint `<dynamics>` projection.** When `friction.model != "none"`
  the identified friction tail $[F_v,\,F_{cp},\,F_{cn}]$ per joint is
  projected onto the URDF schema as
  $\text{damping}=F_v$ and $\text{friction}=\tfrac{1}{2}(|F_{cp}|+|F_{cn}|)$.
  The symmetric average is necessary because URDF `<dynamics friction>`
  cannot represent direction-dependent dry friction; the average minimises
  the L¹ representation error against the identified asymmetric pair.
- **Asymmetric-friction JSON sidecar.** When the URDF projection above
  loses information ($F_{cp} \neq F_{cn}$) the exporter additionally writes
  a sidecar JSON whose contract is
  ```
  {
    "friction_model": "viscous_coulomb",
    "joints": [
      {"name": j, "Fv_viscous": ..., "Fcp_coulomb_positive": ...,
       "Fcn_coulomb_negative": ...}, ...
    ]
  }
  ```
  so simulators that *can* consume direction-dependent friction recover the
  full identified model.
- **Preconditions.** The exporter refuses to write a non-physical URDF: a
  non-positive identified mass on any link aborts Stage 12 with a clear
  error pointing at `identification.parameter_bounds=true` or a
  feasibility method (`cholesky` / `lmi`). Pairing `export.enabled=true`
  with an unconstrained `solver=ols` and `feasibility_method=none` is
  therefore safe but likely to abort late on small / underdetermined
  chains.

### Code path
- [`src/urdf_exporter.py`](../src/urdf_exporter.py) hosts
  `export_adapted_urdf`, the parallel-axis inverse, friction projection,
  and sidecar writer.
- [`src/pipeline.py`](../src/pipeline.py)
  (`SystemIdentificationPipeline._run_stages_7_to_11`) invokes the export
  as the final Stage 12 sub-step after `results_summary.json` has been
  written, then re-serialises the summary with the `export` block.
- [`src/runner.py`](../src/runner.py) threads the unified `export` config
  block through to the pipeline cfg so the same hook is reachable from
  `sysid.py`.
- [`src/export_adapted_urdf.py`](../src/export_adapted_urdf.py) is a
  standalone CLI that delegates to the same `export_adapted_urdf` for
  re-export from a saved `identification_results.npz` without re-running
  the pipeline.

### Important semantic distinction
Pipeline completion and physical feasibility are **not** the same statement.

An unconstrained run can:
- stack the data correctly,
- identify parameters correctly in the least-squares sense,
- save all artifacts successfully,
- and still report `feasible = false` because the identified inertial parameters violate pseudo-inertia PSD.

This distinction is scientifically important. The log file tells you whether the pipeline executed; the feasibility report tells you whether the identified rigid-body parameters correspond to a physically valid mass distribution.

### Verification evidence

**What is verified**: An unconstrained pipeline run can complete successfully while reporting `feasible=False`, demonstrating that pipeline success and physical feasibility are distinct concepts.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_12_pipeline_success_and_feasibility_are_distinct_for_unconstrained_run -v -s
```

**Expected output** (excerpt):
```
STAGE 12: Pipeline success and physical feasibility are distinct
  feasible flag         = False
  pipeline completed    = True
  VERIFIED: Pipeline completed successfully but feasible=False (unconstrained run)
```

**What is verified**: LMI-constrained NE identification returns a feasible model.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_12_constrained_lmi_returns_feasible_newton_euler_model -v -s
```

**Expected output** (excerpt):
```
STAGE 12: LMI-constrained NE identification must return a feasible model
  feasible            = True
  solved_in_full_space= True
  Link 1 pseudo-inertia eigenvalues = [...]
  Link 2 pseudo-inertia eigenvalues = [...]
  VERIFIED: LMI-constrained model is feasible with J_i >= 0 for all links
```

**What is verified**: Cholesky-constrained NE identification returns a feasible model. The test asserts the pipeline's `feasible` flag and validates pseudo-inertia PSD for every rigid-body link block.

**Run**:
```bash
pytest tests/test_pipeline_theory.py::test_stage_12_cholesky_constrained_produces_feasible_model -v -s
```

**Expected output** (excerpt):
```
STAGE 12: Cholesky-constrained pipeline must produce a feasible model
  residual = ...
  feasible = True
  nDoF = 2, pi_corrected length = 20
  Link 1 pseudo-inertia eigenvalues = [...], PSD = True
  Link 2 pseudo-inertia eigenvalues = [...], PSD = True
  VERIFIED: Cholesky-constrained pipeline produces feasible pseudo-inertia for all links
```

**What is verified**: Torque-constrained end-to-end runs emit dense replay data
and summary metrics that stay consistent across the supported torque methods.

**Run**:
```bash
pytest tests/test_torque_constraints.py -v -k "end_to_end or shared_torque_harness"
```

**Expected output** (excerpt):
```
tests/test_torque_constraints.py::test_torque_pipeline_end_to_end_nominal_hard PASSED
tests/test_torque_constraints.py::test_torque_pipeline_end_to_end_robust_box PASSED
tests/test_torque_constraints.py::test_torque_pipeline_end_to_end_chance PASSED
tests/test_torque_constraints.py::test_torque_pipeline_end_to_end_actuator_envelope_with_rms PASSED
tests/test_torque_constraints.py::test_stage_7_to_12_shared_torque_harness_runs_one_method_at_a_time PASSED
```

**What is verified**: The optional Stage 12 adapted-URDF export is
mathematically lossless (the parallel-axis inverse applied to the Atkeson
10-vector round-trips through the URDF parser exactly), the asymmetric
friction model is preserved by the JSON sidecar, the URDF `<dynamics>`
projection writes the symmetric-average friction, the config validator
rejects unsafe export filenames, and the unified runner threads the
`export` block end-to-end. A slow PyBullet round-trip additionally checks
that inverse-dynamics torques on the adapted URDF match the original to
floating-point precision.

**Run**:
```bash
pytest tests/test_urdf_export.py -v
pytest tests/test_urdf_export.py --run-slow -m slow -v   # PyBullet round-trip
```

**Expected output** (excerpt):
```
tests/test_urdf_export.py::TestExportRoundTrip::test_rrbot_no_friction_round_trip PASSED
tests/test_urdf_export.py::TestExportRoundTrip::test_elbow_round_trip_preserves_chain_order PASSED
tests/test_urdf_export.py::TestExportRoundTrip::test_perturbed_inertials_round_trip PASSED
tests/test_urdf_export.py::TestFrictionSidecar::test_viscous_coulomb_sidecar_contents PASSED
tests/test_urdf_export.py::TestFrictionSidecar::test_dynamics_tag_is_written_for_friction PASSED
tests/test_urdf_export.py::TestFrictionSidecar::test_no_friction_means_no_sidecar PASSED
tests/test_urdf_export.py::TestExportValidation::test_non_positive_mass_raises PASSED
tests/test_urdf_export.py::TestExportValidation::test_parameter_length_mismatch_raises PASSED
tests/test_urdf_export.py::TestExportValidation::test_ndof_mismatch_raises PASSED
tests/test_urdf_export.py::TestExportValidation::test_config_loader_rejects_absolute_export_filename PASSED
tests/test_urdf_export.py::TestExportValidation::test_config_loader_rejects_parent_traversal PASSED
tests/test_urdf_export.py::TestStage12Integration::test_export_disabled_writes_no_extra_files PASSED
tests/test_urdf_export.py::TestStage12Integration::test_export_enabled_writes_adapted_urdf PASSED
tests/test_urdf_export.py::TestRunnerThreading::test_unified_runner_writes_adapted_urdf PASSED
tests/test_urdf_export.py::TestStandaloneCLIDelegates::test_tool_exports_from_npz PASSED
tests/test_urdf_export.py::test_pybullet_round_trip_consistency PASSED   # --run-slow
```

## Traceability Appendix

| Stage | Main claim | Code | Verification |
|---|---|---|---|
| 1 | Standalone URDF parsing and serial-chain extraction work | [`src/urdf_parser.py`](../src/urdf_parser.py) | `test_stage_1_and_3_parser_and_kinematics_build_standalone_model` |
| 1 | Reference fixtures cover Drake pendulum, FingerEdu xacro, and spatial Elbow manipulator models | [`src/urdf_parser.py`](../src/urdf_parser.py), [`src/pipeline.py`](../src/pipeline.py), [`src/runner.py`](../src/runner.py) | `test_stage_1_reference_models_remain_supported`, `TestElbowManipulator3DoF`, `test_unified_end_to_end_elbow` |
| 2 | Missing limits are rejected clearly | [`src/urdf_parser.py`](../src/urdf_parser.py) | `test_stage_2_joint_limit_extraction_rejects_missing_json_overrides` |
| 2 | Torque limits use URDF effort first, JSON fallback second, and fail clearly when required | [`src/urdf_parser.py`](../src/urdf_parser.py) | `test_stage_2_torque_limit_precedence_urdf_then_json_then_error` |
| 3 | The inertial parameter vector matches the URDF data and parallel-axis shift | [`src/kinematics.py`](../src/kinematics.py) | `test_stage_1_and_3_parser_and_kinematics_build_standalone_model` |
| 4-5 | NE and EL produce the same torques on the default 2-DoF RRBot | [`src/dynamics_newton_euler.py`](../src/dynamics_newton_euler.py), [`src/dynamics_euler_lagrange.py`](../src/dynamics_euler_lagrange.py) | `test_stage_4_and_5_newton_euler_and_euler_lagrange_match_shared_torques`, `test_slow_ne_el_regressors_match_across_random_default_states` |
| 6 | Sine endpoint conditions hold on integer periods | [`src/trajectory.py`](../src/trajectory.py) | `test_stage_6_sine_basis_enforces_boundary_conditions_on_integer_periods` |
| 6 | Unsupported noninteger sine periods are rejected | [`src/config_loader.py`](../src/config_loader.py) | `test_stage_6_noninteger_sine_periods_are_rejected_by_config` |
| 6 | Drift-limited long-horizon sine excitation is rejected before optimization | [`src/excitation.py`](../src/excitation.py) | `test_preflight_rejects_sine_with_optimize_phase_on_long_horizon` |
| 6 | The literature-standard initial guess stays inside the intended high-harmonic acceleration margin | [`src/excitation.py`](../src/excitation.py) | `test_stage_6_initial_guess_keeps_high_harmonic_ddq_margin` |
| 6 | Torque-limited excitation config and design formulas match the documented methods | [`src/config_loader.py`](../src/config_loader.py), [`src/excitation.py`](../src/excitation.py), [`src/torque_constraints.py`](../src/torque_constraints.py) | `test_stage_0_config_accepts_torque_fields_and_rejects_invalid_combinations`, `test_stage_6_torque_design_dispatch_produces_expected_method_specific_fields`, `test_slow_chance_monte_carlo_empirically_respects_reported_confidence` |
| 6 | Friction augmentation matches the documented matrices | [`src/friction.py`](../src/friction.py) | `test_stage_6_friction_regressor_augmentation_matches_supported_models` |
| 6 | Excitation uses the single SLSQP solver path | [`src/excitation.py`](../src/excitation.py) | `test_stage_6_excitation_uses_single_slsqp_path` |
| 6 | Condition-cost function matches manual SVD computation | [`src/excitation.py`](../src/excitation.py) | `test_slow_condition_cost_matches_manual_base_matrix` |
| 7 | Synthetic data satisfies $\tau_k = Y_k \pi$ | [`src/pipeline.py`](../src/pipeline.py) | `test_stage_7_synthetic_tau_equals_regressor_times_pi` |
| 8 | $W$ is stacked exactly as the theory states | [`src/observation_matrix.py`](../src/observation_matrix.py) | `test_stage_8_observation_matrix_matches_manual_stacking_equation` |
| 8 | Filtering happens before downsampling | [`src/filtering.py`](../src/filtering.py), [`src/observation_matrix.py`](../src/observation_matrix.py) | `test_stage_8_filtering_happens_before_downsampling` |
| 8 | Torque-limited excitation settings do not change the observation-matrix construction | [`src/observation_matrix.py`](../src/observation_matrix.py) | `test_stage_8_observation_matrix_is_unchanged_by_torque_config` |
| 8-9 | Observation-matrix caches store reusable matrix/reduction artifacts and validate compatibility | [`src/observation_matrix_cache.py`](../src/observation_matrix_cache.py), [`src/pipeline.py`](../src/pipeline.py) | `tests/test_observation_matrix_cache.py` |
| 9 | Base reduction preserves the observation equation | [`src/base_parameters.py`](../src/base_parameters.py) | `test_stage_9_base_parameter_reduction_preserves_observation_equation_for_ne_and_el`, `test_slow_base_parameter_reduction_preserves_multiple_random_default_observation_matrices` |
| 10 | Parameter bounds switch the pipeline into bounded LS | [`src/pipeline.py`](../src/pipeline.py), [`src/solver.py`](../src/solver.py) | `test_stage_10_parameter_bounds_enable_bounded_ls` |
| 10 | OLS recovers exact base parameters from noiseless data | [`src/solver.py`](../src/solver.py) | `test_stage_10_ols_recovers_exact_base_parameters_from_noiseless_data` |
| 11 | Pseudo-inertia detects physically invalid bodies | [`src/feasibility.py`](../src/feasibility.py) | `test_stage_11_pseudo_inertia_checks_report_standard_rigid_body_failures` |
| 11 | Regressor models expose full rigid and friction-augmented parameter contracts | [`src/regressor_model.py`](../src/regressor_model.py) | `tests/test_regressor_model.py` |
| 11 | EL constrained feasibility modes are accepted through the full-column wrapper | [`src/config_loader.py`](../src/config_loader.py), [`src/regressor_model.py`](../src/regressor_model.py) | `test_stage_11_euler_lagrange_accepts_constrained_feasibility_modes` |
| 11 | Pseudo-inertia roundtrip $\pi \to J \to \pi$ | [`src/feasibility.py`](../src/feasibility.py) | `test_stage_11_pseudo_inertia_roundtrip` |
| 11 | Cholesky solver guarantees $J \succeq 0$ | [`src/solver.py`](../src/solver.py) | `test_stage_11_cholesky_solver_guarantees_psd` |
| 12 | Successful execution is distinct from physical feasibility | [`src/pipeline.py`](../src/pipeline.py) | `test_stage_12_pipeline_success_and_feasibility_are_distinct_for_unconstrained_run` |
| 12 | Torque-constrained runs emit dense replay summaries and method-specific artifacts consistently across the supported methods | [`src/pipeline.py`](../src/pipeline.py), [`src/torque_constraints.py`](../src/torque_constraints.py) | `test_torque_pipeline_end_to_end_nominal_hard`, `test_torque_pipeline_end_to_end_robust_box`, `test_torque_pipeline_end_to_end_chance`, `test_torque_pipeline_end_to_end_actuator_envelope_with_rms`, `test_torque_pipeline_end_to_end_sequential_redesign_improves_or_matches_initial_ratio`, `test_stage_7_to_12_shared_torque_harness_runs_one_method_at_a_time`, `test_slow_all_six_methods_emit_comparable_summary_metrics`, `test_slow_oversampled_replay_detects_hidden_between_sample_violations` |
| 12 | Constrained NE identification can return a feasible model (LMI) | [`src/solver.py`](../src/solver.py), [`src/feasibility.py`](../src/feasibility.py) | `test_stage_12_constrained_lmi_returns_feasible_newton_euler_model` |
| 12 | Constrained NE identification can return a feasible model (Cholesky) | [`src/solver.py`](../src/solver.py), [`src/feasibility.py`](../src/feasibility.py) | `test_stage_12_cholesky_constrained_produces_feasible_model` |
| 12 | Optional adapted-URDF export round-trips through `parse_urdf` exactly via the parallel-axis inverse on the Atkeson 10-vector | [`src/urdf_exporter.py`](../src/urdf_exporter.py), [`src/pipeline.py`](../src/pipeline.py) | `TestExportRoundTrip::test_rrbot_no_friction_round_trip`, `TestExportRoundTrip::test_elbow_round_trip_preserves_chain_order`, `TestExportRoundTrip::test_perturbed_inertials_round_trip` |
| 12 | URDF `<dynamics>` projection writes `damping=Fv` and `friction=0.5*(\|Fcp\|+\|Fcn\|)`; the JSON sidecar preserves the full asymmetric `Fv/Fcp/Fcn` triple | [`src/urdf_exporter.py`](../src/urdf_exporter.py) | `TestFrictionSidecar::test_viscous_coulomb_sidecar_contents`, `TestFrictionSidecar::test_dynamics_tag_is_written_for_friction`, `TestFrictionSidecar::test_no_friction_means_no_sidecar` |
| 12 | Export refuses non-physical parameter vectors and unsafe export filenames | [`src/urdf_exporter.py`](../src/urdf_exporter.py), [`src/config_loader.py`](../src/config_loader.py) | `TestExportValidation::test_non_positive_mass_raises`, `TestExportValidation::test_parameter_length_mismatch_raises`, `TestExportValidation::test_ndof_mismatch_raises`, `TestExportValidation::test_config_loader_rejects_absolute_export_filename`, `TestExportValidation::test_config_loader_rejects_parent_traversal` |
| 12 | Pipeline and unified runner reach Stage 12 export end-to-end with `export.enabled=true` | [`src/pipeline.py`](../src/pipeline.py), [`src/runner.py`](../src/runner.py) | `TestStage12Integration::test_export_disabled_writes_no_extra_files`, `TestStage12Integration::test_export_enabled_writes_adapted_urdf`, `TestRunnerThreading::test_unified_runner_writes_adapted_urdf` |
| 12 | Standalone CLI re-exports from a saved `identification_results.npz` via the same `export_adapted_urdf` implementation | [`src/export_adapted_urdf.py`](../src/export_adapted_urdf.py) | `TestStandaloneCLIDelegates::test_tool_exports_from_npz` |
| 12 | Adapted URDF is dynamically equivalent to the input under PyBullet inverse dynamics | [`src/urdf_exporter.py`](../src/urdf_exporter.py) | `test_pybullet_round_trip_consistency` (`--run-slow`) |

## References

1. Atkeson, C. G., An, C. H., & Hollerbach, J. M. (1986). *Estimation of inertial parameters of manipulator loads and links*. The International Journal of Robotics Research, 5(3), 101-119.
2. Gautier, M. (1991). *Numerical calculation of the base inertial parameters of robots*. Journal of Robotic Systems, 8(4), 485-506.
3. Gautier, M., & Khalil, W. (1992). *Exciting trajectories for the identification of base inertial parameters of robots*. The International Journal of Robotics Research, 11(4), 362-375.
4. Swevers, J., Ganseman, C., Tukel, D. B., de Schutter, J., & Van Brussel, H. (1997). *Optimal robot excitation and identification*. IEEE Transactions on Robotics and Automation, 13(5), 730-740.
5. Khalil, W., & Dombre, E. (2002). *Modeling, Identification and Control of Robots*. Hermes Penton Science.
6. Sousa, C. D., & Cortesao, R. (2014). *Physical feasibility of robot base inertial parameter identification: A linear matrix inequality approach*. The International Journal of Robotics Research, 33(6), 931-944.
7. Wensing, P. M., Kim, S., & Slotine, J.-J. E. (2018). *Linear matrix inequalities for physically consistent inertial parameter identification: A statistical perspective on the mass distribution*. IEEE Robotics and Automation Letters, 3(1), 60-67.
8. Traversaro, S., Brossette, S., Escande, A., & Nori, F. (2016). *Identification of fully physical consistent inertial parameters using optimization on manifolds*. IEEE/RSJ International Conference on Intelligent Robots and Systems.
