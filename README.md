# Robot System Identification Pipeline

A standalone, modular Python pipeline for robot inverse-dynamics model
identification. Starts from a URDF file and produces an identified dynamics
model in regressor form.

The full mathematical walkthrough is in
[`docs/pipeline_theory_and_validation.md`](docs/pipeline_theory_and_validation.md).
That document explains every stage of the standalone pipeline, cites the
relevant literature, and maps each equation to implementation and tests.

## Quick start

```bash
python run_pipeline.py config/my_robot.json
```

## JSON configuration

Copy `config/default_config.json` and fill in the fields. Key settings:

| Field | Values | Description |
|---|---|---|
| `urdf_path` | file path | Path to the robot URDF file |
| `method` | `newton_euler` / `euler_lagrange` | Dynamics formulation |
| `excitation.basis_functions` | `cosine` / `sine` / `both` | Fourier basis |
| `excitation.optimize_phase` | `true` / `false` | Phase optimisation (with `both`) |
| `excitation.optimize_condition_number` | `true` / `false` | Minimize cond(Y^T Y) |
| `excitation.constraint_style` | `legacy_excTrajGen` / `urdf_reference` / `literature_standard` | Constraint formulation |
| `excitation.num_harmonics` | integer ≥ 1 | Number of Fourier harmonics (default 5) |
| `excitation.base_frequency_hz` | float > 0 | Fundamental frequency (default 0.2 Hz) |
| `friction.model` | `none` / `viscous` / `coulomb` / `viscous_coulomb` | Friction model |
| `identification.solver` | `ols` / `wls` / `bounded_ls` | Parameter solver |
| `identification.feasibility_method` | `none` / `lmi` / `cholesky` | Physical feasibility enforcement. `cholesky` is a **deprecated** alias for `lmi` (no separate algorithm); requires `method=newton_euler` |
| `identification.data_file` | file path or `null` | External measurement data (.npz) |
| `filtering.enabled` | `true` / `false` | Zero-phase Butterworth signal filtering |
| `downsampling.frequency_hz` | float | Downsample frequency (0 = disabled) |
| `joint_limits.position` | `[[lo,hi], ...]` or `null` | Joint position limits |
| `joint_limits.velocity` | `[[lo,hi], ...]` or `null` | Joint velocity limits |
| `joint_limits.acceleration` | `[[lo,hi], ...]` or `null` | Joint acceleration limits |

## Pipeline stages

1. **Parse URDF** -- extract kinematic chain, inertial parameters, joint limits (`.xacro` supported via `xacro` CLI)
2. **Joint limits** -- merge URDF limits with JSON overrides; error if missing
3. **Kinematics** -- build symbolic+lambdified transforms, Jacobians, derivatives
4. **Regressor setup** -- Newton-Euler (numeric recursive) or Euler-Lagrange (symbolic, cached as SymPy, re-lambdified on load)
5. **Excitation design** -- Fourier trajectory optimisation (3 constraint styles; `literature_standard` uses SLSQP with base-parameter condition-number objective). Enforces joint-space limits (q, dq, ddq) only — Cartesian/workspace and torque constraints are **not implemented**
6. **Data generation** -- synthetic from regressor or load external `.npz`
7. **Observation matrix** -- stack regressors, optional Butterworth zero-phase filtering & downsampling. Raises an error if the number of equations is fewer than the number of unknowns
8. **Base parameters** -- QR-based reduction (scipy column-pivoted QR) to identifiable parameter set
9. **Identification** -- OLS / WLS (IRLS-weighted) / bounded LS. When `feasibility_method` is `"lmi"` (or `"cholesky"`, which is an alias), SLSQP is used with **pseudo-inertia PSD** constraints per link — this is the physically correct criterion (see Wensing et al. 2018). Constrained identification requires `method=newton_euler`; the Euler-Lagrange regressor drops zero columns, producing a reduced parameter vector that cannot be mapped to per-link constraints
10. **Feasibility check** -- pseudo-inertia PSD per link (the necessary-and-sufficient condition for physical consistency). Subsumes positive mass, inertia PSD, and triangle inequality checks, which are reported for diagnostics
11. **Results** -- `.npz` + JSON summary + log file

## Output files

All outputs are written to `output_dir`:

- `pipeline.log` — detailed log of every stage
- `excitation_trajectory.npz` — optimised trajectory parameters
- `identification_results.npz` — identified parameters, P matrix, residuals
- `results_summary.json` — human-readable summary

## Dependencies

- Python ≥ 3.9
- NumPy, SciPy, SymPy
- pytest (for testing)

## Testing

```bash
python -m pytest tests/ -v
```

For the slower symbolic and conditioning checks:

```bash
python -m pytest tests/ --run-slow -m slow -v
```

The test suite (`tests/test_pipeline.py`) verifies:

- **URDF parsing** — chain extraction, `Tw_0` base transform, topological ordering
- **Trajectory boundary conditions** — `q(0) = q0`, `dq(0) = 0` for all basis/phase combinations; sine integer-period endpoint guarantee; config rejection of non-integer sine periods; EL+constrained identification rejected; `cholesky` normalised to `lmi` with deprecation warning
- **NE / EL equivalence** — Newton-Euler and Euler-Lagrange regressors produce matching torques (1-DoF and 3-DoF)
- **Base parameter reduction** — observation equation preserved after QR reduction; `pinv(P)` reconstruction consistency
- **Filtering** — passthrough when disabled; lowpass attenuation of high-frequency components
- **Pseudo-inertia feasibility** — physically valid params pass; negative mass fails; large-first-moment-tiny-mass case correctly detected as infeasible via pseudo-inertia; projection produces PSD pseudo-inertia
- **Sample sufficiency** — insufficient samples raise an error
- **End-to-end pipeline** — smoke tests for NE/EL 1-DoF and 3-DoF; constrained identification verifies pseudo-inertia PSD on output

The documentation-linked verification layer lives in:

- `tests/test_pipeline_theory.py`: default standalone theory/evidence checks
- `tests/test_pipeline_theory_slow.py`: optional slower symbolic and conditioning checks

## Module overview

| Module | Purpose |
|---|---|
| `urdf_parser.py` | Parse URDF/XACRO, extract kinematic chain via topology walk |
| `kinematics.py` | Per-joint symbolic transforms, Jacobians; full-chain PI vector |
| `dynamics_newton_euler.py` | Numeric NE recursive regressor `Y(q, dq, ddq)` |
| `dynamics_euler_lagrange.py` | Symbolic EL regressor via Lagrangian (cached to pickle) |
| `trajectory.py` | Fourier-basis trajectory generation with λ-correction terms |
| `excitation.py` | Trajectory parameter optimisation (DE/SLSQP) |
| `observation_matrix.py` | Stack per-sample regressors into observation matrix |
| `base_parameters.py` | QR-based column-pivoted reduction to identifiable parameters |
| `solver.py` | OLS, WLS, bounded LS, and physically-constrained LS (SLSQP) |
| `feasibility.py` | Post-hoc physical feasibility checks and LMI projection |
| `filtering.py` | Zero-phase Butterworth filtering and downsampling |
| `friction.py` | Friction model parameter augmentation |
| `pipeline.py` | Main orchestrator tying all stages together |
| `config_loader.py` | JSON config loading with defaults |
| `math_utils.py` | Rotation matrices, skew-symmetric, constants |

## Limitations and future work

- **Excitation constraints**: only joint-space limits (position, velocity, acceleration) are enforced. Cartesian/workspace and torque constraints are not implemented
- **Constrained identification**: requires `method=newton_euler`. The Euler-Lagrange regressor drops zero columns, so the reduced parameter vector cannot be mapped to per-link pseudo-inertia constraints. The config validator rejects `method=euler_lagrange` with `feasibility_method != "none"`
- **`cholesky` vs `lmi`**: `"cholesky"` is accepted as a **deprecated** config alias for `"lmi"` and emits a `DeprecationWarning`. Both use eigenvalue-clipping projection of the pseudo-inertia matrix J via SLSQP. True Cholesky-factored reparameterisation (optimising directly over Cholesky factors of J) is not implemented
- **Sine basis endpoint guarantee**: `dq(T) = 0` and `ddq(T) = 0` for sine-only basis hold only when `trajectory_duration_periods` is an integer. The config validator enforces this
- **Friction models**: viscous, Coulomb, and combined friction augmentation is supported but the friction coefficients are treated as free parameters in the regressor — no special physical constraints are applied to them

## References

1. Atkeson, An & Hollerbach (1986) — regressor-based identification
2. Swevers et al. (1997) — optimal excitation trajectories
3. Gautier & Khalil (1992) — exciting trajectories for base parameters
4. Khalil & Dombre (2002) — robot modeling and identification
5. Sousa & Cortesão (2014) — physical feasibility via LMI
