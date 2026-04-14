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

For standalone PyBullet torque validation against a saved excitation artifact:

```bash
python run_pybullet_validation.py config/my_pybullet_validation.json
```

To export a Markdown summary, CSV table, and per-joint torque/error plots from an
existing validation run:

```bash
python run_pybullet_validation_report.py tmp_output/pybullet_validation/<robot_name>
```

To aggregate multiple validation runs into one benchmark CSV/Markdown table:

```bash
python run_pybullet_validation_benchmark.py tmp_output/pybullet_validation
```

To automate pipeline execution plus optional validation/report/benchmark stages
from one orchestration config:

```bash
python run_workflow.py config/my_workflow.json
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

The standalone PyBullet validator writes outputs to
`<output_dir>/<robot_name>/`:

- `pybullet_validation.log` — validation log
- `pybullet_validation_data.npz` — replayed trajectory, reference torques, PyBullet torques, and errors
- `pybullet_validation_summary.json` — pass/fail summary and comparison metrics
- `pybullet_validation_report.md` — report-ready Markdown summary (via report exporter)
- `pybullet_validation_metrics.csv` — flat per-joint metrics table (via report exporter)
- `torque_overlay_<joint>.png` and `torque_abs_error_<joint>.png` — report plots (via report exporter)

At a root directory containing several validation run folders, the benchmark
exporter writes:

- `pybullet_validation_benchmark.csv` — one row per validation run
- `pybullet_validation_benchmark.md` — compact multi-run benchmark summary

## Dependencies

- Python ≥ 3.9
- NumPy, SciPy, SymPy
- pytest (for testing)
- `pybullet` (optional, only for standalone simulator validation)

Install the optional validation dependency with:

```bash
pip install pybullet
```

For report export plots, install `matplotlib` in the same environment that runs
`run_pybullet_validation_report.py`.

## Testing

```bash
python -m pytest tests/ -v
```

For the slower symbolic and conditioning checks:

```bash
python -m pytest tests/ --run-slow -m slow -v
```

PyBullet validation tests are optional and are skipped automatically when
`pybullet` is not installed.

## PyBullet validation config

Use `config/default_pybullet_validation_config.json` as the template for the
standalone validation workflow. The validator depends on the excitation artifact
contract written by the main pipeline, specifically these fields in
`excitation_trajectory.npz`:

- `params`
- `freqs`
- `q0`
- `basis`
- `optimize_phase`

The validator replays the excitation with the existing Fourier trajectory code,
computes reference torques with the Newton-Euler regressor, computes PyBullet
inverse-dynamics torques in `DIRECT` mode, and compares both sample-by-sample.

### What validation proves

- **Demonstrates**: regressor correctness, URDF-parsing consistency, parameter-packing consistency
- **Does not demonstrate**: identification quality, friction modeling, generalization to other trajectories

Together with the synthetic-data identification tests (which verify zero
residual when the true parameters are known), the validation provides evidence
that the full pipeline — URDF parsing, regressor construction, excitation
replay, and identification — is implemented correctly.  Neither the validation
nor the smoke tests alone are sufficient to prove identification accuracy on
real measured data.

## Workflow automation config

Use `config/default_workflow_config.json` as the template for the optional
workflow runner. This orchestration layer keeps the main pipeline independent
from PyBullet while still allowing one-command automation.

The workflow config supports:

- `run_pipeline`
- `run_validation`
- `run_report`
- `run_benchmark`
- `allow_missing_optional_dependencies`
- `output_root`
- `pipeline.config_path`
- `validation.config_path`
- `validation.auto_from_pipeline`
- `validation.use_external_artifacts`
- `report.validation_dir`
- `benchmark.validation_root`

When both `run_pipeline=true` and `run_validation=true`, the default behavior is
`validation.auto_from_pipeline=true`. In that mode, the workflow runner derives:

- `validation.urdf_path` from the pipeline config
- `validation.excitation_file` from `<pipeline_output_dir>/excitation_trajectory.npz`
- `validation.base_frequency_hz` from the pipeline excitation config
- `validation.trajectory_duration_periods` from the pipeline excitation config

This means users can keep pipeline and validation configs separate without
manually duplicating the URDF/excitation linkage in the common case.

If `output_root` is set, it overrides stage output locations as follows:

- pipeline output: `<output_root>/pipeline/<pipeline_config_stem>`
- validation output root: `<output_root>/validation`
- benchmark output root: `<output_root>/validation`

The workflow runner performs fail-fast preflight checks before execution. For
example, it rejects missing required stage inputs, mismatched pipeline/validation
URDFs, and validation runs that point to a different excitation artifact than
the just-run pipeline output unless `validation.use_external_artifacts=true`.

The test suite (`tests/test_pipeline.py`) verifies:

- **URDF parsing** — chain extraction, `Tw_0` base transform, topological ordering
- **Trajectory boundary conditions** — `q(0) = q0`, `dq(0) = 0` for all basis/phase combinations; sine integer-period endpoint guarantee; config rejection of non-integer sine periods; EL+constrained identification rejected; `cholesky` normalised to `lmi` with deprecation warning
- **NE / EL equivalence** — Newton-Euler and Euler-Lagrange regressors produce matching torques (default RRBot + SC fixtures)
- **Base parameter reduction** — observation equation preserved after QR reduction; `pinv(P)` reconstruction consistency
- **Filtering** — passthrough when disabled; lowpass attenuation of high-frequency components
- **Pseudo-inertia feasibility** — physically valid params pass; negative mass fails; large-first-moment-tiny-mass case correctly detected as infeasible via pseudo-inertia; projection produces PSD pseudo-inertia
- **Sample sufficiency** — insufficient samples raise an error
- **End-to-end pipeline** — default smoke tests use RRBot (`tests/assets/RRBot_single.urdf`), while SC_1DoF and SC_3DoF fixtures remain covered for compatibility/scalability checks

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
| `workflow.py` | Optional orchestration across pipeline, validation, report, benchmark |
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
