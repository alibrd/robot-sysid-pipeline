# Robot System Identification Pipeline

A standalone, modular Python pipeline for robot inverse-dynamics model
identification. It starts from a URDF file and produces an identified dynamics
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

To export a Markdown summary, CSV table, and per-joint torque/error plots from
an existing validation run:

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
| `excitation.optimize_phase` | `true` / `false` | Phase optimisation (only used with `both`) |
| `excitation.optimize_condition_number` | `true` / `false` | Minimize cond(Y^T Y) |
| `excitation.num_harmonics` | integer >= 1 | Number of Fourier harmonics (default 5) |
| `excitation.base_frequency_hz` | float > 0 | Fundamental frequency (default 0.2 Hz) |
| `excitation.torque_constraint_method` | `none` / `nominal_hard` / `soft_penalty` / `robust_box` / `chance` / `actuator_envelope` / `sequential_redesign` | Torque-limited excitation mode |
| `excitation.torque_validation_oversample_factor` | integer >= 1 | Dense replay factor for torque validation and oversampled constraint checks |
| `excitation.torque_constraint.*` | object | Method-specific torque settings such as uncertainty, envelope, penalty, guard-band, and redesign options |
| `friction.model` | `none` / `viscous` / `coulomb` / `viscous_coulomb` | Friction model |
| `identification.solver` | `ols` / `wls` / `bounded_ls` | Parameter solver |
| `identification.feasibility_method` | `none` / `lmi` / `cholesky` | Physical feasibility enforcement. `lmi` uses SLSQP with pseudo-inertia eigenvalue constraints; `cholesky` reparameterises $J = LL^\top$ guaranteeing PSD by construction (L-BFGS-B). Both require `method=newton_euler` |
| `identification.data_file` | file path or `null` | External measurement data (`.npz`) |
| `filtering.enabled` | `true` / `false` | Zero-phase Butterworth signal filtering |
| `downsampling.frequency_hz` | float | Downsample frequency (0 = disabled) |
| `joint_limits.position` | `[[lo,hi], ...]` or `null` | Joint position limits |
| `joint_limits.velocity` | `[[lo,hi], ...]` or `null` | Joint velocity limits |
| `joint_limits.acceleration` | `[[lo,hi], ...]` or `null` | Joint acceleration limits |
| `joint_limits.torque` | `[[lo,hi], ...]` or `null` | Fallback torque limits when the URDF lacks joint effort limits |

The pipeline uses the literature-standard SLSQP excitation formulation. The
method-specific `excitation.torque_constraint.*` keys are:

- `soft_penalty`: `soft_penalty_weight`, `soft_penalty_smoothing`
- `robust_box`: `relative_uncertainty`, `absolute_uncertainty_floor`
- `chance`: `relative_stddev`, `absolute_stddev_floor`, `chance_confidence`
- `actuator_envelope`: `envelope_type` (`constant` or `speed_linear`), `speed_linear_slope`,
  `velocity_reference`, `min_scale`, `max_scale`, optional `rms_limit_ratio`
- `sequential_redesign`: `max_iterations`, `convergence_tol`
- shared replay controls: `strict_validation`, `optimization_guard_band`

`chance_confidence` is used as the Gaussian quantile level in the symmetric
design interval `tau_nominal +/- z_alpha sigma`, where
`z_alpha = Phi^-1(chance_confidence)`. It is therefore a quantile parameter for
the implemented margin formula rather than a direct two-sided central-coverage
percentage.

`nominal_hard`, `robust_box`, `chance`, and `actuator_envelope` add hard
torque constraints to the SLSQP path. `soft_penalty`
adds a smooth violation penalty to the objective instead, and
`sequential_redesign` runs an outer loop that repeatedly redesigns with
`nominal_hard`.

`sequential_redesign` is further restricted to `method=newton_euler` and
synthetic-data runs (`identification.data_file=null`).

## Torque limit sourcing

When torque-limited excitation is requested, the pipeline resolves per-joint
torque limits with this precedence:

1. URDF/XACRO joint effort limit
2. JSON fallback from `joint_limits.torque`
3. Hard error if torque limits are required and neither source is available

## Pipeline stages

1. **Parse URDF** -- extract kinematic chain, inertial parameters, joint limits
   (`.xacro` supported via `xacro` CLI)
2. **Joint limits** -- merge URDF limits with JSON overrides; error if missing
3. **Kinematics** -- build symbolic+lambdified transforms, Jacobians,
   derivatives
4. **Regressor setup** -- Newton-Euler (numeric recursive) or Euler-Lagrange
   (symbolic, cached as SymPy, re-lambdified on load)
5. **Excitation design** -- Fourier trajectory optimisation with the
   literature-standard SLSQP formulation. It enforces joint-space limits and
   can additionally apply torque-limited excitation. Cartesian/workspace
   constraints are not implemented
6. **Data generation** -- synthetic from regressor or load external `.npz`
7. **Observation matrix** -- stack regressors, optional Butterworth zero-phase
   filtering and downsampling. Raises an error if the number of equations is
   fewer than the number of unknowns
8. **Base parameters** -- QR-based reduction (SciPy column-pivoted QR) to the
   identifiable parameter set
9. **Identification** -- OLS / WLS (IRLS-weighted) / bounded LS. When
   `feasibility_method` is `"lmi"`, SLSQP is used with per-link pseudo-inertia
   PSD eigenvalue constraints (Wensing et al. 2018). When `"cholesky"`, each
   link's pseudo-inertia is reparameterised as $J = LL^\top$ (lower-triangular
   `L`), guaranteeing PSD by construction, and optimised with L-BFGS-B
   (Traversaro et al. 2016). Both require `method=newton_euler`
10. **Feasibility check** -- pseudo-inertia PSD per link (the
    necessary-and-sufficient condition for physical consistency). This subsumes
    positive mass, inertia PSD, and triangle-inequality checks, which are still
    reported for diagnostics
11. **Results** -- `.npz` artifacts, JSON summary, and log file

## Output files

All outputs are written to `output_dir`:

- `pipeline.log` -- detailed log of every stage
- `excitation_trajectory.npz` -- optimised trajectory parameters
- `torque_limit_validation.npz` -- dense replay of torque limits, torque traces,
  normalized torque ratios, and method metadata when torque-limited excitation
  is enabled; hard methods also store design margins and chance quantiles when
  applicable, and `sequential_redesign` stores redesign history
- `identification_results.npz` -- identified parameters, P matrix, residuals,
  torque-method metadata, and sequential redesign history when applicable
- `results_summary.json` -- human-readable summary including torque compliance
  flags, max normalized torque ratio, worst joint/time, and redesign history
  when applicable

The standalone PyBullet validator writes outputs to
`<output_dir>/<robot_name>/`:

- `pybullet_validation.log` -- validation log
- `pybullet_validation_data.npz` -- replayed trajectory, reference torques,
  PyBullet torques, and errors
- `pybullet_validation_summary.json` -- pass/fail summary and comparison metrics
- `pybullet_validation_report.md` -- report-ready Markdown summary (via report
  exporter)
- `pybullet_validation_metrics.csv` -- flat per-joint metrics table (via report
  exporter)
- `torque_overlay_<joint>.png` and `torque_abs_error_<joint>.png` -- report
  plots (via report exporter)

At a root directory containing several validation run folders, the benchmark
exporter writes:

- `pybullet_validation_benchmark.csv` -- one row per validation run
- `pybullet_validation_benchmark.md` -- compact multi-run benchmark summary

## Dependencies

- Python >= 3.9
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

Excitation-specific coverage includes:

- `tests/test_excitation_x0.py` for long-horizon sine preflight and
  initialization regressions
- `tests/test_torque_constraints.py` for torque-limit sourcing, per-method
  torque design math, and fast end-to-end constrained runs
- `tests/test_torque_constraints_slow.py` for slower method comparisons,
  oversampled replay checks, and chance-constraint validation

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

- **Demonstrates**: regressor correctness, URDF-parsing consistency,
  parameter-packing consistency
- **Does not demonstrate**: identification quality, friction modeling,
  generalization to other trajectories

Together with the synthetic-data identification tests (which verify zero
residual when the true parameters are known), the validation provides evidence
that the full pipeline -- URDF parsing, regressor construction, excitation
replay, and identification -- is implemented correctly. Neither the validation
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
The main pipeline also performs an excitation preflight after joint-limit
resolution, including rejecting long-horizon `basis_functions="sine"` setups
whose `lambda_1` drift cap would collapse the usable amplitudes. For long
trajectories, prefer `basis_functions="both"` with `optimize_phase=false`.

The test suite (`tests/test_pipeline.py`) verifies:

- **URDF parsing** -- chain extraction, `Tw_0` base transform, topological
  ordering
- **Trajectory boundary conditions** -- `q(0) = q0`, `dq(0) = 0` for all
  basis/phase combinations; sine integer-period endpoint guarantee; config
  rejection of non-integer sine periods; EL plus constrained identification
  rejected; `cholesky` accepted as a distinct feasibility method
- **NE / EL equivalence** -- Newton-Euler and Euler-Lagrange regressors produce
  matching torques (default RRBot plus SC fixtures)
- **Base parameter reduction** -- observation equation preserved after QR
  reduction; `pinv(P)` reconstruction consistency
- **Filtering** -- passthrough when disabled; lowpass attenuation of
  high-frequency components
- **Pseudo-inertia feasibility** -- physically valid params pass; negative mass
  fails; large-first-moment-tiny-mass case is correctly detected as infeasible
  via pseudo-inertia; projection produces PSD pseudo-inertia
- **Sample sufficiency** -- insufficient samples raise an error
- **End-to-end pipeline** -- default smoke tests use RRBot
  (`tests/assets/RRBot_single.urdf`), while SC_1DoF and SC_3DoF fixtures remain
  covered for compatibility and scalability checks
- **Excitation preflight** -- long-horizon sine-only setups are rejected before
  optimisation when the `lambda_1` drift cap collapses usable amplitudes
- **Torque-limited excitation** -- torque limit sourcing, per-method design
  math, dense replay validation, and end-to-end constrained runs are covered in
  the dedicated torque suites

The documentation-linked verification layer lives in:

- `tests/test_pipeline_theory.py`: default standalone theory/evidence checks
- `tests/test_pipeline_theory_slow.py`: optional slower symbolic and
  conditioning checks
- `tests/test_excitation_x0.py`: excitation preflight and initial-guess
  regression tests
- `tests/test_torque_constraints.py`: fast torque-limited excitation and replay
  tests
- `tests/test_torque_constraints_slow.py`: slower torque-method comparison and
  oversampled replay tests

## Module overview

| Module | Purpose |
|---|---|
| `urdf_parser.py` | Parse URDF/XACRO, extract kinematic chain via topology walk |
| `kinematics.py` | Per-joint symbolic transforms, Jacobians; full-chain PI vector |
| `dynamics_newton_euler.py` | Numeric NE recursive regressor `Y(q, dq, ddq)` |
| `dynamics_euler_lagrange.py` | Symbolic EL regressor via the Lagrangian (cached to pickle) |
| `trajectory.py` | Fourier-basis trajectory generation with lambda correction terms |
| `excitation.py` | Trajectory parameter optimisation, preflight checks, and SLSQP/DE dispatch |
| `torque_constraints.py` | Torque-limit design helpers, replay validation, and torque summaries |
| `observation_matrix.py` | Stack per-sample regressors into the observation matrix |
| `base_parameters.py` | QR-based column-pivoted reduction to identifiable parameters |
| `solver.py` | OLS, WLS, bounded LS, constrained LS (SLSQP/LMI), and Cholesky-reparameterised LS (L-BFGS-B) |
| `feasibility.py` | Post-hoc physical feasibility checks, LMI projection, and Cholesky helpers |
| `filtering.py` | Zero-phase Butterworth filtering and downsampling |
| `friction.py` | Friction model parameter augmentation |
| `pipeline.py` | Main orchestrator tying all stages together |
| `config_loader.py` | JSON config loading with defaults and validation |
| `workflow.py` | Optional orchestration across pipeline, validation, report, benchmark |
| `math_utils.py` | Rotation matrices, skew-symmetric operators, constants |

## Limitations and future work

- **Excitation constraints**: joint-space limits and torque-limited excitation
  are supported. Cartesian/workspace constraints are not implemented
- **Constrained identification**: requires `method=newton_euler`. The
  Euler-Lagrange regressor drops structurally zero columns, so the reduced
  parameter vector cannot be mapped to per-link pseudo-inertia constraints. The
  config validator rejects `method=euler_lagrange` with
  `feasibility_method != "none"`
- **Sine basis endpoint guarantee**: `dq(T) = 0` and `ddq(T) = 0` for
  sine-only basis hold only when `trajectory_duration_periods` is an integer.
  The config validator enforces this
- **Long-horizon sine excitation**: the `lambda_1` drift correction can make
  sine-only trajectories fundamentally infeasible on long runs; the pipeline
  preflight rejects those cases and recommends `basis_functions="both"` with
  `optimize_phase=false`
- **Torque replay semantics**: dense replay summaries are produced for the
  implemented torque-constrained modes, but only `nominal_hard`, `robust_box`,
  `chance`, and `actuator_envelope` become hard optimizer constraints inside
  SLSQP
- **Friction models**: viscous, Coulomb, and combined friction augmentation is
  supported, but the friction coefficients are treated as free parameters in the
  regressor -- no special physical constraints are applied to them

## References

1. Atkeson, C. G., An, C. H., & Hollerbach, J. M. (1986). Estimation of inertial
   parameters of manipulator loads and links. *The International Journal of
   Robotics Research*, 5(3), 101-119.
2. Swevers, J., Ganseman, C., Tukel, D. B., De Schutter, J., & Van Brussel, H.
   (1997). Optimal robot excitation and identification. *IEEE Transactions on
   Robotics and Automation*, 13(5), 730-740.
3. Gautier, M., & Khalil, W. (1992). Exciting trajectories for the
   identification of base inertial parameters of robots. *The International
   Journal of Robotics Research*, 11(4), 362-375.
4. Khalil, W., & Dombre, E. (2002). *Modeling, identification and control of
   robots*. Butterworth-Heinemann.
5. Sousa, C. D., & Cortesao, R. (2014). Physical feasibility of robot base
   inertial parameter identification: A linear matrix inequality approach. *The
   International Journal of Robotics Research*, 33(6), 931-944.
6. Wensing, P. M., Kim, S., & Slotine, J. J. E. (2018). Linear matrix
   inequalities for physically consistent inertial parameter identification: A
   statistical perspective on the mass distribution. *IEEE Robotics and
   Automation Letters*, 3(1), 60-67.
7. Traversaro, S., Brossette, S., Escande, A., & Nori, F. (2016). Identification
   of fully physical consistent inertial parameters using optimization on
   manifolds. In *2016 IEEE/RSJ International Conference on Intelligent Robots
   and Systems (IROS)* (pp. 5446-5451). IEEE.
