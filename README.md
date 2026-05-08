# Robot System Identification Pipeline

A standalone, modular Python pipeline for robot inverse-dynamics model
identification. It starts from a URDF file and produces an identified dynamics
model in regressor form.

The full mathematical walkthrough is in
[`docs/pipeline_theory_and_validation.md`](docs/pipeline_theory_and_validation.md).
That document explains every stage of the standalone pipeline, cites the
relevant literature, and maps each equation to implementation and tests.

## Quick start

The repository exposes a single entry point that runs every stage from one JSON
configuration file:

```bash
python sysid.py config/my_robot.json
```

Use `--only`, `--skip`, `--resume`, or `--dry-run` to control execution:

```bash
# Generate and save an excitation checkpoint, but stop before identification.
python sysid.py config/my_robot.json --only excitation

# Resume identification from a previous unified output directory.
python sysid.py config/my_robot.json --resume tmp_output/elbow_3dof

# Validate an existing excitation artifact from a previous unified output.
python sysid.py config/my_robot.json --only validation_pybullet --resume tmp_output/elbow_3dof

# Print the merged config without running anything.
python sysid.py config/my_robot.json --dry-run
```

## Stages

The unified config has a `stages` block with six boolean flags:

| Flag | What it does |
|---|---|
| `excitation` | Run pipeline Stages 1-6 and save the excitation trajectory; excitation-only runs also save a checkpoint |
| `identification` | Run pipeline Stages 7-11 (observation matrix, base parameters, solver, feasibility, results) |
| `validation_pybullet` | Replay the excitation in PyBullet `DIRECT` mode and compare torques against the Newton-Euler regressor |
| `report` | Export a Markdown summary, CSV table, and per-joint plots from a validation run |
| `benchmark` | Aggregate every validation run under `<output_dir>/validation` into a benchmark CSV/Markdown table |
| `plot` | Render `excitation_trajectory.png` from `<output_dir>/pipeline/excitation_trajectory.npz` |

Common invocations map directly onto these flags:

- **Full pipeline** - `stages.excitation=true` and `stages.identification=true`. Runs Stages 1-11 and writes the pipeline artifacts under `<output_dir>/pipeline/`.
- **Generate excitation only** - `stages.excitation=true` and `stages.identification=false`. Saves `checkpoint.npz` and `checkpoint_config.json` under `<output_dir>/pipeline/`.
- **Resume identification from a saved excitation** - `stages.excitation=false`, `stages.identification=true`, and `resume.from_checkpoint` set to a previous `<output_dir>` or its `pipeline/` subdirectory. The runner loads the saved checkpoint, reconstructs Stages 1-4 from the URDF, and continues with Stages 7-11.
- **Validate an existing excitation** - `stages.validation_pybullet=true`, `stages.excitation=false`, and `resume.from_checkpoint` set to a previous `<output_dir>` or its `pipeline/` subdirectory. The runner replays the saved `excitation_trajectory.npz`.
- **Plot results** - `stages.plot=true`. Reads `<output_dir>/pipeline/excitation_trajectory.npz` and writes `<output_dir>/plots/excitation_trajectory.png`. Combine with any stage that produces or already has that pipeline artifact.
- **Compare with PyBullet after a fresh run** - `stages.validation_pybullet=true` with `stages.excitation=true`. The validation stage consumes the excitation artifact produced in the same invocation.

The unified runner translates these stage choices into the underlying pipeline's
excitation-only and checkpoint-resume controls. Identification without
excitation requires `resume.from_checkpoint`, so an identification-only stage
cannot accidentally run a fresh full pipeline.

## Output layout

For one unified `output_dir`:

```
<output_dir>/
  pipeline/                       # pipeline class output
    pipeline.log
    excitation_trajectory.npz
    identification_results.npz
    results_summary.json
    [checkpoint.npz, checkpoint_config.json]   # excitation-only runs
  validation/                     # PyBullet validation output (per-robot subdir)
    <robot_name>/
      pybullet_validation_summary.json
      pybullet_validation_data.npz
      [report artifacts when the report stage is enabled]
    pybullet_validation_benchmark.csv
    pybullet_validation_benchmark.md
  plots/                          # plot stage output
    excitation_trajectory.png
```

This layout is fixed. Point `output_dir` at a different absolute path if you want a flat directory structure.

## Unified config

Copy `config/default_config.json` and fill in the fields. The most important keys:

| Field | Values | Description |
|---|---|---|
| `urdf_path` | file path | Path to the robot URDF or xacro file |
| `output_dir` | directory | Root for all stage outputs; subdirectories `pipeline/`, `validation/`, and `plots/` are created automatically |
| `stages.*` | booleans | Stage selection (see above) |
| `resume.from_checkpoint` | path or `null` | Reuse the pipeline artifact directory from a previous unified output |
| `method` | `newton_euler` / `euler_lagrange` | Dynamics formulation |
| `excitation.basis_functions` | `cosine` / `sine` / `both` | Fourier basis |
| `excitation.optimize_phase` | `true` / `false` | Phase optimisation (only used with `both`) |
| `excitation.optimize_condition_number` | `true` / `false` | Minimize `cond(Y^T Y)` |
| `excitation.num_harmonics` | integer >= 1 | Number of Fourier harmonics |
| `excitation.base_frequency_hz` | float > 0 | Fundamental frequency |
| `excitation.torque_constraint_method` | `none` / `nominal_hard` / `soft_penalty` / `robust_box` / `chance` / `actuator_envelope` / `sequential_redesign` | Torque-limited excitation mode |
| `excitation.torque_validation_oversample_factor` | integer >= 1 | Dense replay factor for torque validation and oversampled constraint checks |
| `excitation.torque_constraint.*` | object | Method-specific torque settings (uncertainty, envelope, penalty, guard-band, redesign) |
| `friction.model` | `none` / `viscous` / `coulomb` / `viscous_coulomb` | Friction model |
| `identification.solver` | `ols` / `wls` / `bounded_ls` | Parameter solver |
| `identification.feasibility_method` | `none` / `lmi` / `cholesky` | Physical feasibility enforcement (requires `method=newton_euler`) |
| `identification.data_file` | file path or `null` | External measurement data (`.npz`) |
| `filtering.enabled` | `true` / `false` | Zero-phase Butterworth signal filtering |
| `downsampling.frequency_hz` | float | Downsample frequency (0 = disabled) |
| `joint_limits.position` | `[[lo,hi], ...]` or `null` | Joint position limits |
| `joint_limits.velocity` | `[[lo,hi], ...]` or `null` | Joint velocity limits |
| `joint_limits.acceleration` | `[[lo,hi], ...]` or `null` | Joint acceleration limits |
| `joint_limits.torque` | `[[lo,hi], ...]` or `null` | Fallback torque limits when the URDF lacks joint effort limits |
| `validation_pybullet.sample_rate_hz` | float | PyBullet replay sample rate (0 = auto) |
| `validation_pybullet.gravity` | `[gx, gy, gz]` | Gravity vector for PyBullet; must match the Newton-Euler gravity constant |
| `validation_pybullet.use_fixed_base` | `true` / `false` | Fix the base when loading the URDF in PyBullet |
| `validation_pybullet.joint_name_order` | list or `null` | Optional joint-order override |
| `validation_pybullet.comparison.tolerance_abs` | float | Absolute torque tolerance |
| `validation_pybullet.comparison.tolerance_normalized_rms` | float | Normalised RMS tolerance |
| `plot.save_only` | `true` / `false` | Save the figure without opening a window (recommended on headless machines) |
| `plot.format` | `png` / `pdf` / etc. | Output format |
| `plot.dpi` | integer | Output resolution |

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
torque constraints to the SLSQP path. `soft_penalty` adds a smooth violation
penalty to the objective instead, and `sequential_redesign` runs an outer loop
that repeatedly redesigns with `nominal_hard`. `sequential_redesign` is further
restricted to `method=newton_euler` and synthetic-data runs
(`identification.data_file=null`).

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
   literature-standard SLSQP formulation (`log10` condition number cost). It
   enforces joint-space limits with drift-aware sine amplitude bounds and
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

All pipeline outputs are written to `<output_dir>/pipeline/`:

- `pipeline.log` -- detailed log of every stage
- `excitation_trajectory.npz` -- optimised trajectory parameters (`params`,
  `freqs`, `q0`, `basis`, `optimize_phase`, `cost`) plus the dense time series
  (`t`, `q`, `dq`, `ddq`, shape `(nDoF, N)`) and joint limits (`q_lim`,
  `dq_lim`, `ddq_lim`, shape `(nDoF, 2)`). The `plot` stage renders this file
  to a PNG
- `torque_limit_validation.npz` -- dense replay of torque limits, torque traces,
  normalized torque ratios, and method metadata when torque-limited excitation
  is enabled; hard methods also store design margins and chance quantiles when
  applicable, and `sequential_redesign` stores redesign history
- `identification_results.npz` -- identified parameters, P matrix, residuals,
  torque-method metadata, and sequential redesign history when applicable
- `results_summary.json` -- human-readable summary including torque compliance
  flags, max normalized torque ratio, worst joint/time, and redesign history
  when applicable
- `checkpoint.npz` -- serialized excitation and trajectory data for pipeline
  resume (only produced by excitation-only runs)
- `checkpoint_config.json` -- snapshot of the merged pipeline config at the time
  of the excitation-only run

The plot stage writes `<output_dir>/plots/excitation_trajectory.png` (or the
configured extension from `plot.format`).

The PyBullet validator writes outputs to `<output_dir>/validation/<robot_name>/`:

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
- `pybullet` (optional, only for the validation stage)
- `matplotlib` (optional, required by the `plot` and `report` stages)

Install the optional dependencies with:

```bash
pip install pybullet matplotlib
```

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

## What validation proves

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

## Module overview

| Module | Purpose |
|---|---|
| `urdf_parser.py` | Parse URDF/XACRO, extract kinematic chain via topology walk |
| `kinematics.py` | Per-joint symbolic transforms, Jacobians; full-chain PI vector |
| `dynamics_newton_euler.py` | Numeric NE recursive regressor `Y(q, dq, ddq)` |
| `dynamics_euler_lagrange.py` | Symbolic EL regressor via the Lagrangian (cached to pickle) |
| `trajectory.py` | Fourier-basis trajectory generation with lambda correction terms and drift-aware sine bounds |
| `excitation.py` | Trajectory parameter optimisation (`log10` condition number), preflight checks, and SLSQP dispatch |
| `torque_constraints.py` | Torque-limit design helpers, replay validation, and torque summaries |
| `observation_matrix.py` | Stack per-sample regressors into the observation matrix |
| `base_parameters.py` | QR-based column-pivoted reduction to identifiable parameters |
| `solver.py` | OLS, WLS, bounded LS, constrained LS (SLSQP/LMI), and Cholesky-reparameterised LS (L-BFGS-B) |
| `feasibility.py` | Post-hoc physical feasibility checks, LMI projection, and Cholesky helpers |
| `filtering.py` | Zero-phase Butterworth filtering and downsampling |
| `friction.py` | Friction model parameter augmentation |
| `pipeline.py` | Main pipeline orchestrator tying all stages together |
| `pybullet_validation.py` | Standalone PyBullet torque comparison runner |
| `pybullet_validation_report.py` | Markdown / CSV / per-joint plot exporter for a validation run |
| `pybullet_validation_benchmark.py` | Multi-run benchmark aggregator |
| `runner.py` | Unified single-config orchestrator (`UnifiedRunner`) |
| `plot_runner.py` | Minimal matplotlib renderer for the excitation trajectory |
| `config_loader.py` | JSON config loading with defaults and validation |
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
