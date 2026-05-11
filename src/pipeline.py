"""Main pipeline orchestrator: loads config, runs all stages, writes output."""
import json
import logging
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

from .base_parameters import compute_base_parameters
from .config_loader import load_config, load_config_dict
from .dynamics_euler_lagrange import euler_lagrange_regressor_builder
from .dynamics_newton_euler import newton_euler_regressor
from .excitation import optimise_excitation, preflight_excitation_config
from .feasibility import check_feasibility
from .friction import friction_param_count
from .kinematics import RobotKinematics
from .observation_matrix import build_observation_matrix
from .pipeline_logger import setup_logger
from .solver import solve_identification
from .torque_constraints import (
    build_nominal_parameter_vector,
    compute_torque_design_data,
    make_augmented_regressor,
    replay_torque_models,
    validation_time_vector,
)
from .trajectory import fourier_trajectory
from .urdf_parser import (
    extract_joint_limits,
    extract_torque_limits,
    parse_urdf,
)


def _emit_identification_warnings(cfg, log, already_emitted):
    """Log non-blocking warnings for known silent-fallback config combos."""
    if already_emitted:
        return True

    ident = cfg["identification"]
    solver = ident["solver"]
    feas = ident["feasibility_method"]
    friction_model = cfg["friction"]["model"]
    bounds_cfg = ident.get("parameter_bounds")
    bounds_set = (bounds_cfg is True) or (
        isinstance(bounds_cfg, list) and len(bounds_cfg) == 2
    )

    if feas != "none" and friction_model != "none":
        log.warning(
            "feasibility_method='%s' does NOT constrain friction parameters "
            "(Fv, Fcp, Fcn). Negative damping/Coulomb on weakly excited "
            "joints is still possible.",
            feas,
        )
    if feas != "none" and bounds_set:
        log.warning(
            "identification.parameter_bounds is set but will be ignored: "
            "feasibility_method='%s' takes the optimisation off the "
            "bounded-LS path.",
            feas,
        )
    if feas != "none" and solver == "wls":
        log.warning(
            "solver='wls' combined with feasibility_method='%s': WLS "
            "weights are not propagated into the constrained solver; "
            "the feasibility path runs unweighted.",
            feas,
        )
    if feas == "none" and solver == "ols" and friction_model != "none":
        log.warning(
            "solver='ols' with friction.model='%s' and no feasibility "
            "constraints allows negative damping/Coulomb. Consider "
            "solver='bounded_ls' with explicit lower bounds, or "
            "feasibility_method='cholesky' (note: cholesky still leaves "
            "friction unconstrained).",
            friction_model,
        )
    return True


def _clamp_negative_viscous_damping(pi_corrected, n_dof, friction_model, log):
    """Clamp negative viscous damping in the friction tail of pi_corrected."""
    if friction_model not in ("viscous", "viscous_coulomb"):
        return pi_corrected

    n_fric = friction_param_count(n_dof, friction_model)
    fric_start = len(pi_corrected) - n_fric
    fv_start = fric_start
    fv_end = fv_start + n_dof
    fv_block = pi_corrected[fv_start:fv_end]
    neg_idx = np.where(fv_block < 0.0)[0]
    if neg_idx.size == 0:
        return pi_corrected

    # check_feasibility can return the input vector by reference.
    pi_corrected = pi_corrected.copy()
    for j in neg_idx:
        log.warning(
            "Negative viscous damping on joint %d "
            "(Fv=%.6g) -- clamping to 0.0 before "
            "validation/URDF export.",
            int(j) + 1, float(fv_block[j]),
        )
    pi_corrected[fv_start:fv_end] = np.maximum(fv_block, 0.0)
    return pi_corrected


class SystemIdentificationPipeline:
    """End-to-end robot system identification pipeline."""

    def __init__(self, config_path: str | dict):
        if isinstance(config_path, dict):
            self.cfg = load_config_dict(config_path)
        else:
            self.cfg = load_config(config_path)
        self.output_dir = Path(self.cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(str(self.output_dir))
        self._id_warnings_emitted = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        t0 = time.time()
        log = self.logger
        cfg = self.cfg

        excitation_only = cfg.get("excitation_only", False)
        checkpoint_dir = cfg.get("checkpoint_dir")

        try:
            log.info("=" * 60)
            log.info("SYSTEM IDENTIFICATION PIPELINE")
            log.info("=" * 60)
            log.info("Config: method=%s, friction=%s, solver=%s",
                     cfg["method"], cfg["friction"]["model"],
                     cfg["identification"]["solver"])

            if checkpoint_dir:
                # Resume mode
                log.info("Mode: RESUME FROM CHECKPOINT (%s)", checkpoint_dir)
                cp_path = Path(checkpoint_dir)
                log.info("Loading checkpoint from %s", cp_path)
                cp_data = self._load_checkpoint(cp_path)

                ctx = self._run_stages_1_to_4()
                ctx.update(cp_data)
                log.info("Resuming from checkpoint (skipping Stages 5-6)")
                log.info("Excitation cost (from checkpoint): %.6f",
                         ctx["exc_result"]["cost"])
                self._run_stages_7_to_11(ctx)

            elif excitation_only:
                # Excitation-only mode
                log.info("Mode: EXCITATION ONLY (will stop after Stage 6)")
                ctx = self._run_stages_1_to_4()
                self._run_stage_5(ctx)
                self._run_stage_6(ctx)
                self._save_excitation_trajectory(ctx)
                self._save_checkpoint(ctx)
                log.info("Pipeline stopped after excitation (checkpoint saved).")

            else:
                # Full mode (default, backward compatible)
                ctx = self._run_stages_1_to_4()
                self._run_stage_5(ctx)
                self._run_stage_6(ctx)
                self._run_stages_7_to_11(ctx)

            elapsed = time.time() - t0
            log.info("=" * 60)
            log.info("PIPELINE COMPLETED SUCCESSFULLY in %.1f s", elapsed)
            log.info("=" * 60)

        except Exception as e:
            log.error("PIPELINE FAILED at stage above.")
            log.error("Error: %s", str(e))
            log.debug(traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    # Stage groups
    # ------------------------------------------------------------------

    def _run_stages_1_to_4(self) -> dict:
        """Initial setup stages: URDF parse, joint limits, kinematics, regressor build.

        Covers documented Stages 1-4 when ``method=newton_euler`` and Stages
        1-3 + 5 when ``method=euler_lagrange`` (the theory doc treats the two
        regressor formulations as Stages 4 and 5 respectively).

        Returns a context dict that carries all cross-stage state.
        """
        log = self.logger
        cfg = self.cfg

        log.info("-- Stage 1: Parsing URDF --")
        robot = parse_urdf(cfg["urdf_path"])
        log.info("Robot: '%s', %d DoF, revolute joints: %s",
                 robot.name, robot.nDoF, robot.revolute_joint_names)

        log.info("-- Stage 2: Extracting joint limits --")
        q_lim, dq_lim, ddq_lim = extract_joint_limits(
            robot, cfg["joint_limits"], log
        )
        torque_method = cfg["excitation"].get("torque_constraint_method", "none")
        torque_required = torque_method != "none"
        tau_lim, torque_limit_sources = extract_torque_limits(
            robot, cfg["joint_limits"], log, required=torque_required
        )
        log.info("-- Preflight: Checking excitation feasibility --")
        preflight_excitation_config(
            cfg["excitation"], q_lim, dq_lim, ddq_lim, log,
        )

        log.info("-- Stage 3: Building kinematics --")
        kin = RobotKinematics(robot, log)
        log.info("Initial parameter vector (10n=%d): %s",
                 len(kin.PI), kin.PI.flatten()[:6].tolist())

        # The theory doc treats NE and EL as Stages 4 and 5 respectively;
        # only one is built per run, so the log line picks the matching label.
        el_kept_cols = None
        if cfg["method"] == "newton_euler":
            log.info("-- Stage 4: Setting up Newton-Euler regressor --")
            def regressor_fn(q, dq, ddq):
                return newton_euler_regressor(kin, q, dq, ddq)
        else:
            log.info("-- Stage 5: Setting up Euler-Lagrange regressor --")
            cache_dir = str(self.output_dir / "el_cache")
            regressor_fn, el_kept_cols = euler_lagrange_regressor_builder(
                kin, cache_dir
            )
            log.info("EL regressor: %d kept columns.", len(el_kept_cols))

        augmented_regressor_fn = make_augmented_regressor(
            regressor_fn, cfg["friction"]["model"]
        )

        true_nominal_params = self._nominal_parameter_vector(
            kin, el_kept_cols
        )

        return {
            "robot": robot,
            "kin": kin,
            "q_lim": q_lim,
            "dq_lim": dq_lim,
            "ddq_lim": ddq_lim,
            "tau_lim": tau_lim,
            "torque_limit_sources": torque_limit_sources,
            "regressor_fn": regressor_fn,
            "augmented_regressor_fn": augmented_regressor_fn,
            "el_kept_cols": el_kept_cols,
            "torque_method": torque_method,
            "torque_required": torque_required,
            "true_nominal_params": true_nominal_params,
            # These will be populated by later stages:
            "exc_result": None,
            "nominal_params_used": true_nominal_params.copy(),
            "sequential_history": [],
            "identification": None,
            "q_data": None,
            "dq_data": None,
            "ddq_data": None,
            "tau_data": None,
            "data_fs": None,
        }

    def _run_stage_5(self, ctx: dict) -> None:
        """Stage 6: Excitation trajectory optimisation.

        Method name is kept for backward-compatibility; the documented stage
        number is 6 (excitation). Populates ctx with ``exc_result``,
        ``nominal_params_used``, ``sequential_history``, and
        (for ``sequential_redesign``) also ``identification``.
        """
        log = self.logger
        cfg = self.cfg
        kin = ctx["kin"]
        q_lim = ctx["q_lim"]
        dq_lim = ctx["dq_lim"]
        ddq_lim = ctx["ddq_lim"]
        tau_lim = ctx["tau_lim"]
        torque_method = ctx["torque_method"]
        torque_required = ctx["torque_required"]
        regressor_fn = ctx["regressor_fn"]
        true_nominal_params = ctx["true_nominal_params"]

        nominal_params_used = true_nominal_params.copy()
        sequential_history = []

        if torque_method == "sequential_redesign":
            log.info("-- Stage 6: Sequential torque-limited excitation redesign --")
            torque_cfg = cfg["excitation"].get("torque_constraint", {})
            current_nominal = true_nominal_params.copy()
            max_iterations = int(torque_cfg.get("max_iterations", 3))
            convergence_tol = float(torque_cfg.get("convergence_tol", 0.01))

            for iteration in range(max_iterations):
                nominal_params_used = current_nominal.copy()
                iter_cfg = deepcopy(cfg["excitation"])
                iter_cfg["torque_constraint_method"] = "nominal_hard"
                log.info("Sequential redesign iteration %d/%d",
                         iteration + 1, max_iterations)
                exc_result = optimise_excitation(
                    kin, iter_cfg, q_lim, dq_lim, ddq_lim,
                    friction_model=cfg["friction"]["model"],
                    regressor_fn=regressor_fn if cfg["method"] == "euler_lagrange" else None,
                    tau_lim=tau_lim,
                    nominal_params=nominal_params_used,
                )

                log.info("-- Stage 7 (iter %d): Generating trajectory data --",
                         iteration + 1)
                q_data, dq_data, ddq_data, tau_data, data_fs = \
                    self._load_or_generate_data(ctx, exc_result)

                identification = self._solve_identification_pass(ctx, q_data,
                    dq_data, ddq_data, tau_data, data_fs)
                replay, _ = self._validate_torque_models(
                    ctx, exc_result, nominal_params_used,
                    identification["pi_identified_full"],
                    identification["pi_corrected"],
                    "nominal_hard",
                )
                updated_nominal = np.asarray(identification["pi_corrected"],
                                             dtype=float)
                rel_change = np.linalg.norm(
                    updated_nominal - current_nominal
                ) / max(np.linalg.norm(current_nominal), 1e-12)
                sequential_history.append({
                    "iteration": iteration + 1,
                    "nominal_params": nominal_params_used.tolist(),
                    "identified_params": identification["pi_identified_full"].tolist(),
                    "max_nominal_torque_ratio": (
                        float(replay["nominal_summary"]["max_ratio"])
                        if replay is not None and "nominal_summary" in replay
                        else None
                    ),
                    "max_identified_torque_ratio": (
                        float(replay["identified_summary"]["max_ratio"])
                        if replay is not None and "identified_summary" in replay
                        else None
                    ),
                    "relative_model_change": float(rel_change),
                })
                current_nominal = updated_nominal
                if rel_change <= convergence_tol:
                    log.info("Sequential redesign converged after %d iterations.",
                             iteration + 1)
                    break

            ctx["exc_result"] = exc_result
            ctx["nominal_params_used"] = nominal_params_used
            ctx["sequential_history"] = sequential_history
            ctx["identification"] = identification
            # Store the final iteration's data
            ctx["q_data"] = q_data
            ctx["dq_data"] = dq_data
            ctx["ddq_data"] = ddq_data
            ctx["tau_data"] = tau_data
            ctx["data_fs"] = data_fs

        else:
            log.info("-- Stage 6: Excitation trajectory optimisation --")
            exc_result = optimise_excitation(
                kin, cfg["excitation"], q_lim, dq_lim, ddq_lim,
                friction_model=cfg["friction"]["model"],
                regressor_fn=regressor_fn if cfg["method"] == "euler_lagrange" else None,
                tau_lim=tau_lim,
                nominal_params=nominal_params_used if torque_required else None,
            )
            ctx["exc_result"] = exc_result
            ctx["nominal_params_used"] = nominal_params_used
            ctx["sequential_history"] = sequential_history

        log.info("Excitation cost: %.6f", ctx["exc_result"]["cost"])

    def _run_stage_6(self, ctx: dict) -> None:
        """Stage 7: Data generation (synthetic or external).

        Method name is kept for backward-compatibility; the documented stage
        number is 7 (data generation). Populates ctx with ``q_data``,
        ``dq_data``, ``ddq_data``, ``tau_data``, ``data_fs``. For
        ``sequential_redesign``, data was already generated per iteration
        inside the Stage 6 redesign loop.
        """
        log = self.logger
        torque_method = ctx["torque_method"]

        if torque_method == "sequential_redesign":
            # Data already generated inside the sequential loop
            log.info("-- Stage 7: Data generation (done in sequential redesign) --")
            return

        log.info("-- Stage 7: Generating trajectory data --")
        q_data, dq_data, ddq_data, tau_data, data_fs = \
            self._load_or_generate_data(ctx, ctx["exc_result"])
        ctx["q_data"] = q_data
        ctx["dq_data"] = dq_data
        ctx["ddq_data"] = ddq_data
        ctx["tau_data"] = tau_data
        ctx["data_fs"] = data_fs

    def _run_stages_7_to_11(self, ctx: dict) -> None:
        """Identification and post-identification stages.

        Covers documented Stages 8-12: observation matrix, base-parameter
        reduction, identification, feasibility, torque validation, save
        outputs, and (when ``export.enabled=true``) the optional
        adapted-URDF / friction-sidecar export as a Stage 12 sub-step.

        The ``_7_to_11`` suffix in the method name is retained for
        backward-compatibility with external references.
        """
        log = self.logger
        cfg = self.cfg
        kin = ctx["kin"]
        torque_method = ctx["torque_method"]
        torque_limit_sources = ctx["torque_limit_sources"]
        exc_result = ctx["exc_result"]
        nominal_params_used = ctx["nominal_params_used"]
        sequential_history = ctx["sequential_history"]

        # For sequential_redesign, identification was already done per
        # iteration inside the Stage 6 redesign loop. For all other modes,
        # run it now.
        if torque_method == "sequential_redesign":
            identification = ctx["identification"]
        else:
            identification = self._solve_identification_pass(
                ctx,
                ctx["q_data"], ctx["dq_data"], ctx["ddq_data"],
                ctx["tau_data"], ctx["data_fs"],
            )

        # Save excitation trajectory (unless already saved in excitation_only
        # mode — but in full/resume mode we always save it here)
        self._save_excitation_trajectory(ctx)

        # Torque validation
        torque_validation, torque_summary = self._validate_torque_models(
            ctx, exc_result, nominal_params_used,
            identification["pi_identified_full"],
            identification["pi_corrected"],
            "nominal_hard" if torque_method in {
                "soft_penalty", "sequential_redesign", "none"
            } else torque_method,
        )

        if torque_validation is not None:
            torque_path = self.output_dir / "torque_limit_validation.npz"
            np.savez(
                str(torque_path),
                t=torque_validation["t"],
                limit_lower=torque_validation["limit_lower"],
                limit_upper=torque_validation["limit_upper"],
                tau_nominal=torque_validation.get("tau_nominal"),
                tau_identified=torque_validation.get("tau_identified"),
                tau_corrected=torque_validation.get("tau_corrected"),
                nominal_ratio=(
                    torque_validation["nominal_summary"]["ratio"]
                    if "nominal_summary" in torque_validation else None
                ),
                identified_ratio=(
                    torque_validation["identified_summary"]["ratio"]
                    if "identified_summary" in torque_validation else None
                ),
                corrected_ratio=(
                    torque_validation["corrected_summary"]["ratio"]
                    if "corrected_summary" in torque_validation else None
                ),
                design_lower=torque_validation.get("design_lower"),
                design_upper=torque_validation.get("design_upper"),
                design_upper_margin=torque_validation.get("design_upper_margin"),
                design_lower_margin=torque_validation.get("design_lower_margin"),
                design_pass=torque_validation.get("design_pass"),
                chance_quantile=torque_validation.get("chance_quantile"),
                torque_limit_source=np.array(torque_limit_sources, dtype=object),
                sequential_history=np.array(sequential_history, dtype=object),
            )
            log.info("Torque validation saved to %s", torque_path)

        log.info("-- Stage 12: Saving outputs --")
        results = {
            "pi_identified": identification["pi_identified_full"],
            "pi_base": identification["pi_base"],
            "P_matrix": identification["P_mat"],
            "kept_cols": np.array(identification["kept_cols"]),
            "residual": identification["residual"],
            "feasible": identification["feasible"],
            "pi_corrected": identification["pi_corrected"],
            "method": cfg["method"],
            "friction_model": cfg["friction"]["model"],
            "nDoF": kin.nDoF,
            "solved_in_full_space": identification["info"].get(
                "solved_in_full_space", False
            ),
            "torque_constraint_method": torque_method,
            "torque_limit_source": np.array(torque_limit_sources, dtype=object)
            if torque_limit_sources is not None
            else np.array([], dtype=object),
            "sequential_history": np.array(sequential_history, dtype=object),
        }
        out_path = self.output_dir / "identification_results.npz"
        np.savez(str(out_path), **results)
        log.info("Results saved to %s", out_path)

        summary_path = self.output_dir / "results_summary.json"
        summary = {
            "method": cfg["method"],
            "nDoF": kin.nDoF,
            "n_base_params": identification["rank"],
            "n_full_params": len(self._nominal_parameter_vector(
                kin, ctx["el_kept_cols"]
            )),
            "residual": float(identification["residual"]),
            "feasible": bool(identification["feasible"]),
            "friction_model": cfg["friction"]["model"],
            "n_friction_params": identification["n_fric"],
            "solver": identification["solver"],
            "pi_identified": identification["pi_identified_full"].tolist(),
            "torque_constraint_method": torque_method,
            "torque_limit_source": torque_limit_sources,
            "torque_nominal_pass": (
                None if torque_summary is None
                else torque_summary["torque_nominal_pass"]
            ),
            "torque_identified_pass": (
                None if torque_summary is None
                else torque_summary["torque_identified_pass"]
            ),
            "torque_corrected_pass": (
                None if torque_summary is None
                else torque_summary["torque_corrected_pass"]
            ),
            "max_nominal_torque_ratio": (
                None if torque_summary is None
                else torque_summary["max_nominal_torque_ratio"]
            ),
            "max_identified_torque_ratio": (
                None if torque_summary is None
                else torque_summary["max_identified_torque_ratio"]
            ),
            "worst_joint": (
                None if torque_summary is None
                else torque_summary["worst_joint"]
            ),
            "worst_time_s": (
                None if torque_summary is None
                else torque_summary["worst_time_s"]
            ),
            "sequential_history": sequential_history,
        }
        if (torque_summary is not None
                and torque_summary.get("design_summary") is not None):
            summary["torque_design_summary"] = torque_summary["design_summary"]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info("Summary saved to %s", summary_path)

        # ---- Stage 12 (cont.): optional adapted-URDF export ---------------
        # Stage 12 is "save outputs". When export.enabled=true, the same
        # stage additionally writes a simulation-ready URDF (and an optional
        # asymmetric-friction sidecar) alongside the npz/json/log artifacts.
        export_cfg = cfg.get("export") or {}
        if export_cfg.get("enabled", False):
            from .urdf_exporter import export_adapted_urdf

            log.info("Stage 12: exporting adapted URDF")
            urdf_name = export_cfg.get("urdf_filename", "adapted_robot.urdf")
            sidecar_enabled = (
                export_cfg.get("friction_sidecar", True)
                and cfg["friction"]["model"] != "none"
            )
            sidecar_name = (
                export_cfg.get(
                    "friction_sidecar_filename", "adapted_friction.json"
                )
                if sidecar_enabled
                else None
            )
            export_meta = export_adapted_urdf(
                input_urdf_path=cfg["urdf_path"],
                pi_full=identification["pi_corrected"],
                n_dof=kin.nDoF,
                friction_model=cfg["friction"]["model"],
                output_urdf_path=self.output_dir / urdf_name,
                friction_sidecar_path=(
                    self.output_dir / sidecar_name if sidecar_name else None
                ),
                logger=log,
            )
            summary["export"] = {
                "adapted_urdf_path": export_meta["adapted_urdf_path"],
                "friction_sidecar_path": export_meta["friction_sidecar_path"],
                "n_friction_params": export_meta["n_friction_params"],
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _save_checkpoint(self, ctx: dict) -> Path:
        """Serialize excitation + data state to checkpoint files."""
        checkpoint_path = self.output_dir / "checkpoint.npz"
        exc = ctx["exc_result"]
        np.savez(
            str(checkpoint_path),
            exc_params=exc["params"],
            exc_freqs=exc["freqs"],
            exc_q0=exc["q0"],
            exc_cost=np.float64(exc["cost"]),
            exc_basis=np.array(exc["basis"]),
            exc_optimize_phase=np.array(exc["optimize_phase"]),
            exc_torque_constraint_method=np.array(
                exc.get("torque_constraint_method", "none")
            ),
            q_data=ctx["q_data"],
            dq_data=ctx["dq_data"],
            ddq_data=ctx["ddq_data"],
            tau_data=ctx["tau_data"],
            data_fs=np.float64(ctx["data_fs"]),
            nominal_params_used=ctx["nominal_params_used"],
            sequential_history=np.array(
                ctx["sequential_history"], dtype=object
            ),
        )

        config_path = self.output_dir / "checkpoint_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, default=str)

        self.logger.info("Checkpoint saved to %s", checkpoint_path)
        return checkpoint_path

    @staticmethod
    def _load_checkpoint(checkpoint_dir: Path) -> dict:
        """Load checkpoint data from a previous run."""
        cp_file = checkpoint_dir / "checkpoint.npz"
        if not cp_file.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {cp_file}. "
                f"Make sure the checkpoint_dir points to a directory "
                f"produced by an excitation_only run."
            )

        cp = np.load(str(cp_file), allow_pickle=True)

        exc_result = {
            "params": cp["exc_params"],
            "freqs": cp["exc_freqs"],
            "q0": cp["exc_q0"],
            "cost": float(cp["exc_cost"]),
            "basis": str(cp["exc_basis"]),
            "optimize_phase": bool(cp["exc_optimize_phase"]),
            "torque_constraint_method": str(
                cp["exc_torque_constraint_method"]
            ),
        }

        seq_hist_raw = cp["sequential_history"]
        if seq_hist_raw.ndim == 0:
            # Scalar array wrapping an object — unwrap
            seq_hist = seq_hist_raw.item()
            if seq_hist is None:
                seq_hist = []
        else:
            seq_hist = seq_hist_raw.tolist() if seq_hist_raw.size > 0 else []

        return {
            "exc_result": exc_result,
            "q_data": cp["q_data"],
            "dq_data": cp["dq_data"],
            "ddq_data": cp["ddq_data"],
            "tau_data": cp["tau_data"],
            "data_fs": float(cp["data_fs"]),
            "nominal_params_used": cp["nominal_params_used"],
            "sequential_history": seq_hist,
        }

    # ------------------------------------------------------------------
    # Helper methods (formerly closures inside run())
    # ------------------------------------------------------------------

    def _rigid_nominal_vector(self, kin, el_kept_cols):
        """Return the rigid-body nominal parameter vector."""
        if self.cfg["method"] == "euler_lagrange" and el_kept_cols is not None:
            return kin.PI.flatten()[el_kept_cols]
        return kin.PI.flatten()

    def _nominal_parameter_vector(self, kin, el_kept_cols,
                                  seed_vector=None):
        """Build nominal parameter vector (rigid + friction padding)."""
        if seed_vector is None:
            rigid = self._rigid_nominal_vector(kin, el_kept_cols)
        else:
            rigid = np.asarray(seed_vector, dtype=float)
        return build_nominal_parameter_vector(
            rigid, kin.nDoF, self.cfg["friction"]["model"],
        )

    def _excitation_time_series(self, exc_result):
        """Evaluate the Fourier trajectory at a dense time grid."""
        freqs = exc_result["freqs"]
        q0 = exc_result["q0"]
        basis = exc_result["basis"]
        opt_phase = exc_result["optimize_phase"]
        params = exc_result["params"]

        cfg = self.cfg
        f0 = cfg["excitation"]["base_frequency_hz"]
        tf = cfg["excitation"].get("trajectory_duration_periods", 1) / f0
        data_fs = 2.0 * freqs[-1] * 10
        t_data = np.arange(0.0, tf, 1.0 / data_fs)
        q_data_t, dq_data_t, ddq_data_t = fourier_trajectory(
            params, freqs, t_data, q0, basis, opt_phase
        )
        return t_data, q_data_t.T, dq_data_t.T, ddq_data_t.T, data_fs

    def _load_or_generate_data(self, ctx, exc_result):
        """Load external data or generate synthetic trajectory + torques."""
        log = self.logger
        cfg = self.cfg
        kin = ctx["kin"]
        augmented_regressor_fn = ctx["augmented_regressor_fn"]
        true_nominal_params = ctx["true_nominal_params"]

        data_file = cfg["identification"].get("data_file")
        if data_file:
            log.info("Loading external data from %s", data_file)
            data = np.load(data_file)
            q_data = data["q"]
            dq_data = data["dq"]
            ddq_data = data["ddq"]
            tau_data = data["tau"]
            data_fs = float(data.get("fs", 1e4))
            return q_data, dq_data, ddq_data, tau_data, data_fs

        log.info("Generating synthetic trajectory from excitation parameters.")
        _, q_data, dq_data, ddq_data, data_fs = \
            self._excitation_time_series(exc_result)
        N = q_data.shape[0]
        tau_data = np.zeros((N, kin.nDoF))
        for k in range(N):
            tau_data[k] = augmented_regressor_fn(
                q_data[k], dq_data[k], ddq_data[k]
            ) @ true_nominal_params
        log.info("Generated %d data samples at %.1f Hz", N, data_fs)
        return q_data, dq_data, ddq_data, tau_data, data_fs

    def _solve_identification_pass(self, ctx, q_data, dq_data, ddq_data,
                                   tau_data, data_fs):
        """Documented Stages 8-11: observation matrix → base params → solve → feasibility."""
        log = self.logger
        cfg = self.cfg
        kin = ctx["kin"]
        regressor_fn = ctx["regressor_fn"]
        el_kept_cols = ctx["el_kept_cols"]

        log.info("-- Stage 8: Building observation matrix --")
        W, tau_vec = build_observation_matrix(
            q_data, dq_data, ddq_data, tau_data,
            regressor_fn, cfg, data_fs
        )
        log.info("W shape: %s", W.shape)

        log.info("-- Stage 9: Base parameter reduction --")
        pi_full = self._nominal_parameter_vector(kin, el_kept_cols)
        W_base, P_mat, kept_cols, rank, pi_base = compute_base_parameters(
            W, pi_full
        )
        log.info("Base parameters: %d (from %d full)", rank, len(pi_full))

        log.info("-- Stage 10: Solving identification --")
        solver = cfg["identification"]["solver"]
        feas_method = cfg["identification"]["feasibility_method"]
        self._id_warnings_emitted = _emit_identification_warnings(
            cfg, log, self._id_warnings_emitted
        )
        bounds_opt = None
        cfg_bounds = cfg["identification"].get("parameter_bounds")
        if isinstance(cfg_bounds, list) and len(cfg_bounds) == 2:
            lb = np.array(cfg_bounds[0])
            ub = np.array(cfg_bounds[1])
            if len(lb) == rank and len(ub) == rank:
                bounds_opt = (lb, ub)
                if solver == "ols":
                    solver = "bounded_ls"
                    log.info("Switching to bounded_ls due to user parameter_bounds.")
        elif cfg_bounds is True:
            lb = pi_base - np.abs(pi_base) * 0.5 - 1e-3
            ub = pi_base + np.abs(pi_base) * 0.5 + 1e-3
            bounds_opt = (lb, ub)
            if solver == "ols":
                solver = "bounded_ls"
                log.info("Switching to bounded_ls due to parameter_bounds=true")

        pi_hat, residual, info = solve_identification(
            W_base, tau_vec, solver=solver, bounds=bounds_opt,
            nDoF=kin.nDoF, feasibility_method=feas_method,
            P_mat=P_mat if feas_method != "none" else None,
        )
        log.info("Identified %d parameters, residual=%.6e",
                 len(pi_hat), residual)

        if info.get("solved_in_full_space"):
            pi_identified_full = pi_hat
        else:
            pi_identified_full = np.linalg.pinv(P_mat) @ pi_hat

        recon_err = np.linalg.norm(
            W_base @ (P_mat @ pi_identified_full) - W_base @ pi_hat
            if not info.get("solved_in_full_space")
            else W_base @ P_mat @ pi_identified_full - tau_vec
        )
        log.debug(
            "Reconstruction consistency: ||W_b*P*pi_recon - ref|| = %.3e",
            recon_err,
        )

        log.info("-- Stage 11: Feasibility check --")
        report, feasible, pi_corrected = check_feasibility(
            pi_identified_full, kin.nDoF, method=feas_method
        )
        for r in report:
            if r["feasible"]:
                log.info("  Link %d: FEASIBLE (mass=%.4f)",
                         r["link"], r["mass"])
            else:
                log.warning("  Link %d: INFEASIBLE -- %s",
                            r["link"], "; ".join(r["issues"]))

        pi_corrected = _clamp_negative_viscous_damping(
            pi_corrected, kin.nDoF, cfg["friction"]["model"], log
        )

        return {
            "W": W,
            "tau_vec": tau_vec,
            "W_base": W_base,
            "P_mat": P_mat,
            "kept_cols": kept_cols,
            "rank": rank,
            "pi_base": (pi_hat if not info.get("solved_in_full_space")
                        else P_mat @ pi_identified_full),
            "pi_identified_full": pi_identified_full,
            "pi_corrected": pi_corrected,
            "residual": residual,
            "report": report,
            "feasible": feasible,
            "info": info,
            "solver": solver,
            "n_fric": friction_param_count(kin.nDoF, cfg["friction"]["model"]),
        }

    def _validate_torque_models(self, ctx, exc_result, nominal_params_used,
                                identified_params, corrected_params,
                                design_method):
        """Dense torque-limit validation across the trajectory."""
        cfg = self.cfg
        tau_lim = ctx["tau_lim"]
        augmented_regressor_fn = ctx["augmented_regressor_fn"]
        torque_limit_sources = ctx["torque_limit_sources"]

        if tau_lim is None:
            return None, None

        t_dense = validation_time_vector(
            exc_result["freqs"],
            cfg["excitation"]["base_frequency_hz"],
            cfg["excitation"].get("trajectory_duration_periods", 1),
            cfg["excitation"].get("torque_validation_oversample_factor", 1),
        )
        q_dense, dq_dense, ddq_dense = fourier_trajectory(
            exc_result["params"],
            exc_result["freqs"],
            t_dense,
            exc_result["q0"],
            exc_result["basis"],
            exc_result["optimize_phase"],
        )
        replay = replay_torque_models(
            q_dense, dq_dense, ddq_dense, augmented_regressor_fn, tau_lim,
            "actuator_envelope" if design_method == "actuator_envelope"
            else "nominal_hard",
            cfg["excitation"].get("torque_constraint", {}),
            nominal_params=nominal_params_used,
            identified_params=identified_params,
            corrected_params=corrected_params,
        )

        design_summary = None
        if design_method in {
            "nominal_hard", "robust_box", "chance", "actuator_envelope"
        }:
            design_data = compute_torque_design_data(
                q_dense, dq_dense, ddq_dense,
                augmented_regressor_fn,
                nominal_params_used,
                tau_lim,
                design_method,
                cfg["excitation"].get("torque_constraint", {}),
            )
            replay["design_lower"] = design_data["design_lower"]
            replay["design_upper"] = design_data["design_upper"]
            replay["design_upper_margin"] = design_data["design_upper_margin"]
            replay["design_lower_margin"] = design_data["design_lower_margin"]
            replay["design_pass"] = np.array(design_data["design_pass"])
            if design_data["quantile"] is not None:
                replay["chance_quantile"] = np.array(design_data["quantile"])
            design_summary = {
                "design_pass": bool(design_data["design_pass"]),
                "quantile": (
                    None if design_data["quantile"] is None
                    else float(design_data["quantile"])
                ),
            }

        worst_source = replay.get("identified_summary",
                                  replay.get("nominal_summary"))
        worst_joint = None
        worst_time_s = None
        if worst_source is not None:
            worst_joint = int(worst_source["worst_joint"])
            worst_time_s = float(t_dense[worst_source["worst_time_index"]])

        summary = {
            "torque_constraint_method": cfg["excitation"].get(
                "torque_constraint_method", "none"
            ),
            "torque_limit_source": torque_limit_sources,
            "torque_nominal_pass": (
                bool(replay["nominal_summary"]["pass"])
                if "nominal_summary" in replay else None
            ),
            "torque_identified_pass": (
                bool(replay["identified_summary"]["pass"])
                if "identified_summary" in replay else None
            ),
            "torque_corrected_pass": (
                bool(replay["corrected_summary"]["pass"])
                if "corrected_summary" in replay else None
            ),
            "max_nominal_torque_ratio": (
                float(replay["nominal_summary"]["max_ratio"])
                if "nominal_summary" in replay else None
            ),
            "max_identified_torque_ratio": (
                float(replay["identified_summary"]["max_ratio"])
                if "identified_summary" in replay else None
            ),
            "worst_joint": worst_joint,
            "worst_time_s": worst_time_s,
            "design_summary": design_summary,
        }
        replay["t"] = t_dense
        return replay, summary

    def _save_excitation_trajectory(self, ctx: dict) -> None:
        """Save excitation_trajectory.npz (for PyBullet validation compat)."""
        exc_result = ctx["exc_result"]
        q_lim = ctx["q_lim"]
        dq_lim = ctx["dq_lim"]
        ddq_lim = ctx["ddq_lim"]

        exc_path = self.output_dir / "excitation_trajectory.npz"
        t_exc, q_exc_T, dq_exc_T, ddq_exc_T, _ = \
            self._excitation_time_series(exc_result)
        np.savez(
            str(exc_path),
            **exc_result,
            t=t_exc,
            q=q_exc_T.T,
            dq=dq_exc_T.T,
            ddq=ddq_exc_T.T,
            q_lim=q_lim,
            dq_lim=dq_lim if dq_lim is not None else np.zeros((0, 2)),
            ddq_lim=ddq_lim if ddq_lim is not None else np.zeros((0, 2)),
        )
        self.logger.info("Saved excitation trajectory to %s", exc_path)
