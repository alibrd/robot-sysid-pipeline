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

    def run(self):
        t0 = time.time()
        log = self.logger
        cfg = self.cfg

        try:
            log.info("=" * 60)
            log.info("SYSTEM IDENTIFICATION PIPELINE")
            log.info("=" * 60)
            log.info("Config: method=%s, friction=%s, solver=%s",
                     cfg["method"], cfg["friction"]["model"],
                     cfg["identification"]["solver"])

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

            log.info("-- Stage 4: Setting up regressor (%s) --", cfg["method"])
            el_regressor_fn = None
            el_kept_cols = None
            if cfg["method"] == "newton_euler":
                def regressor_fn(q, dq, ddq):
                    return newton_euler_regressor(kin, q, dq, ddq)
            else:
                cache_dir = str(self.output_dir / "el_cache")
                el_regressor_fn, el_kept_cols = euler_lagrange_regressor_builder(
                    kin, cache_dir
                )
                regressor_fn = el_regressor_fn
                log.info("EL regressor: %d kept columns.", len(el_kept_cols))

            augmented_regressor_fn = make_augmented_regressor(
                regressor_fn, cfg["friction"]["model"]
            )

            def _rigid_nominal_vector():
                if cfg["method"] == "euler_lagrange" and el_kept_cols is not None:
                    return kin.PI.flatten()[el_kept_cols]
                return kin.PI.flatten()

            def _nominal_parameter_vector(seed_vector=None):
                rigid = _rigid_nominal_vector() if seed_vector is None else np.asarray(seed_vector, dtype=float)
                return build_nominal_parameter_vector(
                    rigid,
                    kin.nDoF,
                    cfg["friction"]["model"],
                )

            def _excitation_time_series(exc_result):
                freqs = exc_result["freqs"]
                q0 = exc_result["q0"]
                basis = exc_result["basis"]
                opt_phase = exc_result["optimize_phase"]
                params = exc_result["params"]

                f0 = cfg["excitation"]["base_frequency_hz"]
                tf = cfg["excitation"].get("trajectory_duration_periods", 1) / f0
                data_fs = 2.0 * freqs[-1] * 10
                t_data = np.arange(0.0, tf, 1.0 / data_fs)
                q_data_t, dq_data_t, ddq_data_t = fourier_trajectory(
                    params, freqs, t_data, q0, basis, opt_phase
                )
                return t_data, q_data_t.T, dq_data_t.T, ddq_data_t.T, data_fs

            true_nominal_params = _nominal_parameter_vector()

            def _load_or_generate_data(exc_result):
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
                _, q_data, dq_data, ddq_data, data_fs = _excitation_time_series(exc_result)
                N = q_data.shape[0]
                tau_data = np.zeros((N, kin.nDoF))
                for k in range(N):
                    tau_data[k] = augmented_regressor_fn(
                        q_data[k], dq_data[k], ddq_data[k]
                    ) @ true_nominal_params
                log.info("Generated %d data samples at %.1f Hz", N, data_fs)
                return q_data, dq_data, ddq_data, tau_data, data_fs

            def _solve_identification_pass(q_data, dq_data, ddq_data, tau_data, data_fs):
                log.info("-- Stage 7: Building observation matrix --")
                W, tau_vec = build_observation_matrix(
                    q_data, dq_data, ddq_data, tau_data,
                    regressor_fn, cfg, data_fs
                )
                log.info("W shape: %s", W.shape)

                log.info("-- Stage 8: Base parameter reduction --")
                pi_full = _nominal_parameter_vector()
                W_base, P_mat, kept_cols, rank, pi_base = compute_base_parameters(W, pi_full)
                log.info("Base parameters: %d (from %d full)", rank, len(pi_full))

                log.info("-- Stage 9: Solving identification --")
                solver = cfg["identification"]["solver"]
                feas_method = cfg["identification"]["feasibility_method"]
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
                log.info("Identified %d parameters, residual=%.6e", len(pi_hat), residual)

                if info.get("solved_in_full_space"):
                    pi_identified_full = pi_hat
                else:
                    pi_identified_full = np.linalg.pinv(P_mat) @ pi_hat

                recon_err = np.linalg.norm(
                    W_base @ (P_mat @ pi_identified_full) - W_base @ pi_hat
                    if not info.get("solved_in_full_space")
                    else W_base @ P_mat @ pi_identified_full - tau_vec
                )
                log.debug("Reconstruction consistency: ||W_b*P*pi_recon - ref|| = %.3e",
                          recon_err)

                log.info("-- Stage 10: Feasibility check --")
                report, feasible, pi_corrected = check_feasibility(
                    pi_identified_full, kin.nDoF, method=feas_method
                )
                for r in report:
                    if r["feasible"]:
                        log.info("  Link %d: FEASIBLE (mass=%.4f)", r["link"], r["mass"])
                    else:
                        log.warning("  Link %d: INFEASIBLE -- %s",
                                    r["link"], "; ".join(r["issues"]))

                return {
                    "W": W,
                    "tau_vec": tau_vec,
                    "W_base": W_base,
                    "P_mat": P_mat,
                    "kept_cols": kept_cols,
                    "rank": rank,
                    "pi_base": pi_hat if not info.get("solved_in_full_space") else P_mat @ pi_identified_full,
                    "pi_identified_full": pi_identified_full,
                    "pi_corrected": pi_corrected,
                    "residual": residual,
                    "report": report,
                    "feasible": feasible,
                    "info": info,
                    "solver": solver,
                    "n_fric": friction_param_count(kin.nDoF, cfg["friction"]["model"]),
                }

            def _validate_torque_models(exc_result, nominal_params_used, identified_params, corrected_params,
                                        design_method):
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
                    "actuator_envelope" if design_method == "actuator_envelope" else "nominal_hard",
                    cfg["excitation"].get("torque_constraint", {}),
                    nominal_params=nominal_params_used,
                    identified_params=identified_params,
                    corrected_params=corrected_params,
                )

                design_summary = None
                if design_method in {"nominal_hard", "robust_box", "chance", "actuator_envelope"}:
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
                        "quantile": None if design_data["quantile"] is None else float(design_data["quantile"]),
                    }

                worst_source = replay.get("identified_summary", replay.get("nominal_summary"))
                worst_joint = None
                worst_time_s = None
                if worst_source is not None:
                    worst_joint = int(worst_source["worst_joint"])
                    worst_time_s = float(t_dense[worst_source["worst_time_index"]])

                summary = {
                    "torque_constraint_method": cfg["excitation"].get("torque_constraint_method", "none"),
                    "torque_limit_source": torque_limit_sources,
                    "torque_nominal_pass": bool(replay["nominal_summary"]["pass"]) if "nominal_summary" in replay else None,
                    "torque_identified_pass": bool(replay["identified_summary"]["pass"]) if "identified_summary" in replay else None,
                    "torque_corrected_pass": bool(replay["corrected_summary"]["pass"]) if "corrected_summary" in replay else None,
                    "max_nominal_torque_ratio": float(replay["nominal_summary"]["max_ratio"]) if "nominal_summary" in replay else None,
                    "max_identified_torque_ratio": float(replay["identified_summary"]["max_ratio"]) if "identified_summary" in replay else None,
                    "worst_joint": worst_joint,
                    "worst_time_s": worst_time_s,
                    "design_summary": design_summary,
                }
                replay["t"] = t_dense
                return replay, summary

            nominal_params_used = true_nominal_params.copy()
            exc_result = None
            identification = None
            sequential_history = []

            if torque_method == "sequential_redesign":
                log.info("-- Stage 5: Sequential torque-limited excitation redesign --")
                torque_cfg = cfg["excitation"].get("torque_constraint", {})
                current_nominal = true_nominal_params.copy()
                max_iterations = int(torque_cfg.get("max_iterations", 3))
                convergence_tol = float(torque_cfg.get("convergence_tol", 0.01))

                for iteration in range(max_iterations):
                    nominal_params_used = current_nominal.copy()
                    iter_cfg = deepcopy(cfg["excitation"])
                    iter_cfg["torque_constraint_method"] = "nominal_hard"
                    log.info("Sequential redesign iteration %d/%d", iteration + 1, max_iterations)
                    exc_result = optimise_excitation(
                        kin, iter_cfg, q_lim, dq_lim, ddq_lim,
                        friction_model=cfg["friction"]["model"],
                        regressor_fn=regressor_fn if cfg["method"] == "euler_lagrange" else None,
                        tau_lim=tau_lim,
                        nominal_params=nominal_params_used,
                    )
                    q_data, dq_data, ddq_data, tau_data, data_fs = _load_or_generate_data(exc_result)
                    identification = _solve_identification_pass(q_data, dq_data, ddq_data, tau_data, data_fs)
                    replay, _ = _validate_torque_models(
                        exc_result,
                        nominal_params_used,
                        identification["pi_identified_full"],
                        identification["pi_corrected"],
                        "nominal_hard",
                    )
                    updated_nominal = np.asarray(identification["pi_corrected"], dtype=float)
                    rel_change = np.linalg.norm(updated_nominal - current_nominal) / max(
                        np.linalg.norm(current_nominal), 1e-12
                    )
                    sequential_history.append({
                        "iteration": iteration + 1,
                        "nominal_params": nominal_params_used.tolist(),
                        "identified_params": identification["pi_identified_full"].tolist(),
                        "max_nominal_torque_ratio": (
                            float(replay["nominal_summary"]["max_ratio"])
                            if replay is not None and "nominal_summary" in replay else None
                        ),
                        "max_identified_torque_ratio": (
                            float(replay["identified_summary"]["max_ratio"])
                            if replay is not None and "identified_summary" in replay else None
                        ),
                        "relative_model_change": float(rel_change),
                    })
                    current_nominal = updated_nominal
                    if rel_change <= convergence_tol:
                        log.info("Sequential redesign converged after %d iterations.", iteration + 1)
                        break
            else:
                log.info("-- Stage 5: Excitation trajectory optimisation --")
                exc_result = optimise_excitation(
                    kin, cfg["excitation"], q_lim, dq_lim, ddq_lim,
                    friction_model=cfg["friction"]["model"],
                    regressor_fn=regressor_fn if cfg["method"] == "euler_lagrange" else None,
                    tau_lim=tau_lim,
                    nominal_params=nominal_params_used if torque_required else None,
                )
                q_data, dq_data, ddq_data, tau_data, data_fs = _load_or_generate_data(exc_result)
                identification = _solve_identification_pass(q_data, dq_data, ddq_data, tau_data, data_fs)

            log.info("Excitation cost: %.6f", exc_result["cost"])

            exc_path = self.output_dir / "excitation_trajectory.npz"
            np.savez(str(exc_path), **exc_result)
            log.info("Saved excitation parameters to %s", exc_path)

            torque_validation, torque_summary = _validate_torque_models(
                exc_result,
                nominal_params_used,
                identification["pi_identified_full"],
                identification["pi_corrected"],
                "nominal_hard" if torque_method in {"soft_penalty", "sequential_redesign", "none"} else torque_method,
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

            log.info("-- Stage 11: Saving results --")
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
                "solved_in_full_space": identification["info"].get("solved_in_full_space", False),
                "torque_constraint_method": torque_method,
                "torque_limit_source": np.array(torque_limit_sources, dtype=object)
                if torque_limit_sources is not None else np.array([], dtype=object),
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
                "n_full_params": len(_nominal_parameter_vector()),
                "residual": float(identification["residual"]),
                "feasible": bool(identification["feasible"]),
                "friction_model": cfg["friction"]["model"],
                "n_friction_params": identification["n_fric"],
                "solver": identification["solver"],
                "pi_identified": identification["pi_identified_full"].tolist(),
                "torque_constraint_method": torque_method,
                "torque_limit_source": torque_limit_sources,
                "torque_nominal_pass": None if torque_summary is None else torque_summary["torque_nominal_pass"],
                "torque_identified_pass": None if torque_summary is None else torque_summary["torque_identified_pass"],
                "torque_corrected_pass": None if torque_summary is None else torque_summary["torque_corrected_pass"],
                "max_nominal_torque_ratio": None if torque_summary is None else torque_summary["max_nominal_torque_ratio"],
                "max_identified_torque_ratio": None if torque_summary is None else torque_summary["max_identified_torque_ratio"],
                "worst_joint": None if torque_summary is None else torque_summary["worst_joint"],
                "worst_time_s": None if torque_summary is None else torque_summary["worst_time_s"],
                "sequential_history": sequential_history,
            }
            if torque_summary is not None and torque_summary.get("design_summary") is not None:
                summary["torque_design_summary"] = torque_summary["design_summary"]
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            log.info("Summary saved to %s", summary_path)

            elapsed = time.time() - t0
            log.info("=" * 60)
            log.info("PIPELINE COMPLETED SUCCESSFULLY in %.1f s", elapsed)
            log.info("=" * 60)

        except Exception as e:
            log.error("PIPELINE FAILED at stage above.")
            log.error("Error: %s", str(e))
            log.debug(traceback.format_exc())
            raise
