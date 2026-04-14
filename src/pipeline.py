"""Main pipeline orchestrator: loads config, runs all stages, writes output."""
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np

from .config_loader import load_config, load_config_dict
from .pipeline_logger import setup_logger
from .urdf_parser import parse_urdf, extract_joint_limits
from .kinematics import RobotKinematics
from .dynamics_newton_euler import newton_euler_regressor
from .dynamics_euler_lagrange import euler_lagrange_regressor_builder
from .friction import friction_param_count, friction_param_names
from .trajectory import fourier_trajectory, build_frequencies
from .excitation import optimise_excitation
from .observation_matrix import build_observation_matrix
from .base_parameters import compute_base_parameters
from .solver import solve_identification
from .feasibility import check_feasibility


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

            # -- Stage 1: Parse URDF ----------------------------------------
            log.info("-- Stage 1: Parsing URDF --")
            robot = parse_urdf(cfg["urdf_path"])
            log.info("Robot: '%s', %d DoF, revolute joints: %s",
                     robot.name, robot.nDoF, robot.revolute_joint_names)

            # -- Stage 2: Extract joint limits -------------------------------
            log.info("-- Stage 2: Extracting joint limits --")
            q_lim, dq_lim, ddq_lim = extract_joint_limits(
                robot, cfg["joint_limits"], log
            )

            # -- Stage 3: Build kinematics -----------------------------------
            log.info("-- Stage 3: Building kinematics --")
            kin = RobotKinematics(robot, log)
            log.info("Initial parameter vector (10n=%d): %s",
                     len(kin.PI), kin.PI.flatten()[:6].tolist())

            # -- Stage 4: Setup regressor function ---------------------------
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

            # -- Stage 5: Excitation trajectory optimisation -----------------
            log.info("-- Stage 5: Excitation trajectory optimisation --")
            exc_result = optimise_excitation(
                kin, cfg["excitation"], q_lim, dq_lim, ddq_lim,
                friction_model=cfg["friction"]["model"],
                regressor_fn=regressor_fn if cfg["method"] == "euler_lagrange" else None,
            )
            log.info("Excitation cost: %.6f", exc_result["cost"])

            # Save excitation result
            exc_path = self.output_dir / "excitation_trajectory.npz"
            np.savez(str(exc_path), **exc_result)
            log.info("Saved excitation parameters to %s", exc_path)

            # -- Stage 6: Generate trajectory data ---------------------------
            log.info("-- Stage 6: Generating trajectory data --")
            data_file = cfg["identification"].get("data_file")
            if data_file:
                log.info("Loading external data from %s", data_file)
                data = np.load(data_file)
                q_data = data["q"]
                dq_data = data["dq"]
                ddq_data = data["ddq"]
                tau_data = data["tau"]
                data_fs = float(data.get("fs", 1e4))
            else:
                log.info("Generating synthetic trajectory from excitation parameters.")
                freqs = exc_result["freqs"]
                q0 = exc_result["q0"]
                basis = exc_result["basis"]
                opt_phase = exc_result["optimize_phase"]
                params = exc_result["params"]

                f0 = cfg["excitation"]["base_frequency_hz"]
                tf = cfg["excitation"].get("trajectory_duration_periods", 1) / f0
                data_fs = 2.0 * freqs[-1] * 10  # 10x Nyquist
                t_data = np.arange(0, tf, 1.0 / data_fs)
                q_data_t, dq_data_t, ddq_data_t = fourier_trajectory(
                    params, freqs, t_data, q0, basis, opt_phase
                )
                q_data = q_data_t.T   # (N, nDoF)
                dq_data = dq_data_t.T
                ddq_data = ddq_data_t.T

                # Compute torques from regressor
                N = q_data.shape[0]
                nDoF = kin.nDoF
                tau_data = np.zeros((N, nDoF))
                for k in range(N):
                    Y = regressor_fn(q_data[k], dq_data[k], ddq_data[k])
                    if cfg["method"] == "euler_lagrange" and el_kept_cols is not None:
                        pi_used = kin.PI.flatten()[el_kept_cols]
                    else:
                        pi_used = kin.PI.flatten()
                    # Friction contribution
                    if cfg["friction"]["model"] != "none":
                        from .friction import build_friction_regressor
                        Yf = build_friction_regressor(dq_data[k], cfg["friction"]["model"])
                        n_fric = Yf.shape[1]
                        Y = np.hstack((Y, Yf))
                        pi_fric = np.zeros(n_fric)  # true friction = 0 for synthetic
                        pi_used = np.concatenate([pi_used, pi_fric])
                    tau_data[k] = Y @ pi_used

                log.info("Generated %d data samples at %.1f Hz", N, data_fs)

            # -- Stage 7: Build observation matrix ---------------------------
            log.info("-- Stage 7: Building observation matrix --")
            W, tau_vec = build_observation_matrix(
                q_data, dq_data, ddq_data, tau_data,
                regressor_fn, cfg, data_fs
            )
            log.info("W shape: %s", W.shape)

            # -- Stage 8: Base parameter reduction ---------------------------
            log.info("-- Stage 8: Base parameter reduction --")
            if cfg["method"] == "euler_lagrange" and el_kept_cols is not None:
                pi_full = kin.PI.flatten()[el_kept_cols]
            else:
                pi_full = kin.PI.flatten()
            # Append friction params (zeros as nominal)
            n_fric = friction_param_count(kin.nDoF, cfg["friction"]["model"])
            if n_fric > 0:
                pi_full = np.concatenate([pi_full, np.zeros(n_fric)])

            W_base, P_mat, kept_cols, rank, pi_base = compute_base_parameters(
                W, pi_full
            )
            log.info("Base parameters: %d (from %d full)", rank, len(pi_full))

            # -- Stage 9: Solve identification -------------------------------
            log.info("-- Stage 9: Solving identification --")
            solver = cfg["identification"]["solver"]
            feas_method = cfg["identification"]["feasibility_method"]

            # User-specified bounds from config
            bounds_opt = None
            cfg_bounds = cfg["identification"].get("parameter_bounds")
            if isinstance(cfg_bounds, list) and len(cfg_bounds) == 2:
                # Expect [[lb_0, lb_1, ...], [ub_0, ub_1, ...]]
                lb = np.array(cfg_bounds[0])
                ub = np.array(cfg_bounds[1])
                if len(lb) == rank and len(ub) == rank:
                    bounds_opt = (lb, ub)
                    if solver == "ols":
                        solver = "bounded_ls"
                        log.info("Switching to bounded_ls due to user parameter_bounds.")
            elif cfg_bounds is True:
                # Auto-generate bounds from base parameter magnitudes
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

            # Map back to full parameter space
            if info.get("solved_in_full_space"):
                # Constrained solver already returned full-space params
                pi_identified_full = pi_hat
            else:
                # Map base params to one representative full vector via pinv(P)
                # NOTE: this is not unique — it is the minimum-norm solution
                # satisfying P @ pi_full = pi_base.
                pi_identified_full = np.linalg.pinv(P_mat) @ pi_hat

            # Verify reconstruction consistency
            recon_err = np.linalg.norm(
                W_base @ (P_mat @ pi_identified_full) - W_base @ pi_hat
                if not info.get("solved_in_full_space")
                else W_base @ P_mat @ pi_identified_full - tau_vec
            )
            log.debug("Reconstruction consistency: ||W_b*P*pi_recon - ref|| = %.3e",
                       recon_err)

            # -- Stage 10: Feasibility check ---------------------------------
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

            # -- Stage 11: Save results --------------------------------------
            log.info("-- Stage 11: Saving results --")
            results = {
                "pi_identified": pi_identified_full,  # representative full vector (not unique)
                "pi_base": pi_hat if not info.get("solved_in_full_space") else P_mat @ pi_identified_full,
                "P_matrix": P_mat,
                "kept_cols": np.array(kept_cols),
                "residual": residual,
                "feasible": feasible,
                "pi_corrected": pi_corrected,
                "method": cfg["method"],
                "friction_model": cfg["friction"]["model"],
                "nDoF": kin.nDoF,
                "solved_in_full_space": info.get("solved_in_full_space", False),
            }
            out_path = self.output_dir / "identification_results.npz"
            np.savez(str(out_path), **results)
            log.info("Results saved to %s", out_path)

            # Save human-readable summary
            summary_path = self.output_dir / "results_summary.json"
            summary = {
                "method": cfg["method"],
                "nDoF": kin.nDoF,
                "n_base_params": rank,
                "n_full_params": len(pi_full),
                "residual": float(residual),
                "feasible": bool(feasible),
                "friction_model": cfg["friction"]["model"],
                "n_friction_params": n_fric,
                "solver": solver,
                "pi_identified": pi_identified_full.tolist(),
            }
            with open(summary_path, "w") as f:
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
