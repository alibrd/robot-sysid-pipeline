"""Minimal plotting for the unified runner. Reads pipeline artifacts only."""
from pathlib import Path

import numpy as np


def plot_excitation_outputs(pipeline_dir: Path, output_dir: Path, plot_cfg: dict) -> dict:
    """Render excitation_trajectory.png from a pipeline output directory.

    Reads <pipeline_dir>/excitation_trajectory.npz and writes a single PNG
    figure with three stacked rows (q, dq, ddq) and one curve per DoF.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exc_path = Path(pipeline_dir) / "excitation_trajectory.npz"
    if not exc_path.exists():
        raise FileNotFoundError(f"Cannot plot: missing {exc_path}")

    with np.load(str(exc_path), allow_pickle=True) as data:
        t = data["t"]
        q = data["q"]
        dq = data["dq"]
        ddq = data["ddq"]

    nDoF = q.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for j in range(nDoF):
        axes[0].plot(t, q[j], label=f"q{j + 1}")
        axes[1].plot(t, dq[j], label=f"dq{j + 1}")
        axes[2].plot(t, ddq[j], label=f"ddq{j + 1}")

    axes[0].set_ylabel("q [rad]")
    axes[1].set_ylabel("dq [rad/s]")
    axes[2].set_ylabel("ddq [rad/s²]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="upper right", ncol=min(nDoF, 4), fontsize="small")

    fig.tight_layout()

    fmt = plot_cfg.get("format", "png")
    dpi = int(plot_cfg.get("dpi", 150))
    out_file = output_dir / f"excitation_trajectory.{fmt}"
    fig.savefig(str(out_file), dpi=dpi)

    plt.close(fig)

    return {"excitation_plot": str(out_file)}
