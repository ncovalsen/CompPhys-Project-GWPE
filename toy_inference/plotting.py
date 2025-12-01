"""Plotting utilities for the toy GW inference project."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_data_and_model(
    t: np.ndarray,
    data: np.ndarray,
    model: np.ndarray,
    title: str = "Synthetic data and best-fit model",
    outfile: Optional[Path] = None,
) -> None:
    """Plot the time-series data and a model waveform.

    If ``outfile`` is provided, the plot is saved to disk and the figure
    is closed.  Otherwise, the plot is shown interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, data, label="data", linewidth=1.0, alpha=0.7)
    ax.plot(t, model, label="best-fit model", linewidth=2.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Strain (arb. units)")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_1d_posteriors(
    samples: np.ndarray,
    labels: Sequence[str],
    truths: Optional[Sequence[float]] = None,
    outfile: Optional[Path] = None,
    bins: int = 50,
) -> None:
    """Plot simple 1D marginal posterior histograms for each parameter.

    Parameters
    ----------
    samples :
        2D array of shape ``(n_samples, n_params)``.
    labels :
        Sequence of parameter labels for the x-axes.
    truths :
        Optional sequence of true values to mark with vertical lines.
    outfile :
        If given, save the figure to this path instead of showing it.
    bins :
        Number of histogram bins.
    """
    samples = np.asarray(samples)
    n_params = samples.shape[1]

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3))

    if n_params == 1:
        axes = [axes]  # make iterable

    for i, ax in enumerate(axes):
        ax.hist(samples[:, i], bins=bins, density=True, histtype="stepfilled", alpha=0.7)
        ax.set_xlabel(labels[i])
        if truths is not None:
            ax.axvline(truths[i], linestyle="--", linewidth=2.0, label="truth")
            ax.legend()

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
    else:
        plt.show()