"""Entry point script for the sine-Gaussian toy inference.

This script:
1. Generates synthetic data with a known injected sine-Gaussian signal.
2. Runs a Metropolis–Hastings MCMC sampler to infer the parameters
   (A, t0, f0) of the signal.
3. Saves the raw chain and summary statistics to the ``results/``
   directory.
4. Produces diagnostic plots in ``results/``:
   - Time series with best-fit model overlaid.
   - 1D marginal posterior histograms for each parameter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from toy_inference.config import ToyModelConfig
from toy_inference.data import simulate_sine_gaussian_data, save_simulated_data
from toy_inference.likelihood import log_posterior
from toy_inference.mcmc import run_mcmc
from toy_inference.plotting import plot_data_and_model, plot_1d_posteriors
from toy_inference.waveform import sine_gaussian


# Project-level directories (relative to the repository root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_directories() -> None:
    """Create the standard ``data/`` and ``results/`` directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()

    # Configuration for this run.
    cfg = ToyModelConfig()

    # 1. Generate synthetic data and save it under data/
    sim_data = simulate_sine_gaussian_data(
        duration=0.5,
        fs=2048.0,
        noise_sigma=0.1,
        A=1.0,
        t0=0.25,
        f0=150.0,
        tau=cfg.tau,
        phi0=cfg.phi0,
        random_seed=42,
    )

    data_path = DATA_DIR / "sine_gaussian_injection.npz"
    save_simulated_data(sim_data, data_path)

    print(f"Synthetic data saved to: {data_path}")
    print("True injected parameters (A, t0, f0):", sim_data.true_params)

    # 2. Build a closure for the log-posterior that only depends on theta.
    def log_post_closure(theta: np.ndarray) -> float:
        return log_posterior(
            theta,
            t=sim_data.t,
            d=sim_data.d,
            sigma=sim_data.sigma,
            bounds=cfg.prior_bounds,
            tau=cfg.tau,
            phi0=cfg.phi0,
        )

    # Choose a sensible starting point for the chain.
    initial_theta = np.array([0.8, 0.24, 120.0])

    # 3. Run MCMC
    mcmc_result = run_mcmc(
        log_posterior=log_post_closure,
        initial_theta=initial_theta,
        config=cfg.mcmc,
    )

    print(f"MCMC finished. Acceptance rate: {mcmc_result.acceptance_rate:.3f}")

    # Discard burn-in and keep the remainder as posterior samples.
    samples = mcmc_result.chain[cfg.burn_in :]

    # Compute a simple posterior mean as a crude "best-fit" estimate.
    theta_mean = samples.mean(axis=0)
    A_mean, t0_mean, f0_mean = theta_mean
    print("Posterior mean parameters (A, t0, f0):", theta_mean)

    # 4. Save chain and summary statistics to results/
    chain_path = RESULTS_DIR / "mcmc_chain.npy"
    logp_path = RESULTS_DIR / "mcmc_logp.npy"
    summary_path = RESULTS_DIR / "summary.json"

    np.save(chain_path, mcmc_result.chain)
    np.save(logp_path, mcmc_result.logp_chain)

    summary: Dict[str, Any] = {
        "true_parameters": {
            "A": sim_data.true_params.A,
            "t0": sim_data.true_params.t0,
            "f0": sim_data.true_params.f0,
        },
        "posterior_mean": {
            "A": float(A_mean),
            "t0": float(t0_mean),
            "f0": float(f0_mean),
        },
        "acceptance_rate": mcmc_result.acceptance_rate,
        "burn_in": cfg.burn_in,
        "n_steps": cfg.mcmc.n_steps,
        "step_sizes": cfg.mcmc.step_sizes.tolist(),
        "data_file": str(data_path),
        "chain_file": str(chain_path),
        "logp_file": str(logp_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved MCMC chain to: {chain_path}")
    print(f"Saved log-posterior values to: {logp_path}")
    print(f"Saved summary JSON to: {summary_path}")

    # 5. Make plots in results/
    best_fit = sine_gaussian(
        sim_data.t,
        A=A_mean,
        t0=t0_mean,
        f0=f0_mean,
        tau=cfg.tau,
        phi0=cfg.phi0,
    )

    ts_plot_path = RESULTS_DIR / "timeseries_with_bestfit.png"
    plot_data_and_model(
        sim_data.t,
        sim_data.d,
        best_fit,
        title="Synthetic data with best-fit sine-Gaussian",
        outfile=ts_plot_path,
    )
    print(f"Saved time-series plot to: {ts_plot_path}")

    labels = ["A", "t0 [s]", "f0 [Hz]"]
    truths = [sim_data.true_params.A, sim_data.true_params.t0, sim_data.true_params.f0]
    post_plot_path = RESULTS_DIR / "posterior_1d_histograms.png"
    plot_1d_posteriors(
        samples,
        labels=labels,
        truths=truths,
        outfile=post_plot_path,
    )
    print(f"Saved 1D posterior histograms to: {post_plot_path}")


if __name__ == "__main__":
    main()