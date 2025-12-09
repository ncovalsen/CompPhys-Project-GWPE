"""Entry point script for the GW toy inference.

This script:
1. Generates synthetic data with a known injected GW signal.
2. Runs a Metropolis–Hastings MCMC sampler to infer the parameters
   (mass1, mass2, spin1z, spin2z) of the signal.
2.5. Option to toggle between normal and relative binning
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
from toy_inference.data import simulate_gw_data, save_simulated_data
from toy_inference.likelihood import log_posterior, USE_RELATIVE_BINNING, initialize_relative_binning
from toy_inference.mcmc import run_mcmc
from toy_inference.plotting import plot_data_and_model, plot_1d_posteriors
from toy_inference.waveform import gw_waveform
from toy_inference.utils import mc_q_to_masses

# Project-level directories (relative to the repository root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()

    cfg = ToyModelConfig()

    # 1. Generate synthetic GW data and save it under data/
    sim_data = simulate_gw_data(
        fs=2048.0,
        noise_sigma=1e-22,
        mass1=10.0,
        mass2=10.0,
        spin1z=0.9,
        spin2z=0.0,
        f_lower=40.0,
        random_seed=42,
    )

    data_path = DATA_DIR / "gw_injection.npz"
    save_simulated_data(sim_data, data_path)

    tp = sim_data.true_params
    print(f"Synthetic data saved to: {data_path}")
    print(
        "True injected parameters (mass1, mass2, spin1z, spin2z):",
        (tp.mass1, tp.mass2, tp.spin1z, tp.spin2z),
    )

    # 2. Build a closure for the log-posterior that only depends on theta.
    def log_post_closure(theta: np.ndarray) -> float:
        return log_posterior(
            theta,
            d=sim_data.d,
            sigma=sim_data.sigma,
            delta_t=sim_data.delta_t,
            f_lower=tp.f_lower,
            bounds=cfg.prior_bounds,
        )

    # Choose a sensible starting point for the chain.
    # compute true Mc and q
    true_m1, true_m2 = tp.mass1, tp.mass2
    q_true = true_m1 / true_m2
    mc_true = (true_m1*true_m2)**(3/5) / (true_m1 + true_m2)**(1/5)

    initial_theta = np.array(
    [mc_true, q_true, tp.spin1z, tp.spin2z])

    # 2.5: Enable or disable Relative Binning
    # USE_RELATIVE_BINNING = False   # Use standard likelihood
    USE_RELATIVE_BINNING = True      # Use relative-binning likelihood

    if USE_RELATIVE_BINNING:
        print("Initializing relative binning...")
        initialize_relative_binning(
            d=sim_data.d,
            sigma=sim_data.sigma,
            delta_t=sim_data.delta_t,
            f_lower=tp.f_lower,
        )


    # 3. Run MCMC
    mcmc_result = run_mcmc(
        log_posterior=log_post_closure,
        initial_theta=initial_theta,
        config=cfg.mcmc,
    )

    print(f"MCMC finished. Acceptance rate: {mcmc_result.acceptance_rate:.3f}")

    # Discard burn-in and keep the remainder as posterior samples.
    samples = mcmc_result.chain[cfg.burn_in :]

    # Posterior mean as a crude "best-fit" estimate.
    theta_mean = samples.mean(axis=0)
    mc_mean, q_mean, spin1z_mean, spin2z_mean = theta_mean
    mass1_mean, mass2_mean = mc_q_to_masses(mc_mean, q_mean)

    print("Posterior mean parameters:", theta_mean)

    # 4. Save chain and summary statistics to results/
    chain_path = RESULTS_DIR / "mcmc_chain.npy"
    logp_path = RESULTS_DIR / "mcmc_logp.npy"
    summary_path = RESULTS_DIR / "summary.json"

    np.save(chain_path, mcmc_result.chain)
    np.save(logp_path, mcmc_result.logp_chain)

    summary: Dict[str, Any] = {
        "true_parameters": {
            "mass1": tp.mass1,
            "mass2": tp.mass2,
            "spin1z": tp.spin1z,
            "spin2z": tp.spin2z,
            "f_lower": tp.f_lower,
        },
        "posterior_mean": {
            "mass1": float(mass1_mean),
            "mass2": float(mass2_mean),
            "spin1z": float(spin1z_mean),
            "spin2z": float(spin2z_mean),
        },
        "acceptance_rate": mcmc_result.acceptance_rate,
        "burn_in": cfg.burn_in,
        "n_steps": cfg.mcmc.n_steps,
        "step_sizes": cfg.mcmc.step_sizes.tolist(),
        "data_file": str(data_path),
        "chain_file": str(chain_path),
        "logp_file": str(logp_path),
        "sigma": sim_data.sigma,
        "delta_t": sim_data.delta_t,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved MCMC chain to: {chain_path}")
    print(f"Saved log-posterior values to: {logp_path}")
    print(f"Saved summary JSON to: {summary_path}")

    # 5. Make plots in results/
    t_model, best_fit = gw_waveform(
        mass1=mass1_mean,
        mass2=mass2_mean,
        spin1z=spin1z_mean,
        spin2z=spin2z_mean,
        delta_t=sim_data.delta_t,
        f_lower=tp.f_lower,
    )

    # Align model to data length for plotting
    n = min(len(sim_data.t), len(t_model), len(best_fit))
    ts_plot_path = RESULTS_DIR / "timeseries_with_bestfit.png"
    plot_data_and_model(
        sim_data.t[:n],
        sim_data.d[:n],
        best_fit[:n],
        title="Synthetic GW data with best-fit model",
        outfile=ts_plot_path,
    )
    print(f"Saved time-series plot to: {ts_plot_path}")

    labels = [
    r"$\mathcal{M}_c\ [M_\odot]$",
    r"$q$",
    r"$\chi_{1z}$",
    r"$\chi_{2z}$",]
    truths = [mc_true, q_true, tp.spin1z, tp.spin2z]
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
