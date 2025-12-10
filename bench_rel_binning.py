"""
bench_relative_binning.py

Compare runtime of standard likelihood vs relative-binning likelihood
for the same GW data set and the same MCMC configuration.

Run from the project root:

    python bench_relative_binning.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from toy_inference.data import simulate_gw_data
from toy_inference.config import ToyModelConfig
from toy_inference.mcmc import run_mcmc
from toy_inference.likelihood import (
    log_posterior,
    initialize_relative_binning,
)
import toy_inference.likelihood as lk_mod  # to flip USE_RELATIVE_BINNING


def make_log_posterior(d, sigma, delta_t, f_lower, bounds):
    """
    Wrap toy_inference.likelihood.log_posterior so we only have to
    pass theta when sampling.
    """
    def _log_post(theta):
        return log_posterior(
            theta,
            d=d,
            sigma=sigma,
            delta_t=delta_t,
            f_lower=f_lower,
            bounds=bounds,
        )
    return _log_post


def main():
    # ------------------------------------------------------------------
    # 1. Make a single GW-like data set
    # ------------------------------------------------------------------
    sim = simulate_gw_data(
        fs=1024.0,
        noise_sigma=1e-22,
        mass1=15.0,
        mass2=10.0,
        spin1z=0.1,
        spin2z=-0.1,
        f_lower=30.0,
        random_seed=0,
    )

    # Model configuration (prior + MCMC settings)
    cfg = ToyModelConfig()

    # Build a reasonable starting point in (mc, q, spin1z, spin2z) space:
    pb = cfg.prior_bounds
    theta0 = np.array(
        [
            0.5 * (pb.mc[0] + pb.mc[1]),
            0.5 * (pb.q[0] + pb.q[1]),
            0.5 * (pb.spin1z[0] + pb.spin1z[1]),
            0.5 * (pb.spin2z[0] + pb.spin2z[1]),
        ]
    )

    log_post = make_log_posterior(
        d=sim.d,
        sigma=sim.sigma,
        delta_t=sim.delta_t,
        f_lower=sim.true_params.f_lower,
        bounds=cfg.prior_bounds,
    )

    # We'll compare timings for several MCMC lengths.
    n_steps_grid = np.array([500, 1_000, 2_000, 4_000])
    speedups = []
    times_normal = []
    times_rb = []

    for n_steps in n_steps_grid:
        cfg.mcmc.n_steps = int(n_steps)
        cfg.mcmc.random_seed = 123  # keep the proposal sequence fixed

        # --------------------------------------------------------------
        # 2a. Standard likelihood timing
        # --------------------------------------------------------------
        lk_mod.USE_RELATIVE_BINNING = False

        t0 = time.perf_counter()
        _ = run_mcmc(log_post, initial_theta=theta0, config=cfg.mcmc)
        t_normal = time.perf_counter() - t0
        times_normal.append(t_normal)

        # --------------------------------------------------------------
        # 2b. Relative-binning likelihood timing
        # --------------------------------------------------------------
        # Build / refresh RB summary state *once* for this data set.
        initialize_relative_binning(
            d=sim.d,
            sigma=sim.sigma,
            delta_t=sim.delta_t,
            f_lower=sim.true_params.f_lower,
        )
        lk_mod.USE_RELATIVE_BINNING = True

        t0 = time.perf_counter()
        _ = run_mcmc(log_post, initial_theta=theta0, config=cfg.mcmc)
        t_rb = time.perf_counter() - t0
        times_rb.append(t_rb)

        # Record speedup factor
        speedups.append(t_normal / t_rb)

        # Reset flag for safety
        lk_mod.USE_RELATIVE_BINNING = False

        print(
            f"n_steps={n_steps:5d}  "
            f"normal={t_normal:7.3f}s  RB={t_rb:7.3f}s  "
            f"speedup={t_normal/t_rb:5.1f}x"
        )

    speedups = np.array(speedups)

    # ------------------------------------------------------------------
    # 3. Make a plot of the speedup
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(n_steps_grid, speedups, marker="o")
    plt.xscale("log")
    plt.xlabel("Number of MCMC steps")
    plt.ylabel("Speedup factor  (t_normal / t_RB)")
    plt.title("Relative binning speedup vs MCMC chain length")
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
