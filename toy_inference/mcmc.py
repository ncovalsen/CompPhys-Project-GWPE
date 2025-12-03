"""Simple Metropolis–Hastings MCMC sampler for the toy GW project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

LogPosteriorFn = Callable[[np.ndarray], float]


@dataclass
class MCMCConfig:
    """Configuration parameters for the Metropolis–Hastings sampler."""

    n_steps: int = 20_000
    step_sizes: Optional[np.ndarray] = None
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.step_sizes is None:
            # Default: proposal widths for (mass1, mass2, spin1z, spin2z)
            self.step_sizes = np.array([2.0, 2.0, 0.05, 0.05], dtype=float)


@dataclass
class MCMCResult:
    """Result of an MCMC run."""

    chain: np.ndarray
    logp_chain: np.ndarray
    acceptance_rate: float


def run_mcmc(
    log_posterior: LogPosteriorFn,
    initial_theta: np.ndarray,
    config: Optional[MCMCConfig] = None,
) -> MCMCResult:
    """Run a basic Metropolis–Hastings MCMC chain."""
    if config is None:
        config = MCMCConfig()

    if config.random_seed is not None:
        rng = np.random.default_rng(config.random_seed)
    else:
        rng = np.random.default_rng()

    theta_current = np.asarray(initial_theta, dtype=float).copy()
    n_params = theta_current.size

    if config.step_sizes.shape != (n_params,):
        raise ValueError(
            f"step_sizes must have shape ({n_params},), got {config.step_sizes.shape}"
        )

    logp_current = float(log_posterior(theta_current))

    chain = np.zeros((config.n_steps, n_params), dtype=float)
    logp_chain = np.zeros(config.n_steps, dtype=float)

    accepted = 0

    for i in range(config.n_steps):
        proposal = theta_current + config.step_sizes * rng.standard_normal(n_params)
        logp_proposal = float(log_posterior(proposal))

        log_alpha = logp_proposal - logp_current

        if np.log(rng.random()) < log_alpha:
            theta_current = proposal
            logp_current = logp_proposal
            accepted += 1

        chain[i] = theta_current
        logp_chain[i] = logp_current

    acceptance_rate = accepted / float(config.n_steps)

    return MCMCResult(chain=chain, logp_chain=logp_chain, acceptance_rate=acceptance_rate)
