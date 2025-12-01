"""Simple Metropolis–Hastings MCMC sampler for the toy GW project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np


LogPosteriorFn = Callable[[np.ndarray], float]


@dataclass
class MCMCConfig:
    """Configuration parameters for the Metropolis–Hastings sampler.

    Attributes
    ----------
    n_steps :
        Total number of MCMC steps.
    step_sizes :
        1D array with proposal standard deviation for each parameter.
    random_seed :
        Optional seed for reproducible sampling.
    """

    n_steps: int = 20_000
    step_sizes: np.ndarray = None  # type: ignore[assignment]
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.step_sizes is None:
            # Default: modest jumps for (A, t0, f0)
            self.step_sizes = np.array([0.05, 0.001, 3.0], dtype=float)


@dataclass
class MCMCResult:
    """Result of an MCMC run.

    Attributes
    ----------
    chain :
        Array of shape ``(n_steps, n_params)`` with the parameter samples.
    logp_chain :
        Array of shape ``(n_steps,)`` with the log-posterior at each step.
    acceptance_rate :
        Fraction of proposed moves that were accepted.
    """

    chain: np.ndarray
    logp_chain: np.ndarray
    acceptance_rate: float


def run_mcmc(
    log_posterior: LogPosteriorFn,
    initial_theta: np.ndarray,
    config: Optional[MCMCConfig] = None,
) -> MCMCResult:
    """Run a basic Metropolis–Hastings MCMC chain.

    Parameters
    ----------
    log_posterior :
        Callable that takes a 1D parameter array and returns the log
        posterior value (up to a constant).
    initial_theta :
        1D array with the starting point in parameter space.
    config :
        :class:`MCMCConfig` instance.  If omitted, sensible defaults are
        used.

    Returns
    -------
    result :
        :class:`MCMCResult` instance with the full chain and diagnostics.
    """
    import numpy as _np

    if config is None:
        config = MCMCConfig()

    if config.random_seed is not None:
        rng = _np.random.default_rng(config.random_seed)
    else:
        rng = _np.random.default_rng()

    theta_current = _np.asarray(initial_theta, dtype=float).copy()
    n_params = theta_current.size

    if config.step_sizes.shape != (n_params,):
        raise ValueError(
            f"step_sizes must have shape ({n_params},), got {config.step_sizes.shape}"
        )

    logp_current = float(log_posterior(theta_current))

    chain = _np.zeros((config.n_steps, n_params), dtype=float)
    logp_chain = _np.zeros(config.n_steps, dtype=float)

    accepted = 0

    for i in range(config.n_steps):
        # Propose a Gaussian jump.
        proposal = theta_current + config.step_sizes * rng.standard_normal(n_params)
        logp_proposal = float(log_posterior(proposal))

        # Compute MH acceptance probability in log-space.
        log_alpha = logp_proposal - logp_current

        if _np.log(rng.random()) < log_alpha:
            theta_current = proposal
            logp_current = logp_proposal
            accepted += 1

        chain[i] = theta_current
        logp_chain[i] = logp_current

    acceptance_rate = accepted / float(config.n_steps)

    return MCMCResult(chain=chain, logp_chain=logp_chain, acceptance_rate=acceptance_rate)