"""Configuration helpers for the toy GW inference project."""

from __future__ import annotations

from dataclasses import dataclass, field

from .likelihood import PriorBounds, DEFAULT_PRIOR
from .mcmc import MCMCConfig


@dataclass
class ToyModelConfig:
    """Bundle together common configuration for the toy model run.

    Attributes
    ----------
    prior_bounds :
        Prior intervals for (A, t0, f0).
    mcmc :
        MCMC configuration (number of steps, proposal widths, seed).
    burn_in :
        Number of initial MCMC samples to discard as burn-in.
    tau :
        Width of the sine-Gaussian envelope in seconds.
    phi0 :
        Initial phase of the sine-Gaussian in radians.
    """

    prior_bounds: PriorBounds = field(
        default_factory=lambda: PriorBounds(
            A=DEFAULT_PRIOR.A,
            t0=DEFAULT_PRIOR.t0,
            f0=DEFAULT_PRIOR.f0,
        )
    )
    mcmc: MCMCConfig = field(default_factory=lambda: MCMCConfig(n_steps=30_000))
    burn_in: int = 5_000
    tau: float = 0.02
    phi0: float = 0.0